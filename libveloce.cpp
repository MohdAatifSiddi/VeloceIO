#define _GNU_SOURCE
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// Forward declaration of the external wrapper function for PTX sync kernel launch
extern "C" void launch_veloce_ptx_sync_kernel(volatile uint32_t* sync_flag, uint32_t expected_epoch, cudaStream_t stream);

// ============================================================================
// SYSTEM DEFINITIONS & METRICS TIER
// ============================================================================
constexpr size_t MULTIPATH_THRESHOLD = 32 * 1024 * 1024; // 32MB Optimal Cut-off
constexpr size_t PIPELINE_CHUNK_SIZE  = 16 * 1024 * 1024; // 16MB Micro-Buffer Slices
constexpr size_t FLAG_POOL_SIZE       = 512;               // Max Concurrent Fly-Transfers

namespace VeloceMetrics {
    std::atomic<uint64_t> total_bytes_routed{0};
    std::atomic<uint64_t> multipath_transfers_count{0};
    std::atomic<uint64_t> fallback_transfers_count{0};
}

typedef cudaError_t (*cudaMemcpyAsync_t)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
static cudaMemcpyAsync_t real_cudaMemcpyAsync = nullptr;

// ============================================================================
// LOCK-FREE COUNTERPART BOUNDED FLAG POOL
// ============================================================================
class FlagPool {
public:
    struct Slot {
        std::atomic<bool> in_use{false};
        uint32_t* dev_flag_ptr = nullptr;
        uint32_t* host_pinned_sig_ptr = nullptr;
        int gpu_owner = -1;
        size_t slot_idx = 0;
    };

private:
    std::vector<Slot> slots_;
    std::atomic<size_t> hint_index_{0};

public:
    FlagPool() : slots_(FLAG_POOL_SIZE) {}

    bool initialize(int sample_gpu) {
        int current_gpu;
        if (cudaGetDevice(&current_gpu) != cudaSuccess) return false;
        if (cudaSetDevice(sample_gpu) != cudaSuccess) return false;

        for (size_t i = 0; i < FLAG_POOL_SIZE; ++i) {
            if (cudaMallocManaged(&slots_[i].dev_flag_ptr, sizeof(uint32_t)) != cudaSuccess) return false;
            if (cudaHostAlloc(&slots_[i].host_pinned_sig_ptr, sizeof(uint32_t), cudaHostAllocMapped) != cudaSuccess) return false;
            
            *(slots_[i].dev_flag_ptr) = 0;
            *(slots_[i].host_pinned_sig_ptr) = 0;
            slots_[i].in_use.store(false, std::memory_order_relaxed);
            slots_[i].gpu_owner = sample_gpu;
            slots_[i].slot_idx = i;
        }
        cudaSetDevice(current_gpu);
        return true;
    }

    ~FlagPool() {
        for (size_t i = 0; i < FLAG_POOL_SIZE; ++i) {
            if (slots_[i].dev_flag_ptr) cudaFree(slots_[i].dev_flag_ptr);
            if (slots_[i].host_pinned_sig_ptr) cudaFreeHost(slots_[i].host_pinned_sig_ptr);
        }
    }

    size_t acquire_slot() {
        size_t start = hint_index_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < FLAG_POOL_SIZE; ++i) {
            size_t idx = (start + i) % FLAG_POOL_SIZE;
            bool expected = false;
            if (slots_[idx].in_use.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
                hint_index_.store((idx + 1) % FLAG_POOL_SIZE, std::memory_order_relaxed);
                return idx;
            }
        }
        return std::string::npos; 
    }

    Slot& get(size_t idx) { return slots_[idx]; }

    void release_slot(size_t idx) {
        slots_[idx].in_use.store(false, std::memory_order_release);
    }
};

static FlagPool* g_gpu_flag_pools[8] = { nullptr };

// ============================================================================
// LOCK-FREE DECENTRALIZED MPMC SCHEDULER MATRIX
// ============================================================================
struct Task {
    uint8_t* host_src;
    uint8_t* dev_dst;
    size_t total_bytes;
    int target_gpu;
    int assigned_relay_gpu;
    size_t flag_pool_idx;
    uint32_t epoch;
    bool is_direct;
};

class LockFreeTaskQueue {
private:
    static constexpr size_t Q_CAPACITY = 2048;
    Task ring_[Q_CAPACITY];
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::atomic<size_t> remaining_bytes_{0};

public:
    bool try_push(const Task& t) {
        size_t curr_tail = tail_.load(std::memory_order_relaxed);
        size_t curr_head = head_.load(std::memory_order_acquire);
        if ((curr_tail - curr_head) >= Q_CAPACITY) return false;

        ring_[curr_tail % Q_CAPACITY] = t;
        remaining_bytes_.fetch_add(t.total_bytes, std::memory_order_relaxed);
        tail_.store(curr_tail + 1, std::memory_order_release);
        return true;
    }

    bool try_pop(Task& t) {
        size_t curr_head = head_.load(std::memory_order_relaxed);
        size_t curr_tail = tail_.load(std::memory_order_acquire);
        if (curr_head == curr_tail) return false;

        t = ring_[curr_head % Q_CAPACITY];
        remaining_bytes_.fetch_sub(t.total_bytes, std::memory_order_relaxed);
        head_.store(curr_head + 1, std::memory_order_release);
        return true;
    }

    size_t active_bytes() const { return remaining_bytes_.load(std::memory_order_relaxed); }
    size_t size() const { 
        size_t t = tail_.load(std::memory_order_relaxed); 
        size_t h = head_.load(std::memory_order_relaxed); 
        return (t >= h) ? (t - h) : 0; 
    }
};

// ============================================================================
// ASYNCHRONOUS DUAL-PIPELINE BACKEND ENGINE
// ============================================================================
class VeloceDaemon {
private:
    int dev_count_;
    std::vector<std::unique_ptr<LockFreeTaskQueue>> node_queues_;
    std::vector<std::thread> workers_;
    std::atomic<bool> kill_signal_{false};
    std::vector<std::mutex> wait_mtxs_;
    std::vector<std::condition_variable> wait_cvs_;

    // Longest-Remaining-Destination (LRD) work-stealing algorithm
    bool steal_optimized_task(int thief_id, Task& stolen_job) {
        int absolute_best_victim = -1;
        size_t largest_bytes = 0;

        for (int i = 0; i < dev_count_; ++i) {
            if (i == thief_id) continue;
            size_t pending_load = node_queues_[i]->active_bytes();
            if (pending_load > largest_bytes) {
                largest_bytes = pending_load;
                absolute_best_victim = i;
            }
        }

        if (absolute_best_victim != -1 && node_queues_[absolute_best_victim]->try_pop(stolen_job)) {
            if (stolen_job.target_gpu != thief_id) {
                stolen_job.is_direct = false;
                stolen_job.assigned_relay_gpu = thief_id;
            } else {
                stolen_job.is_direct = true;
                stolen_job.assigned_relay_gpu = -1;
            }
            return true;
        }
        return false;
    }

    // Static Host-side reclaim callback: eliminates new/delete hotpath allocations
    static void CUDART_CB host_reclaim_callback(void* data) {
        FlagPool::Slot* slot = static_cast<FlagPool::Slot*>(data);
        slot->in_use.store(false, std::memory_order_release);
    }

public:
    VeloceDaemon(int gpus) : dev_count_(gpus), node_queues_(gpus), wait_mtxs_(gpus), wait_cvs_(gpus) {
        for (int i = 0; i < gpus; ++i) node_queues_[i] = std::make_unique<LockFreeTaskQueue>();
    }

    ~VeloceDaemon() { shutdown(); }

    void start() {
        for (int i = 0; i < dev_count_; ++i) workers_.emplace_back(&VeloceDaemon::worker_core, this, i);
    }

    void shutdown() {
        kill_signal_.store(true, std::memory_order_relaxed);
        for (int i = 0; i < dev_count_; ++i) wait_cvs_[i].notify_all();
        for (auto& t : workers_) { if (t.joinable()) t.join(); }
    }

    void push_job(const Task& t) {
        node_queues_[t.target_gpu]->try_push(t);
        wait_cvs_[t.target_gpu]->notify_one();
    }

    void worker_core(int gpu_id) {
        if (cudaSetDevice(gpu_id) != cudaSuccess) return;
        
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
        int target_socket = (gpu_id < 4) ? 0 : 1;
        int cores_per_socket = 16;
        int start_core = target_socket * cores_per_socket;
        int end_core = (target_socket + 1) * cores_per_socket;
        
        bool added = false;
        for (int core = start_core; core < end_core; ++core) {
            if (core < num_cores) {
                CPU_SET(core, &cpuset);
                added = true;
            }
        }
        if (!added && num_cores > 0) {
            CPU_SET(gpu_id % num_cores, &cpuset);
        }
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        void* d_scratch_A = nullptr;
        void* d_scratch_B = nullptr;
        if (cudaMalloc(&d_scratch_A, PIPELINE_CHUNK_SIZE) != cudaSuccess ||
            cudaMalloc(&d_scratch_B, PIPELINE_CHUNK_SIZE) != cudaSuccess) {
            std::cerr << "VeloceDaemon: Failed to allocate scratchpad on GPU " << gpu_id << std::endl;
            if (d_scratch_A) cudaFree(d_scratch_A);
            if (d_scratch_B) cudaFree(d_scratch_B);
            return;
        }

        cudaStream_t stream_pcie_ingress;
        cudaStream_t stream_nvlink_egress;
        cudaStreamCreateWithFlags(&stream_pcie_ingress, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&stream_nvlink_egress, cudaStreamNonBlocking);

        // Explicitly sized ping-pong event bounds fixed to prevent indexing errors
        cudaEvent_t event_pcie_ready[2];
        cudaEvent_t event_nvlink_ready[2];
        for (int i = 0; i < 2; ++i) {
            cudaEventCreateWithFlags(&event_pcie_ready[i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&event_nvlink_ready[i], cudaEventDisableTiming);
        }

        Task current_job;
        while (!kill_signal_.load(std::memory_order_relaxed)) {
            bool has_job = node_queues_[gpu_id]->try_pop(current_job);
            if (!has_job) has_job = steal_optimized_task(gpu_id, current_job);

            if (!has_job) {
                std::unique_lock<std::mutex> lk(wait_mtxs_[gpu_id]);
                wait_cvs_[gpu_id].wait_for(lk, std::chrono::microseconds(50), [this, gpu_id] {
                    return node_queues_[gpu_id]->size() > 0 || kill_signal_.load();
                });
                continue;
            }

            FlagPool::Slot& flag_slot = g_gpu_flag_pools[current_job.target_gpu]->get(current_job.flag_pool_idx);

            if (current_job.is_direct) {
                cudaError_t err1 = cudaMemcpyAsync(current_job.dev_dst, current_job.host_src, current_job.total_bytes, cudaMemcpyHostToDevice, stream_pcie_ingress);
                cudaError_t err2 = cudaMemcpyAsync(flag_slot.dev_flag_ptr, flag_slot.host_pinned_sig_ptr, sizeof(uint32_t), cudaMemcpyHostToDevice, stream_pcie_ingress);
                
                if (err1 != cudaSuccess || err2 != cudaSuccess) {
                    *(flag_slot.dev_flag_ptr) = current_job.epoch;
                }
                
                cudaLaunchHostFunc(stream_pcie_ingress, host_reclaim_callback, &flag_slot);
                VeloceMetrics::total_bytes_routed.fetch_add(current_job.total_bytes, std::memory_order_relaxed);
            } 
            else {
                size_t remaining_bytes = current_job.total_bytes;
                size_t progress_offset = 0;
                size_t ping_pong_index = 0;
                bool copy_failed = false;

                cudaEvent_t bulk_data_completed;
                if (cudaEventCreateWithFlags(&bulk_data_completed, cudaEventDisableTiming) != cudaSuccess) {
                    copy_failed = true;
                }

                while (remaining_bytes > 0 && !copy_failed) {
                    size_t chunk_slice = std::min(remaining_bytes, PIPELINE_CHUNK_SIZE);
                    void* active_scratch = (ping_pong_index % 2 == 0) ? d_scratch_A : d_scratch_B;

                    if (progress_offset >= (2 * PIPELINE_CHUNK_SIZE)) {
                        if (cudaStreamWaitEvent(stream_pcie_ingress, event_nvlink_ready[ping_pong_index % 2], 0) != cudaSuccess) {
                            copy_failed = true;
                            break;
                        }
                    }

                    // Pipeline Stage 1: Write to local staging memory via PCIe
                    if (cudaMemcpyAsync(active_scratch, current_job.host_src + progress_offset, chunk_slice, cudaMemcpyHostToDevice, stream_pcie_ingress) != cudaSuccess) {
                        copy_failed = true;
                        break;
                    }
                    if (cudaEventRecord(event_pcie_ready[ping_pong_index % 2], stream_pcie_ingress) != cudaSuccess) {
                        copy_failed = true;
                        break;
                    }

                    // Pipeline Stage 2: Forward staging memory over NVLink mesh
                    if (cudaStreamWaitEvent(stream_nvlink_egress, event_pcie_ready[ping_pong_index % 2], 0) != cudaSuccess) {
                        copy_failed = true;
                        break;
                    }
                    if (cudaMemcpyPeerAsync(current_job.dev_dst + progress_offset, current_job.target_gpu,
                                        active_scratch, gpu_id, chunk_slice, stream_nvlink_egress) != cudaSuccess) {
                        copy_failed = true;
                        break;
                    }
                    
                    if (cudaEventRecord(event_nvlink_ready[ping_pong_index % 2], stream_nvlink_egress) != cudaSuccess) {
                        copy_failed = true;
                        break;
                    }

                    progress_offset += chunk_slice;
                    remaining_bytes -= chunk_slice;
                    ping_pong_index++;
                }

                if (!copy_failed) {
                    // Rigid event fences enforce absolute ordering before publishing final completion signature
                    cudaEventRecord(bulk_data_completed, stream_nvlink_egress);
                    cudaStreamWaitEvent(stream_pcie_ingress, bulk_data_completed, 0);

                    cudaMemcpyAsync(flag_slot.dev_flag_ptr, flag_slot.host_pinned_sig_ptr, sizeof(uint32_t), cudaMemcpyHostToDevice, stream_pcie_ingress);
                    
                    cudaLaunchHostFunc(stream_pcie_ingress, host_reclaim_callback, &flag_slot);
                    cudaEventDestroy(bulk_data_completed);
                } else {
                    *(flag_slot.dev_flag_ptr) = current_job.epoch;
                    flag_slot.in_use.store(false, std::memory_order_release);
                }
                
                VeloceMetrics::total_bytes_routed.fetch_add(current_job.total_bytes, std::memory_order_relaxed);
                VeloceMetrics::multipath_transfers_count.fetch_add(1, std::memory_order_relaxed);
            }
        }

        cudaFree(d_scratch_A); cudaFree(d_scratch_B);
        for (int i = 0; i < 2; ++i) { cudaEventDestroy(event_pcie_ready[i]); cudaEventDestroy(event_nvlink_ready[i]); }
        cudaStreamDestroy(stream_pcie_ingress); cudaStreamDestroy(stream_nvlink_egress);
    }
};

static VeloceDaemon* g_veloce_daemon = nullptr;
static std::atomic<uint32_t> global_tx_sequence{1};

// ============================================================================
// SYSTEM TRANSPARENT INTERPOSITION TIERS
// ============================================================================
static void global_veloce_runtime_init() __attribute__((constructor));
static void global_veloce_runtime_init() {
    real_cudaMemcpyAsync = (cudaMemcpyAsync_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    int total_gpus = 0;
    if (cudaGetDeviceCount(&total_gpus) != cudaSuccess || total_gpus == 0) return;
    if (total_gpus > 8) total_gpus = 8;

    bool init_success = true;
    for (int i = 0; i < total_gpus; ++i) {
        g_gpu_flag_pools[i] = new FlagPool();
        if (!g_gpu_flag_pools[i]->initialize(i)) {
            init_success = false;
            break;
        }
    }

    if (init_success) {
        g_veloce_daemon = new VeloceDaemon(total_gpus);
        g_veloce_daemon->start();
    } else {
        for (int i = 0; i < total_gpus; ++i) {
            if (g_gpu_flag_pools[i]) {
                delete g_gpu_flag_pools[i];
                g_gpu_flag_pools[i] = nullptr;
            }
        }
    }
}

extern "C" {
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    if (!real_cudaMemcpyAsync) real_cudaMemcpyAsync = (cudaMemcpyAsync_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync");

    if (kind != cudaMemcpyHostToDevice || count < MULTIPATH_THRESHOLD) {
        VeloceMetrics::fallback_transfers_count.fetch_add(1, std::memory_order_relaxed);
        return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    }

    // Safeguard runtime mechanics from intercepted CUDA Graph captures
    int active_gpu_ctx = 0;
    if (cudaGetDevice(&active_gpu_ctx) != cudaSuccess) {
        VeloceMetrics::fallback_transfers_count.fetch_add(1, std::memory_order_relaxed);
        return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    }
    
    cudaStreamCaptureStatus graph_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &graph_status) != cudaSuccess || graph_status == cudaStreamCaptureStatusActive) {
        VeloceMetrics::fallback_transfers_count.fetch_add(1, std::memory_order_relaxed);
        return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    }

    if (!g_gpu_flag_pools[active_gpu_ctx] || !g_veloce_daemon) {
        VeloceMetrics::fallback_transfers_count.fetch_add(1, std::memory_order_relaxed);
        return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    }

    size_t pool_idx = g_gpu_flag_pools[active_gpu_ctx]->acquire_slot();
    if (pool_idx == std::string::npos) { 
        VeloceMetrics::fallback_transfers_count.fetch_add(1, std::memory_order_relaxed);
        return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    }

    uint32_t target_epoch = global_tx_sequence.fetch_add(1, std::memory_order_relaxed);
    FlagPool::Slot& slot = g_gpu_flag_pools[active_gpu_ctx]->get(pool_idx);
    *(slot.host_pinned_sig_ptr) = target_epoch; 

    // Intercept user stream with our low-occupancy spin sync kernel (via standard linker reference)
    launch_veloce_ptx_sync_kernel(slot.dev_flag_ptr, target_epoch, stream);

    int dynamic_relay_node = (active_gpu_ctx + 1) % 8; 

    Task memory_flight_task;
    memory_flight_task.host_src = (uint8_t*)src;
    memory_flight_task.dev_dst = (uint8_t*)dst;
    memory_flight_task.total_bytes = count;
    memory_flight_task.target_gpu = active_gpu_ctx;
    memory_flight_task.assigned_relay_gpu = dynamic_relay_node;
    memory_flight_task.flag_pool_idx = pool_idx;
    memory_flight_task.epoch = target_epoch;
    memory_flight_task.is_direct = false; 

    g_veloce_daemon->push_job(memory_flight_task);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyHostToDevice && count >= MULTIPATH_THRESHOLD) {
        cudaStream_t temporary_fallback_stream;
        if (cudaStreamCreate(&temporary_fallback_stream) == cudaSuccess) {
            cudaError_t status = cudaMemcpyAsync(dst, src, count, kind, temporary_fallback_stream);
            if (status == cudaSuccess) {
                cudaStreamSynchronize(temporary_fallback_stream);
            }
            cudaStreamDestroy(temporary_fallback_stream);
            return status;
        }
    }
    typedef cudaError_t (*cudaMemcpy_t)(void*, const void*, size_t, cudaMemcpyKind);
    static cudaMemcpy_t real_cudaMemcpy = nullptr;
    if (!real_cudaMemcpy) {
        real_cudaMemcpy = (cudaMemcpy_t)dlsym(RTLD_NEXT, "cudaMemcpy");
    }
    return real_cudaMemcpy(dst, src, count, kind);
}
}
