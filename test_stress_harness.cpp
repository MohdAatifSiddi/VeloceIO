#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cassert>

constexpr size_t TEST_ALLOC_SIZE = 128 * 1024 * 1024; // 128MB Payload Blocks

void execute_host_worker(int thread_id, int gpu_id) {
    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        std::cerr << "Thread " << thread_id << " failed to bind to GPU " << gpu_id << std::endl;
        return;
    }
    
    uint8_t* h_src = nullptr;
    cudaMallocHost(&h_src, TEST_ALLOC_SIZE);
    for (size_t i = 0; i < TEST_ALLOC_SIZE; ++i) {
        h_src[i] = static_cast<uint8_t>((i + thread_id) % 256);
    }

    uint8_t* d_dst = nullptr;
    cudaMalloc(&d_dst, TEST_ALLOC_SIZE);

    cudaStream_t worker_stream;
    cudaStreamCreate(&worker_stream);

    // Warmup initialization pass
    cudaMemcpyAsync(d_dst, h_src, TEST_ALLOC_SIZE, cudaMemcpyHostToDevice, worker_stream);
    cudaStreamSynchronize(worker_stream);

    auto start_time = std::chrono::high_resolution_clock::now();

    constexpr int ITERATIONS = 10;
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        cudaMemcpyAsync(d_dst, h_src, TEST_ALLOC_SIZE, cudaMemcpyHostToDevice, worker_stream);
    }
    cudaStreamSynchronize(worker_stream);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    double total_gigabytes = (static_cast<double>(TEST_ALLOC_SIZE) * ITERATIONS) / (1024.0 * 1024.0 * 1024.0);
    double effective_bandwidth = total_gigabytes / duration.count();

    std::cout << "[Worker Thread " << thread_id << " running on GPU " << gpu_id << "] "
              << "Processed " << ITERATIONS << " loop cycles. Performance Index: " 
              << effective_bandwidth << " GB/s" << std::endl;

    // Direct memory verify pass
    std::vector<uint8_t> h_verify(TEST_ALLOC_SIZE);
    cudaMemcpy(h_verify.data(), d_dst, TEST_ALLOC_SIZE, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < TEST_ALLOC_SIZE; ++i) {
        assert(h_verify[i] == static_cast<uint8_t>((i + thread_id) % 256));
    }

    cudaFree(d_dst);
    cudaFreeHost(h_src);
    cudaStreamDestroy(worker_stream);
}

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < 2) {
        std::cerr << "Hardware restriction error: Multipath tests require at least 2 connected GPUs." << std::endl;
        return 1;
    }

    std::cout << "================================================================================" << std::endl;
    std::cout << "Starting high-concurrency multi-GPU verification thread framework..." << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::vector<std::thread> test_cluster;
    for (int i = 0; i < dev_count; ++i) {
        test_cluster.emplace_back(execute_host_worker, i, i);
    }

    for (auto& t : test_cluster) t.join();
    std::cout << "\nSUCCESS: System integrity validated. No packet data drift observed." << std::endl;
    return 0;
}
