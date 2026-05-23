#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void veloce_ptx_sync_kernel(volatile uint32_t* sync_flag, uint32_t expected_epoch) {
    // Restrict processing boundary strictly to thread 0 of block 0
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t current_epoch = 0;
        uint32_t sleep_ns = 128; // Initialize with a precise 128ns sleep step
        constexpr uint32_t MAX_SLEEP_NS = 2048; // Cap duration to mitigate latency tail-spikes

        while (true) {
            // Read direct from HBM memory arrays, bypassing local L1/L2 data cache architectures
            // PTX target: ld.global.cg.volatile.u32
            asm volatile(
                "ld.global.cg.volatile.u32 %0, [%1];"
                : "=r"(current_epoch)
                : "l"(sync_flag)
                : "memory"
            );

            if (current_epoch >= expected_epoch) {
                break; // Transmission cycle successfully matched
            }

            // Execute power-aware SM thread yield via inline assembly nanosleep calls
            asm volatile(
                "nanosleep.u32 %0;"
                :
                : "r"(sleep_ns)
            );

            // Scale sleep duration exponentially up to hardware constraint caps
            sleep_ns = (sleep_ns * 2 > MAX_SLEEP_NS) ? MAX_SLEEP_NS : sleep_ns * 2;
        }
    }
}

extern "C" void launch_veloce_ptx_sync_kernel(volatile uint32_t* sync_flag, uint32_t expected_epoch, cudaStream_t stream) {
    veloce_ptx_sync_kernel<<<1, 1, 0, stream>>>(sync_flag, expected_epoch);
}
