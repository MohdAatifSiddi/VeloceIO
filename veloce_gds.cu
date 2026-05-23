#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

/**
 * Coalesced 64-bit vector alignment shift kernel.
 * Corrects byte alignment shifts caused by variable JSON metadata headers.
 */
extern "C" __global__ void veloce_gds_shift_alignment_kernel(
    const uint8_t* __restrict__ src_unaligned_scratchpad,
    uint8_t* __restrict__ dst_aligned_tensor_payload,
    size_t total_tensor_bytes,
    size_t shift_offset_bytes)
{
    size_t global_thread_id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t element_stride   = (size_t)gridDim.x * blockDim.x;

    // Map source pointers onto 64-bit read structures
    const uint64_t* src_aligned_64 = (const uint64_t*)(src_unaligned_scratchpad + shift_offset_bytes);
    uint64_t* dst_aligned_64       = (uint64_t*)dst_aligned_tensor_payload;

    size_t blocks_64 = total_tensor_bytes / 8;
    size_t residual_bytes = total_tensor_bytes % 8;

    // Step 1: High-throughput 8-byte coalesced transaction block loop
    for (size_t idx = global_thread_id; idx < blocks_64; idx += element_stride) {
        dst_aligned_64[idx] = src_aligned_64[idx];
    }

    // Step 2: Dedicated single-thread cleanup pass managing trailing scalar bytes
    if (global_thread_id == 0 && residual_bytes > 0) {
        size_t baseline_byte_index = blocks_64 * 8;
        for (size_t b = 0; b < residual_bytes; ++b) {
            dst_aligned_tensor_payload[baseline_byte_index + b] = 
                src_unaligned_scratchpad[baseline_byte_index + shift_offset_bytes + b];
        }
    }
}

// Host orchestration layer wrapper interface
extern "C" void trigger_gds_realignment_pipeline(
    const uint8_t* d_scratchpad,
    uint8_t* d_tensor,
    size_t bytes,
    size_t offset,
    cudaStream_t stream)
{
    if (bytes == 0) return;
    
    constexpr int THREADS_PER_BLOCK = 256;
    size_t needed_blocks = (bytes / 8 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int optimized_grid = std::min(static_cast<size_t>(2048), std::max(static_cast<size_t>(1), needed_blocks));

    veloce_gds_shift_alignment_kernel<<<optimized_grid, THREADS_PER_BLOCK, 0, stream>>>(
        d_scratchpad, d_tensor, bytes, offset
    );
}
