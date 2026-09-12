// cuda/utils/cuda_utils.cuh
//
// Shared CUDA error-checking and GPU timing helpers. Every kernel from here
// on wraps its host-side CUDA API calls (malloc, memcpy, kernel launch) in
// CUDA_CHECK, and uses GpuTimer for the event-based timing FR-12 needs
// (kernel time separate from H2D/D2H transfer time).

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Wrap every CUDA runtime API call in this. Aborts with a clear message
// (file, line, and the CUDA error string) instead of silently continuing
// with a failed allocation/copy/launch.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err__));               \
            std::exit(1);                                                   \
        }                                                                    \
    } while (0)

// Call immediately after a kernel launch to catch launch-configuration
// errors (e.g. too many threads per block) that don't show up as a return
// value the way regular API calls do.
#define CUDA_CHECK_KERNEL_LAUNCH()                                           \
    CUDA_CHECK(cudaGetLastError())

namespace gcv {

// Shared device-side helpers used by convolution-style kernels (naive,
// tiled, Sobel, blur) so boundary/clamping logic is defined exactly once
// instead of copy-pasted into every kernel file.
__device__ inline int clamp_coord(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline uint8_t clamp_to_byte(float v) {
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return static_cast<uint8_t>(v + 0.5f);
}

// Event-based GPU timer. More accurate than wall-clock host timers for
// measuring kernel execution time, since it's timed on the GPU's own clock
// and isn't polluted by host-side scheduling jitter.
class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // Non-copyable (owns CUDA event handles).
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    void stop() { CUDA_CHECK(cudaEventRecord(stop_)); }

    // Blocks until the stop event completes, then returns elapsed ms.
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

} // namespace gcv
