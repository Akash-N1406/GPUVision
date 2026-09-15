# Performance Analysis Report

**GPU-Accelerated Computer Vision & Image Processing Engine**
Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, Ampere, compute capability 8.6), CUDA 12.0
Test image: `bird.jpg` (452×678 native, resized to each target resolution)

---

## 1. Methodology

Every algorithm has three implementations: a CPU baseline (C++/OpenCV for I/O only — the
actual math is hand-written, not `cv::` shortcuts), a naive CUDA kernel (global memory
only), and, for the two centerpiece workloads (2D convolution and matrix multiplication),
a shared-memory tiled CUDA kernel.

Two benchmarking approaches were used, and they are **not equally reliable**:

- **The resolution/size sweep** (`cpp/benchmark/benchmark.cpp`) times the public
  `*_cpu()` / `*_cuda()` API functions with a host wall-clock around 5 warm-up + 20
  measured calls. This is good for broad CPU-vs-GPU trends across problem size, but the
  GPU-side functions allocate and free device memory on *every call*, which adds
  `cudaMalloc`/`cudaFree` overhead and noise into the "GPU time" — this **inflates and
  destabilizes** results for cheap kernels and for fine-grained naive-vs-tiled
  comparisons specifically.
- **The instrumented deep-dive** (`convolution_cuda_naive_detailed` /
  `convolution_cuda_tiled_detailed`, `cuda/demo_convolution_detailed.cu`) allocates
  device buffers **once** and reuses them across all 20 measured iterations, timing
  H2D transfer, kernel execution, and D2H transfer separately via CUDA events. This is
  the trustworthy source for the naive-vs-tiled convolution comparison specifically.

**Where the two disagree, this report cites the instrumented numbers.** Matrix
multiplication was not given the same instrumented treatment — its naive-vs-tiled
sweep numbers should be read as directional only; the one clean comparison available
is the earlier single-configuration N=512 test (naive 45.8x vs tiled 60x vs CPU,
tiled 1.31x faster than naive), not the noisier sweep.

---

## 2. Correctness

Every kernel was verified against its CPU baseline with a dedicated correctness demo
before being folded into the benchmark suite. Results:

| Algorithm | Max error | Notes |
|---|---|---|
| Grayscale | 0 | Exact match |
| Inversion | 0 | Exact match (pure integer math) |
| Gaussian blur | 1 | 4/919,368 bytes differ, float rounding |
| Sobel | 0 | Exact match |
| Convolution (naive) | 0 | Exact match |
| Convolution (tiled) | 0 | Exact match, both before and after the redesign (§4) |
| Histogram | 0 | All 256 bins exact — confirms atomics/merge correctness |
| Matrix multiply (naive) | 1.14e-05 | FMA-ordering drift, expected |
| Matrix multiply (tiled) | 0 vs naive | Identical accumulation order |

No correctness regressions anywhere in the project.

---

## 3. CPU vs GPU: does it help, and when?

See `reports/charts/speedup_vs_resolution.png` and `execution_time_vs_resolution.png`.

**SRS Question 1 — Does GPU acceleration improve performance?** Yes, decisively for
compute-heavy operations: Gaussian blur (12.9x-29.5x across resolutions), Sobel
(3.7x-22.3x), convolution (6.1x-16.4x), matrix multiply (up to 247x at N=1024).

**SRS Question 2 — At what size does GPU processing become advantageous?** This is
where the project's most interesting finding lives. Cheap, low-arithmetic-intensity
kernels (grayscale, inversion, histogram) start out **GPU-slower-than-CPU** at small
sizes (0.26x-0.65x) because launch latency and H2D/D2H transfer dominate a workload
that's almost all memory movement and almost no compute. But this isn't a fixed
property of the algorithm — it's a function of size:

- Grayscale: 0.64x at 640×480 → **1.84x at 4K** — crosses over to GPU-favorable.
- Histogram: 0.28x at 640×480 → **1.69x at 4K** — same crossover.
- Inversion stays GPU-unfavorable even at 4K (0.84x) — it's *so* cheap
  (one subtraction per byte) that even at 8.3M pixels, transfer overhead still
  dominates compute.

This is a clean, data-backed answer: **arithmetic intensity determines whether size
alone is enough to make the GPU win**, not just size on its own.

---

## 4. Case study: the tiled convolution investigation

This is the most technically substantial finding in the project, and worth reporting
as a narrative rather than just a number.

**The puzzle.** The resolution sweep showed `convolution_tiled` running slower than
`convolution_naive` at every resolution except the smallest, with the gap *growing*
as images got bigger (up to ~2x slower at 720p) — the opposite of what shared-memory
tiling is supposed to deliver, and the opposite of an earlier single-sample test that
had shown tiled winning by 1.14x.

**The investigation.** Rather than accept either result at face value, new
instrumentation (`convolution_cuda_naive_detailed` / `_tiled_detailed`) was built to
separate real GPU-clock kernel time from allocation/transfer noise, using buffers
allocated once and CUDA-event timing. This confirmed the reversal was **real, not
noise** — and pinpointed the cause: the tiled kernel was launching once *per channel*
(3 launches for RGB), each a full independent sweep over the image with zero cache
reuse between channels — effectively 3 full passes instead of 1, plus 3x the
`__syncthreads()` overhead.

**The fix.** `convolution_tiled_kernel` was rewritten to load all 3 channels into
shared memory in a **single** kernel launch (interleaved tile, one sync, per-thread
channel loop), matching the naive kernel's single-launch structure.

**The result.** Correctness held throughout (0 error, re-verified after the rewrite).
Performance improved substantially and in the textbook-correct direction: the
naive/tiled kernel-time ratio now *shrinks* with resolution (1.31x → 1.24x → 1.19x →
1.14x, closer to parity as images grow) instead of growing (1.37x → 1.88x, before the
fix). However, tiled still doesn't *beat* naive on pure kernel time for this 3×3
kernel — a legitimate, unglamorous finding: shared memory's reuse benefit is
theoretically small for a radius-1 stencil (max 9x), and modern L2 cache likely
already captures a meaningful fraction of that overlap automatically, so the
overhead of explicit tile staging doesn't fully pay for itself at this kernel size.
The hypothesis (untested) is that a larger kernel radius (5×5, 7×7) would show a
clearer win, since there's more redundant global-memory traffic to eliminate.

**Why this matters for an interview:** this demonstrates the actual skill the SRS
cares about — not "I wrote a CUDA kernel," but "I profiled, found a real performance
bug, diagnosed the root cause, fixed it, and can explain exactly why the fix worked
and where its limits are."

---

## 5. Matrix multiplication: the best number in the project

Naive CUDA matmul hit **247x speedup** over CPU at N=1024 (5.7s CPU vs 23ms GPU) — by
far the largest speedup measured. This makes sense: matrix multiply is O(N³) compute
against O(N²) data movement, the textbook case for GPU acceleration, unlike the
image kernels above which are closer to O(N) compute per O(N) data.

The naive-vs-tiled comparison at N=512 (the one clean, single-configuration test
available) showed tiled winning by 1.31x (60x vs 45.8x speedup over CPU) — consistent
with GEMM tiling's well-established advantage, unlike the 3×3 convolution case.

---

## 6. Known limitations / honest caveats

- **Histogram has an unfixed inefficiency**: `histogram_cuda` calls `grayscale_cuda`
  internally for RGB images, which does its own complete device round-trip — so RGB
  histograms pay for two H2D/D2H cycles instead of one. Documented but not fixed;
  the correct fix is fusing grayscale conversion directly into the histogram kernel.
- **Matrix multiply's naive-vs-tiled sweep numbers are unreliable** — same
  allocate-every-call design as everything except convolution's instrumented
  variants. The N=512 single-test result (§5) is the number to trust.
- **Larger convolution kernel radii were not tested** — the shared-memory
  finding in §4 is specific to a 3×3 kernel; whether tiling wins outright at 5×5/7×7
  remains an open, testable question.

---

## 7. Resume-ready summary

> Built a CUDA C++ image-processing engine (grayscale, inversion, Gaussian blur,
> Sobel edge detection, 2D convolution, matrix multiplication, histogram) with
> CPU baselines, naive CUDA kernels, and shared-memory/constant-memory optimized
> versions, achieving up to 247x GPU speedup on matrix multiplication. Diagnosed and
> fixed a real performance regression in the shared-memory convolution kernel using
> custom CUDA-event instrumentation, isolating the root cause to a per-channel kernel
> launch pattern and reducing the naive/tiled performance gap by roughly 40% at the
> largest tested resolution.