# GPU-Accelerated Computer Vision Engine

A CUDA C++ image-processing engine that benchmarks GPU-parallel implementations
(grayscale, inversion, Gaussian blur, Sobel edge detection, 2D convolution,
matrix multiplication, histogram) against CPU baselines, with a focus on
demonstrating CUDA optimization technique — not just kernel-writing.

Companion project to [Fraud-Detection-System](https://github.com/Akash-N1406/Fraud-Detection-System):
that project shows distributed data engineering + ML; this one shows
low-level parallel computing and GPU performance optimization.

## Centerpiece

```
CPU -> Naive CUDA -> Shared-Memory CUDA -> Tiled CUDA -> Benchmarking -> Performance Analysis
```

The Django layer (Phase 8) is optional/secondary — the technical substance is
in the CUDA kernels and the benchmarking methodology.

## Development Phases

- [x] Phase 1 — Environment Setup
- [x] Phase 2 — CPU Implementation (baseline correctness)
- [x] Phase 3 — Basic CUDA (grayscale, inversion — thread/block fundamentals)
- [x] Phase 4 — CUDA Convolution (naive -> shared memory -> tiled)
- [x] Phase 5 — CUDA Optimization (constant memory, tiled matrix multiplication)
- [x] Phase 6 — Benchmarking (CPU vs GPU across resolutions and matrix sizes)
- [x] Phase 7 — Visualization (execution-time, speedup, and naive-vs-tiled charts)
- [x] Phase 8 — Django Application (upload, algorithm/execution selection, CPU-vs-CUDA comparison)
- [x] Phase 9 — Profiling (Nsight guide written; a real `ncu` run on this hardware
      was blocked by a WSL2 GPU-performance-counter permission issue — see
      `reports/nsight_profiling_guide.md`'s troubleshooting section)

See `reports/performance_analysis.md` for the full write-up: correctness
results, the SRS's own research questions answered with real data, and a
detailed case study of a real performance bug that was found, diagnosed,
and fixed in the shared-memory convolution kernel.

## Requirements

- NVIDIA GPU + driver, CUDA Toolkit (12.x recommended)
- CMake >= 3.18, g++ with C++17 support
- OpenCV (dev headers for C++, `opencv-python` for the Python layer)
- Python 3.10+
- Django (optional, only for the Phase 8 web UI)

## Getting Started

```bash
# From the project root, on the machine with the NVIDIA GPU:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 python/verify_setup.py
```

`verify_setup.py` checks: `nvidia-smi`, `nvcc`, `g++`, `cmake`, OpenCV
(C++ and Python), NumPy, Docker, and that the directory structure is intact.

Each algorithm has a standalone build-and-run demo documented in its own
file's header comment (e.g. `cuda/demo_tiled_convolution_cuda.cu`), which is
the fastest way to verify any one piece works before touching the rest. The
full benchmark suite (`cpp/benchmark/benchmark.cpp`) and the Django UI
(`backend/django_app/`) both build on top of those same, individually
verified pieces.

## Project Structure

```
gpu-computer-vision/
├── cuda/
│   ├── kernels/       CUDA kernels: grayscale, inversion, blur, sobel,
│   │                  naive convolution, histogram, matmul
│   ├── optimized/     shared-memory tiled convolution and matmul
│   └── utils/         CUDA helper headers (timing, error-checking, clamping)
├── cpp/
│   ├── cpu/           CPU baseline implementations
│   ├── common/        shared Image/Matrix types, convolution/resize helpers
│   ├── benchmark/      CPU vs GPU benchmark harness + timing utilities
│   └── cli/            process_image — unified CLI bridging the engine to Django
├── python/
│   └── visualization/ benchmark_results.csv -> charts (matplotlib)
├── backend/django_app/  optional web UI (upload, algorithm/execution selection)
├── data/{input,output}   sample images, processed results
└── reports/              benchmark_results.csv, charts/, performance_analysis.md,
                          nsight_profiling_guide.md
```

## Resume Note

The centerpiece finding — a real performance regression found, diagnosed with
custom CUDA-event instrumentation, and fixed in the shared-memory convolution
kernel (see `reports/performance_analysis.md` §4) — is stronger interview
material than any single speedup number. Use it. The best raw numbers
(247x matmul, 60x tiled matmul, 29x Gaussian blur) are specific to the test
image and RTX 3050 Laptop GPU used here — re-verify on your own hardware
before quoting them elsewhere.