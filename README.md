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
- [ ] Phase 2 — CPU Implementation (baseline correctness)
- [ ] Phase 3 — Basic CUDA (grayscale, inversion — thread/block fundamentals)
- [ ] Phase 4 — CUDA Convolution (naive -> shared memory -> tiled)
- [ ] Phase 5 — CUDA Optimization (coalescing, constant memory, occupancy)
- [ ] Phase 6 — Benchmarking (CPU vs GPU, multiple resolutions/configs)
- [ ] Phase 7 — Visualization (execution-time and speedup graphs)
- [ ] Phase 8 — Django Application (optional web UI)
- [ ] Phase 9 — Profiling (Nsight Systems / Nsight Compute)

## Requirements

- NVIDIA GPU + driver, CUDA Toolkit (12.x recommended)
- CMake >= 3.18, g++ with C++17 support
- OpenCV (dev headers for C++, `opencv-python` for the Python layer)
- Python 3.10+

## Phase 1 — Getting Started

```bash
# From the project root, on the machine with the NVIDIA GPU (your WSL2 box):
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 python/verify_setup.py
```

`verify_setup.py` checks: `nvidia-smi`, `nvcc`, `g++`, `cmake`, OpenCV
(C++ and Python), NumPy, Docker, and that the directory structure is intact.
Both `nvidia-smi` and `nvcc` are treated as critical — the script exits
non-zero if either is missing, since nothing in Phase 3 onward works without
them.

Once Phase 1 passes clean, Phase 2 starts: CPU implementations in
`cpp/cpu/` that establish the correctness baseline every CUDA kernel gets
checked against.

## Project Structure

```
gpu-computer-vision/
├── cuda/
│   ├── kernels/       naive CUDA kernels (Phase 3-4)
│   ├── optimized/     shared-memory / tiled versions (Phase 4-5)
│   └── utils/         CUDA helper headers (timing, error-checking)
├── cpp/
│   ├── cpu/           CPU baseline implementations (Phase 2)
│   └── benchmark/     CPU vs GPU benchmark harness (Phase 6)
├── python/
│   ├── preprocessing/ image I/O helpers
│   ├── benchmarking/  result aggregation, CSV generation
│   └── visualization/ matplotlib/Chart.js report generation
├── backend/django_app/  optional web UI (Phase 8)
├── tests/                unit + correctness tests
├── data/{input,output}   sample images, processed results
└── reports/              benchmark_results.csv, analysis
```

## Resume Note

Don't put a specific speedup number (e.g. "10x") on a resume/portfolio until
it's measured on your actual hardware in Phase 6.
