cmake_minimum_required(VERSION 3.18)
project(GpuComputerVisionEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Adjust to match your GPU's compute capability (verify_setup.py / nvidia-smi will tell you).
# 75 = Turing (RTX 20xx / GTX 16xx), 86 = Ampere (RTX 30xx), 89 = Ada (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(cuda/utils)

# ---- CPU baseline library (Phase 2) ----
add_library(cpu_algorithms
    cpp/cpu/grayscale.cpp
    cpp/cpu/convolution.cpp
    cpp/cpu/gaussian_blur.cpp
    cpp/cpu/sobel.cpp
)
target_link_libraries(cpu_algorithms PUBLIC ${OpenCV_LIBS})

# ---- CUDA kernels library (Phases 3-5) ----
add_library(cuda_kernels
    cuda/kernels/grayscale.cu
    cuda/kernels/inversion.cu
    cuda/kernels/convolution.cu
    cuda/kernels/gaussian_blur.cu
    cuda/kernels/sobel.cu
    cuda/kernels/histogram.cu
    cuda/kernels/matrix_mul.cu
    cuda/optimized/tiled_convolution.cu
    cuda/optimized/tiled_matrix_mul.cu
)
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda_kernels PUBLIC CUDA::cudart)

# ---- Benchmark executable (Phase 6) ----
add_executable(benchmark cpp/benchmark/benchmark.cpp)
target_link_libraries(benchmark PRIVATE cpu_algorithms cuda_kernels ${OpenCV_LIBS} CUDA::cudart)

enable_testing()
add_subdirectory(tests)
