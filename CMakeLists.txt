cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(VeloceIO LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure high-performance systems compilation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fPIC -pthread -Wall -Wextra")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xcompiler -fPIC -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 --use_fast_math")

set(CMAKE_BUILD_TYPE Release)

# Locate native CUDA runtime installations
find_package(CUDAToolkit REQUIRED)

# Build the core shared runtime interposition library
add_library(veloce SHARED
    libveloce.cpp
    veloce_sync.cu
    veloce_gds.cu
)

target_include_directories(veloce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(veloce PRIVATE CUDA::cudart dl)

# Build the high-concurrency verification stress engine
add_executable(veloce_test test_stress_harness.cpp)
target_link_libraries(veloce_test PRIVATE veloce CUDA::cudart pthread)

message(STATUS "VeloceIO Engine configuration complete. Ready for multi-GPU deployment.")
