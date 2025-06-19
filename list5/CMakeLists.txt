cmake_minimum_required(VERSION 3.18)
project(CudaNeuralNetwork LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories(src/include)

# CUDA sources
set(CUDA_SOURCES
    src/cuda/matrix_ops.cu
    src/cuda/activation_funcs.cu
    src/cuda/neural_network.cu
)

# C++ sources
set(CPP_SOURCES
    src/cpp/neural_network.cpp
    src/cpp/python_bindings.cpp
)

# Create the Python module
pybind11_add_module(cuda_neural_network ${CUDA_SOURCES} ${CPP_SOURCES})

# Link CUDA libraries
target_link_libraries(cuda_neural_network PRIVATE CUDA::cudart CUDA::cublas)

# Set properties - this is crucial for CUDA device linking
set_target_properties(cuda_neural_network PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON  # This line is important!
)

# Set output directory to current directory for easy import
set_target_properties(cuda_neural_network PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
)

# Ensure proper CUDA compilation flags
target_compile_options(cuda_neural_network PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_61>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
