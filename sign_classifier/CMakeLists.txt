cmake_minimum_required(VERSION 3.18)
project(SignClassifier LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(src/include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA sources
set(CUDA_SOURCES
    src/cuda/matrix_ops.cu
    src/cuda/activation_funcs.cu
    src/cuda/neural_network.cu
)

# C++ sources
set(CPP_SOURCES
    src/cpp/sign_classifier.cpp
    src/cpp/python_bindings.cpp
)

# Create the Python module
pybind11_add_module(sign_classifier ${CUDA_SOURCES} ${CPP_SOURCES})

target_link_libraries(sign_classifier PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(sign_classifier PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
)

target_compile_options(sign_classifier PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_61>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
