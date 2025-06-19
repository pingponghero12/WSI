#!/bin/bash

echo "Building CUDA Neural Network with CMake..."

# Clean previous builds
rm -rf build/
rm -f python/cuda_neural_network*.so

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
