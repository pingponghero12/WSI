#!/bin/bash

echo "=== CUDA Debug Build Script ==="

# Clean everything
rm -rf build/
rm -f python/cuda_neural_network*.so

# Check CUDA installation
echo "=== CUDA Environment Check ==="
echo "CUDA_PATH: $CUDA_PATH"
echo "CUDA_ROOT: $CUDA_ROOT"
nvcc --version
echo ""

# Check what CUDA libraries we have
echo "=== CUDA Libraries Check ==="
ls -la $CUDA_PATH/lib64/ | grep -E "(cudart|cudadevrt)"
echo ""

# Create build directory
mkdir build
cd build

# Configure with maximum debug info
echo "=== CMake Configuration ==="
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_VERBOSE_MAKEFILE=ON \
         -DCMAKE_CUDA_FLAGS="-arch=sm_61 --expt-relaxed-constexpr -g -G" \
         -DCUDA_VERBOSE_BUILD=ON

echo ""
echo "=== Building with verbose output ==="
make VERBOSE=1

cd ..

echo ""
echo "=== Post-build Analysis ==="
if [ -f python/cuda_neural_network*.so ]; then
    echo "✅ .so file created: $(ls python/cuda_neural_network*.so)"
    
    # Check what symbols the .so file has
    echo ""
    echo "=== Checking .so file symbols ==="
    nm python/cuda_neural_network*.so | grep -i cuda || echo "No CUDA symbols found"
    
    # Check dependencies
    echo ""
    echo "=== Checking .so file dependencies ==="
    ldd python/cuda_neural_network*.so
    
else
    echo "❌ No .so file created"
fi
