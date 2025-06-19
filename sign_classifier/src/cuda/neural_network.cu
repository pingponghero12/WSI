#include "cuda_neural_network.h"
#include <cuda_runtime.h>

// This file can contain additional neural network specific kernels
// For now, we'll keep it minimal to avoid linking issues

__global__ void dummy_kernel() {
  // Empty kernel to ensure this compilation unit contributes something
}

extern "C" {
void cuda_dummy_function() {
  // This ensures the compilation unit is not empty
  dummy_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}
}
