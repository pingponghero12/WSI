#include "activation_funcs.h"
#include <cmath>
#include <cuda_runtime.h>

// ReLU activation function
__global__ void relu_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

// ReLU derivative
__global__ void relu_derivative_kernel(const float *input, float *output,
                                       int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] > 0.0f ? 1.0f : 0.0f;
  }
}

// Find maximum value in each row (for numerical stability in softmax)
__global__ void softmax_max_kernel(const float *input, float *max_vals,
                                   int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float max_val = input[row * cols];
    for (int col = 1; col < cols; col++) {
      max_val = fmaxf(max_val, input[row * cols + col]);
    }
    max_vals[row] = max_val;
  }
}

// Compute exp and sum for each row
__global__ void softmax_exp_sum_kernel(const float *input,
                                       const float *max_vals, float *exp_sums,
                                       float *output, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
      float exp_val = expf(input[row * cols + col] - max_vals[row]);
      output[row * cols + col] = exp_val;
      sum += exp_val;
    }
    exp_sums[row] = sum;
  }
}

// Normalize by the sum to get final softmax values
__global__ void softmax_normalize_kernel(float *output, const float *exp_sums,
                                         int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    int row = idx / cols;
    output[idx] /= exp_sums[row];
  }
}

// Host wrapper functions
extern "C" {
void cuda_relu(const float *input, float *output, int size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  relu_kernel<<<gridSize, blockSize>>>(input, output, size);
  cudaDeviceSynchronize();
}

void cuda_relu_derivative(const float *input, float *output, int size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  relu_derivative_kernel<<<gridSize, blockSize>>>(input, output, size);
  cudaDeviceSynchronize();
}

void cuda_softmax(const float *input, float *output, float *max_vals,
                  float *exp_sums, int rows, int cols) {
  int blockSize = 256;
  int gridSize;

  // Step 1: Find max values for numerical stability
  gridSize = (rows + blockSize - 1) / blockSize;
  softmax_max_kernel<<<gridSize, blockSize>>>(input, max_vals, rows, cols);
  cudaDeviceSynchronize();

  // Step 2: Compute exp and sum
  softmax_exp_sum_kernel<<<gridSize, blockSize>>>(input, max_vals, exp_sums,
                                                  output, rows, cols);
  cudaDeviceSynchronize();

  // Step 3: Normalize
  gridSize = (rows * cols + blockSize - 1) / blockSize;
  softmax_normalize_kernel<<<gridSize, blockSize>>>(output, exp_sums, rows,
                                                    cols);
  cudaDeviceSynchronize();
}
}
