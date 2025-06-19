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

// FIXED: Find maximum value in each column (for each sample)
__global__ void softmax_max_kernel(const float *input, float *max_vals,
                                   int rows, int cols) {
  int col =
      blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles one sample
  if (col < cols) {
    float max_val = input[col]; // Start with first element of column
    for (int row = 1; row < rows; row++) {
      float val = input[row * cols + col];
      max_val = fmaxf(max_val, val);
    }
    max_vals[col] = max_val;
  }
}

// FIXED: Compute exp and sum for each column (sample)
__global__ void softmax_exp_sum_kernel(const float *input,
                                       const float *max_vals, float *exp_sums,
                                       float *output, int rows, int cols) {
  int col =
      blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles one sample
  if (col < cols) {
    float sum = 0.0f;
    for (int row = 0; row < rows; row++) {
      int idx = row * cols + col;
      float exp_val = expf(input[idx] - max_vals[col]);

      // Check for invalid values
      if (isnan(exp_val) || isinf(exp_val)) {
        exp_val = 0.0f;
      }

      output[idx] = exp_val;
      sum += exp_val;
    }

    // Prevent division by zero
    if (sum == 0.0f || isnan(sum) || isinf(sum)) {
      sum = 1.0f;
    }

    exp_sums[col] = sum;
  }
}

// FIXED: Normalize by the sum to get final softmax values
__global__ void softmax_normalize_kernel(float *output, const float *exp_sums,
                                         int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    int col = idx % cols; // Which sample
    float sum = exp_sums[col];

    // Prevent division by zero/invalid values
    if (sum > 0.0f && !isnan(sum) && !isinf(sum)) {
      output[idx] /= sum;
    } else {
      // If sum is invalid, set to uniform distribution
      output[idx] = 1.0f / rows;
    }

    // Final safety check
    if (isnan(output[idx]) || isinf(output[idx])) {
      output[idx] = 1.0f / rows;
    }
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

  // Step 1: Find max values for numerical stability (one thread per
  // sample/column)
  gridSize = (cols + blockSize - 1) / blockSize;
  softmax_max_kernel<<<gridSize, blockSize>>>(input, max_vals, rows, cols);
  cudaDeviceSynchronize();

  // Step 2: Compute exp and sum (one thread per sample/column)
  softmax_exp_sum_kernel<<<gridSize, blockSize>>>(input, max_vals, exp_sums,
                                                  output, rows, cols);
  cudaDeviceSynchronize();

  // Step 3: Normalize (one thread per element)
  gridSize = (rows * cols + blockSize - 1) / blockSize;
  softmax_normalize_kernel<<<gridSize, blockSize>>>(output, exp_sums, rows,
                                                    cols);
  cudaDeviceSynchronize();
}
}
