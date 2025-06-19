#pragma once

#include <cuda_runtime.h>

// Activation function kernels
__global__ void relu_kernel(const float* input, float* output, int size);
__global__ void relu_derivative_kernel(const float* input, float* output, int size);
__global__ void softmax_max_kernel(const float* input, float* max_vals, int rows, int cols);
__global__ void softmax_exp_sum_kernel(const float* input, const float* max_vals, float* exp_sums, float* output, int rows, int cols);
__global__ void softmax_normalize_kernel(float* output, const float* exp_sums, int rows, int cols);
