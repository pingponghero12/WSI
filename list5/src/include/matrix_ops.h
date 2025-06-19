#pragma once

#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024

// Matrix multiplication kernels
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K);
__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K);
__global__ void matmul_transposed_kernel(const float* A, const float* B, float* C, int M, int N, int K);

// Utility kernels
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols);
__global__ void add_bias_kernel(float* matrix, const float* bias, int rows, int cols);
__global__ void scale_kernel(float* data, float scale, int size);
__global__ void subtract_kernel(const float* A, const float* B, float* C, int size);
__global__ void multiply_kernel(const float* A, const float* B, float* C, int size);
__global__ void sum_rows_kernel(const float* input, float* output, int rows, int cols);
