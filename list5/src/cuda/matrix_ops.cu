#include "matrix_ops.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Basic matrix multiplication kernel
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

// Optimized matrix multiplication with shared memory
__global__ void matmul_shared_kernel(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    // Load tiles into shared memory
    if (row < M && (tile * TILE_SIZE + threadIdx.x) < K)
      As[threadIdx.y][threadIdx.x] =
          A[row * K + tile * TILE_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    if (col < N && (tile * TILE_SIZE + threadIdx.y) < K)
      Bs[threadIdx.y][threadIdx.x] =
          B[(tile * TILE_SIZE + threadIdx.y) * N + col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Compute partial sum
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// Matrix multiplication with B transposed (A * B^T)
__global__ void matmul_transposed_kernel(const float *A, const float *B,
                                         float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[col * K + k]; // B is accessed as if transposed
    }
    C[row * N + col] = sum;
  }
}

// Matrix transpose
__global__ void transpose_kernel(const float *input, float *output, int rows,
                                 int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols) {
    output[col * rows + row] = input[row * cols + col];
  }
}

// Add bias to each row of matrix
__global__ void add_bias_kernel(float *matrix, const float *bias, int rows,
                                int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    int row = idx / cols;
    matrix[idx] += bias[row];
  }
}

// Scale all elements by a constant
__global__ void scale_kernel(float *data, float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] *= scale;
  }
}

// Element-wise subtraction: C = A - B
__global__ void subtract_kernel(const float *A, const float *B, float *C,
                                int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] - B[idx];
  }
}

// Element-wise multiplication: C = A * B
__global__ void multiply_kernel(const float *A, const float *B, float *C,
                                int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] * B[idx];
  }
}

// Sum along rows (reduce columns)
__global__ void sum_rows_kernel(const float *input, float *output, int rows,
                                int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0f;
    for (int col = 0; col < cols; col++) {
      sum += input[row * cols + col];
    }
    output[row] = sum;
  }
}

// Host wrapper functions
extern "C" {
void cuda_matmul(const float *A, const float *B, float *C, int M, int N,
                 int K) {
  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                (M + blockSize.y - 1) / blockSize.y);

  if (M >= TILE_SIZE && N >= TILE_SIZE && K >= TILE_SIZE) {
    matmul_shared_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
  } else {
    matmul_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
  }
  cudaDeviceSynchronize();
}

void cuda_matmul_transposed(const float *A, const float *B, float *C, int M,
                            int N, int K) {
  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                (M + blockSize.y - 1) / blockSize.y);
  matmul_transposed_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

void cuda_transpose(const float *input, float *output, int rows, int cols) {
  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                (rows + blockSize.y - 1) / blockSize.y);
  transpose_kernel<<<gridSize, blockSize>>>(input, output, rows, cols);
  cudaDeviceSynchronize();
}

void cuda_add_bias(float *matrix, const float *bias, int rows, int cols) {
  int size = rows * cols;
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  add_bias_kernel<<<gridSize, blockSize>>>(matrix, bias, rows, cols);
  cudaDeviceSynchronize();
}

void cuda_scale(float *data, float scale, int size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  scale_kernel<<<gridSize, blockSize>>>(data, scale, size);
  cudaDeviceSynchronize();
}

void cuda_subtract(const float *A, const float *B, float *C, int size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  subtract_kernel<<<gridSize, blockSize>>>(A, B, C, size);
  cudaDeviceSynchronize();
}

void cuda_multiply(const float *A, const float *B, float *C, int size) {
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  multiply_kernel<<<gridSize, blockSize>>>(A, B, C, size);
  cudaDeviceSynchronize();
}

void cuda_sum_rows(const float *input, float *output, int rows, int cols) {
  int blockSize = 256;
  int gridSize = (rows + blockSize - 1) / blockSize;
  sum_rows_kernel<<<gridSize, blockSize>>>(input, output, rows, cols);
  cudaDeviceSynchronize();
}
}
