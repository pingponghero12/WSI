#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

class SignClassifierNetwork {
private:
    // Network parameters: 2 inputs, 4 hidden neurons, 1 output
    static const int INPUT_SIZE = 2;
    static const int HIDDEN_SIZE = 4;
    static const int OUTPUT_SIZE = 1;
    
    int batch_size;
    
    // GPU memory pointers
    float *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_dW1, *d_db1, *d_dW2, *d_db2;
    float *d_X, *d_Y, *d_L1u, *d_L1, *d_L2u, *d_L2;
    float *d_temp1, *d_temp2, *d_temp3;
    
    void allocateMemory();
    void initializeWeights();
    void freeMemory();
    void backward();
    void update_weights(float learning_rate);
    
public:
    SignClassifierNetwork(int batch_sz);
    ~SignClassifierNetwork();
    
    std::vector<float> forward(const std::vector<float>& X);
    void train_step(const std::vector<float>& X, const std::vector<float>& Y, float learning_rate);
    float calculate_accuracy(const std::vector<float>& predictions, const std::vector<float>& labels);
    void train(const std::vector<float>& X, const std::vector<float>& Y, int epochs, float learning_rate);
    
    // CHANGED: Return a pair of vectors instead of using references
    static std::pair<std::vector<float>, std::vector<float>> generate_training_data(int n_samples);
    
private:
    void forward_internal(const std::vector<float>& X, std::vector<float>& predictions);
};

// CUDA kernel declarations
extern "C" {
    void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K);
    void cuda_matmul_transposed(const float* A, const float* B, float* C, int M, int N, int K);
    void cuda_transpose(const float* input, float* output, int rows, int cols);
    void cuda_add_bias(float* matrix, const float* bias, int rows, int cols);
    void cuda_relu(const float* input, float* output, int size);
    void cuda_relu_derivative(const float* input, float* output, int size);
    void cuda_sigmoid(const float* input, float* output, int size);
    void cuda_sigmoid_derivative(const float* input, float* output, int size);
    void cuda_subtract(const float* A, const float* B, float* C, int size);
    void cuda_multiply(const float* A, const float* B, float* C, int size);
    void cuda_scale(float* data, float scale, int size);
    void cuda_sum_rows(const float* input, float* output, int rows, int cols);
}
