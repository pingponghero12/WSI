#include "cuda_neural_network.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cmath>

CudaNeuralNetwork::CudaNeuralNetwork(int input_sz, int hidden_sz, int output_sz, int batch_sz)
    : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz), batch_size(batch_sz) {
    
    std::cout << "Initializing CUDA Neural Network..." << std::endl;
    std::cout << "Input size: " << input_size << std::endl;
    std::cout << "Hidden size: " << hidden_size << std::endl;
    std::cout << "Output size: " << output_size << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    
    allocateMemory();
    initializeWeights();
    
    std::cout << "CUDA Neural Network initialized successfully!" << std::endl;
}

CudaNeuralNetwork::~CudaNeuralNetwork() {
    freeMemory();
}

void CudaNeuralNetwork::allocateMemory() {
    // Allocate device memory for weights and biases
    cudaMalloc(&d_W1, hidden_size * input_size * sizeof(float));
    cudaMalloc(&d_b1, hidden_size * sizeof(float));
    cudaMalloc(&d_W2, output_size * hidden_size * sizeof(float));
    cudaMalloc(&d_b2, output_size * sizeof(float));
    
    // Allocate device memory for gradients
    cudaMalloc(&d_dW1, hidden_size * input_size * sizeof(float));
    cudaMalloc(&d_db1, hidden_size * sizeof(float));
    cudaMalloc(&d_dW2, output_size * hidden_size * sizeof(float));
    cudaMalloc(&d_db2, output_size * sizeof(float));
    
    // Allocate device memory for activations
    cudaMalloc(&d_X, input_size * batch_size * sizeof(float));
    cudaMalloc(&d_Y, output_size * batch_size * sizeof(float));
    cudaMalloc(&d_L1u, hidden_size * batch_size * sizeof(float));
    cudaMalloc(&d_L1, hidden_size * batch_size * sizeof(float));
    cudaMalloc(&d_L2u, output_size * batch_size * sizeof(float));
    cudaMalloc(&d_L2, output_size * batch_size * sizeof(float));
    
    // Temporary memory for computations
    int max_temp_size = std::max({
        hidden_size * batch_size,
        output_size * batch_size,
        hidden_size * input_size,
        output_size * hidden_size
    });
    
    cudaMalloc(&d_temp1, max_temp_size * sizeof(float));
    cudaMalloc(&d_temp2, max_temp_size * sizeof(float));
    cudaMalloc(&d_temp3, max_temp_size * sizeof(float));
    
    // For softmax computation
    cudaMalloc(&d_max_vals, std::max(batch_size, output_size) * sizeof(float));
    cudaMalloc(&d_exp_sums, std::max(batch_size, output_size) * sizeof(float));
    
    // Check for allocation errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA memory allocation failed");
    }
    
}

void CudaNeuralNetwork::initializeWeights() {
    // Initialize weights using Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // W1 initialization: Xavier initialization for ReLU
    float w1_std = sqrtf(2.0f / input_size);
    std::normal_distribution<float> w1_dist(0.0f, w1_std);
    std::vector<float> h_W1(hidden_size * input_size);
    for (auto& w : h_W1) w = w1_dist(gen);
    cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // W2 initialization: Xavier initialization for softmax
    float w2_std = sqrtf(1.0f / hidden_size);
    std::normal_distribution<float> w2_dist(0.0f, w2_std);
    std::vector<float> h_W2(output_size * hidden_size);
    for (auto& w : h_W2) w = w2_dist(gen);
    cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize biases to zero
    cudaMemset(d_b1, 0, hidden_size * sizeof(float));
    cudaMemset(d_b2, 0, output_size * sizeof(float));
    
}

void CudaNeuralNetwork::forward_internal(const std::vector<float>& X, std::vector<float>& predictions) {
    cudaError_t error = cudaMemcpy(d_X, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        return;
    }
    
    // Forward pass: L1u = W1 * X + b1
    cuda_matmul(d_W1, d_X, d_L1u, hidden_size, batch_size, input_size);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return;
    }
    
    cuda_add_bias(d_L1u, d_b1, hidden_size, batch_size);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return;
    }
    
    // Forward pass: L1 = ReLU(L1u)
    cuda_relu(d_L1u, d_L1, hidden_size * batch_size);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return;
    }
    
    // Forward pass: L2u = W2 * L1 + b2
    cuda_matmul(d_W2, d_L1, d_L2u, output_size, batch_size, hidden_size);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return;
    }
    
    cuda_add_bias(d_L2u, d_b2, output_size, batch_size);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return;
    }
    
    // Forward pass: L2 = Softmax(L2u)
    cuda_softmax(d_L2u, d_L2, d_max_vals, d_exp_sums, output_size, batch_size);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "❌ Softmax failed: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    // Copy results back to host
    predictions.resize(output_size * batch_size);
    
    error = cudaMemcpy(predictions.data(), d_L2, 
                      predictions.size() * sizeof(float), 
                      cudaMemcpyDeviceToHost);
    
    if (error != cudaSuccess) {
        predictions.clear();
        return;
    }
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Check if we actually have data
    if (predictions.size() > 0) {
        for (int i = 0; i < std::min(5, (int)predictions.size()); i++) {
        }
    } else {
    }
    
}

// NEW: Public forward function that returns predictions
std::vector<float> CudaNeuralNetwork::forward(const std::vector<float>& X) {
    std::vector<float> predictions;
    forward_internal(X, predictions);
    
    std::cout << "Public forward returning " << predictions.size() << " predictions" << std::endl;
    
    return predictions;  // This should work better with pybind11
}


void CudaNeuralNetwork::backward() {
    // Backward pass implementation
    // CL2 = L2 - Y (already computed and stored in d_temp1)
    
    // dW2 = (1/m) * CL2 * L1^T
    cuda_matmul_transposed(d_temp1, d_L1, d_dW2, output_size, hidden_size, batch_size);
    cuda_scale(d_dW2, 1.0f / batch_size, output_size * hidden_size);
    
    // db2 = (1/m) * sum(CL2, axis=1)
    cuda_sum_rows(d_temp1, d_db2, output_size, batch_size);
    cuda_scale(d_db2, 1.0f / batch_size, output_size);
    
    // CL1 = W2^T * CL2 * dReLU(L1u)
    cuda_matmul(d_W2, d_temp1, d_temp2, hidden_size, batch_size, output_size); // W2^T * CL2 (note: we need transpose)
    cuda_transpose(d_W2, d_temp3, output_size, hidden_size); // Transpose W2
    cuda_matmul(d_temp3, d_temp1, d_temp2, hidden_size, batch_size, output_size); // W2^T * CL2
    
    // Apply ReLU derivative
    cuda_relu_derivative(d_L1u, d_temp3, hidden_size * batch_size);
    cuda_multiply(d_temp2, d_temp3, d_temp2, hidden_size * batch_size); // Element-wise multiply
    
    // dW1 = (1/m) * CL1 * X^T
    cuda_matmul_transposed(d_temp2, d_X, d_dW1, hidden_size, input_size, batch_size);
    cuda_scale(d_dW1, 1.0f / batch_size, hidden_size * input_size);
    
    // db1 = (1/m) * sum(CL1, axis=1)
    cuda_sum_rows(d_temp2, d_db1, hidden_size, batch_size);
    cuda_scale(d_db1, 1.0f / batch_size, hidden_size);
}

void CudaNeuralNetwork::update_weights(float learning_rate) {
    // W1 = W1 - learning_rate * dW1
    cuda_scale(d_dW1, learning_rate, hidden_size * input_size);
    cuda_subtract(d_W1, d_dW1, d_W1, hidden_size * input_size);
    
    // b1 = b1 - learning_rate * db1
    cuda_scale(d_db1, learning_rate, hidden_size);
    cuda_subtract(d_b1, d_db1, d_b1, hidden_size);
    
    // W2 = W2 - learning_rate * dW2
    cuda_scale(d_dW2, learning_rate, output_size * hidden_size);
    cuda_subtract(d_W2, d_dW2, d_W2, output_size * hidden_size);
    
    // b2 = b2 - learning_rate * db2
    cuda_scale(d_db2, learning_rate, output_size);
    cuda_subtract(d_b2, d_db2, d_b2, output_size);
}

void CudaNeuralNetwork::train_step(const std::vector<float>& X, const std::vector<float>& Y, float learning_rate) {
    // Forward pass
    std::vector<float> predictions;
    forward_internal(X, predictions);
    
    // Copy labels to device
    cudaMemcpy(d_Y, Y.data(), Y.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute loss gradient: CL2 = L2 - Y
    cuda_subtract(d_L2, d_Y, d_temp1, output_size * batch_size);
    
    // Backward pass
    backward();
    
    // Update weights
    update_weights(learning_rate);
}

float CudaNeuralNetwork::calculate_accuracy(const std::vector<float>& predictions, const std::vector<float>& labels) {
    int correct = 0;
    
    for (int sample = 0; sample < batch_size; sample++) {
        // Find predicted class (argmax of predictions)
        int pred_class = 0;
        float max_pred = predictions[sample];
        for (int cls = 1; cls < output_size; cls++) {
            if (predictions[cls * batch_size + sample] > max_pred) {
                max_pred = predictions[cls * batch_size + sample];
                pred_class = cls;
            }
        }
        
        // Find true class (argmax of labels)
        int true_class = 0;
        float max_true = labels[sample];
        for (int cls = 1; cls < output_size; cls++) {
            if (labels[cls * batch_size + sample] > max_true) {
                max_true = labels[cls * batch_size + sample];
                true_class = cls;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / batch_size;
}

void CudaNeuralNetwork::train(const std::vector<float>& X, const std::vector<float>& Y, int epochs, float learning_rate) {
    std::cout << "Starting training for " << epochs << " epochs with learning rate " << learning_rate << std::endl;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Training step
        train_step(X, Y, learning_rate);
        
        // Calculate and print accuracy every 50 epochs
        if (epoch % 50 == 0) {
            std::vector<float> predictions;
            forward_internal(X, predictions);
            float accuracy = calculate_accuracy(predictions, Y);
            
            std::cout << "Epoch " << epoch << ", Accuracy: " << accuracy << std::endl;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
}

void CudaNeuralNetwork::freeMemory() {
    // Free all allocated device memory
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_dW1); cudaFree(d_db1); cudaFree(d_dW2); cudaFree(d_db2);
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_L1u); cudaFree(d_L1); cudaFree(d_L2u); cudaFree(d_L2);
    cudaFree(d_temp1); cudaFree(d_temp2); cudaFree(d_temp3);
    cudaFree(d_max_vals); cudaFree(d_exp_sums);
}
