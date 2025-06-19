#include "sign_classifier.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cmath>

SignClassifierNetwork::SignClassifierNetwork(int batch_sz) : batch_size(batch_sz) {
    std::cout << "Initializing sign classification network..." << std::endl;
    std::cout << "Input size: " << INPUT_SIZE << std::endl;
    std::cout << "Hidden layer size: " << HIDDEN_SIZE << std::endl;
    std::cout << "Output size: " << OUTPUT_SIZE << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    
    allocateMemory();
    initializeWeights();
    
    std::cout << "Network initialized successfully!" << std::endl;
}

SignClassifierNetwork::~SignClassifierNetwork() {
    freeMemory();
}

void SignClassifierNetwork::allocateMemory() {
    // Allocate memory for weights and biases
    cudaMalloc(&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float));
    
    // Allocate memory for gradients
    cudaMalloc(&d_dW1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_db1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dW2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db2, OUTPUT_SIZE * sizeof(float));
    
    // Allocate memory for activations
    cudaMalloc(&d_X, INPUT_SIZE * batch_size * sizeof(float));
    cudaMalloc(&d_Y, OUTPUT_SIZE * batch_size * sizeof(float));
    cudaMalloc(&d_L1u, HIDDEN_SIZE * batch_size * sizeof(float));
    cudaMalloc(&d_L1, HIDDEN_SIZE * batch_size * sizeof(float));
    cudaMalloc(&d_L2u, OUTPUT_SIZE * batch_size * sizeof(float));
    cudaMalloc(&d_L2, OUTPUT_SIZE * batch_size * sizeof(float));
    
    // Temporary memory
    int max_temp_size = std::max({
        HIDDEN_SIZE * batch_size,
        OUTPUT_SIZE * batch_size,
        HIDDEN_SIZE * INPUT_SIZE,
        OUTPUT_SIZE * HIDDEN_SIZE
    });
    
    cudaMalloc(&d_temp1, max_temp_size * sizeof(float));
    cudaMalloc(&d_temp2, max_temp_size * sizeof(float));
    cudaMalloc(&d_temp3, max_temp_size * sizeof(float));
    
    // Check for allocation errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA memory allocation failed");
    }
    
    std::cout << "Memory allocated successfully" << std::endl;
}

void SignClassifierNetwork::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // W1 initialization: Xavier initialization for ReLU
    float w1_std = sqrtf(2.0f / INPUT_SIZE);
    std::normal_distribution<float> w1_dist(0.0f, w1_std);
    std::vector<float> h_W1(HIDDEN_SIZE * INPUT_SIZE);
    for (auto& w : h_W1) w = w1_dist(gen);
    cudaMemcpy(d_W1, h_W1.data(), h_W1.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // W2 initialization: Xavier initialization for sigmoid
    float w2_std = sqrtf(1.0f / HIDDEN_SIZE);
    std::normal_distribution<float> w2_dist(0.0f, w2_std);
    std::vector<float> h_W2(OUTPUT_SIZE * HIDDEN_SIZE);
    for (auto& w : h_W2) w = w2_dist(gen);
    cudaMemcpy(d_W2, h_W2.data(), h_W2.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize biases to zero
    cudaMemset(d_b1, 0, HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_b2, 0, OUTPUT_SIZE * sizeof(float));
    
    std::cout << "Weights initialized" << std::endl;
}

void SignClassifierNetwork::forward_internal(const std::vector<float>& X, std::vector<float>& predictions) {
    // Copy input to GPU
    cudaMemcpy(d_X, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Forward pass: L1u = W1 * X + b1
    cuda_matmul(d_W1, d_X, d_L1u, HIDDEN_SIZE, batch_size, INPUT_SIZE);
    cuda_add_bias(d_L1u, d_b1, HIDDEN_SIZE, batch_size);
    
    // Forward pass: L1 = ReLU(L1u)
    cuda_relu(d_L1u, d_L1, HIDDEN_SIZE * batch_size);
    
    // Forward pass: L2u = W2 * L1 + b2
    cuda_matmul(d_W2, d_L1, d_L2u, OUTPUT_SIZE, batch_size, HIDDEN_SIZE);
    cuda_add_bias(d_L2u, d_b2, OUTPUT_SIZE, batch_size);
    
    // Forward pass: L2 = Sigmoid(L2u) - for binary classification
    cuda_sigmoid(d_L2u, d_L2, OUTPUT_SIZE * batch_size);
    
    // Copy results back to host
    predictions.resize(OUTPUT_SIZE * batch_size);
    cudaMemcpy(predictions.data(), d_L2, predictions.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

std::vector<float> SignClassifierNetwork::forward(const std::vector<float>& X) {
    std::vector<float> predictions;
    forward_internal(X, predictions);
    return predictions;
}

void SignClassifierNetwork::backward() {
    // Backpropagation for binary classification with sigmoid
    
    // dW2 = (1/m) * CL2 * L1^T
    cuda_matmul_transposed(d_temp1, d_L1, d_dW2, OUTPUT_SIZE, HIDDEN_SIZE, batch_size);
    cuda_scale(d_dW2, 1.0f / batch_size, OUTPUT_SIZE * HIDDEN_SIZE);
    
    // db2 = (1/m) * sum(CL2, axis=1)
    cuda_sum_rows(d_temp1, d_db2, OUTPUT_SIZE, batch_size);
    cuda_scale(d_db2, 1.0f / batch_size, OUTPUT_SIZE);
    
    // CL1 = W2^T * CL2 * dReLU(L1u)
    cuda_transpose(d_W2, d_temp3, OUTPUT_SIZE, HIDDEN_SIZE);
    cuda_matmul(d_temp3, d_temp1, d_temp2, HIDDEN_SIZE, batch_size, OUTPUT_SIZE);
    
    // Apply ReLU derivative
    cuda_relu_derivative(d_L1u, d_temp3, HIDDEN_SIZE * batch_size);
    cuda_multiply(d_temp2, d_temp3, d_temp2, HIDDEN_SIZE * batch_size);
    
    // dW1 = (1/m) * CL1 * X^T
    cuda_matmul_transposed(d_temp2, d_X, d_dW1, HIDDEN_SIZE, INPUT_SIZE, batch_size);
    cuda_scale(d_dW1, 1.0f / batch_size, HIDDEN_SIZE * INPUT_SIZE);
    
    // db1 = (1/m) * sum(CL1, axis=1)
    cuda_sum_rows(d_temp2, d_db1, HIDDEN_SIZE, batch_size);
    cuda_scale(d_db1, 1.0f / batch_size, HIDDEN_SIZE);
}

void SignClassifierNetwork::update_weights(float learning_rate) {
    // Weight updates: W = W - learning_rate * dW
    cuda_scale(d_dW1, learning_rate, HIDDEN_SIZE * INPUT_SIZE);
    cuda_subtract(d_W1, d_dW1, d_W1, HIDDEN_SIZE * INPUT_SIZE);
    
    cuda_scale(d_db1, learning_rate, HIDDEN_SIZE);
    cuda_subtract(d_b1, d_db1, d_b1, HIDDEN_SIZE);
    
    cuda_scale(d_dW2, learning_rate, OUTPUT_SIZE * HIDDEN_SIZE);
    cuda_subtract(d_W2, d_dW2, d_W2, OUTPUT_SIZE * HIDDEN_SIZE);
    
    cuda_scale(d_db2, learning_rate, OUTPUT_SIZE);
    cuda_subtract(d_b2, d_db2, d_b2, OUTPUT_SIZE);
}

void SignClassifierNetwork::train_step(const std::vector<float>& X, const std::vector<float>& Y, float learning_rate) {
    // Forward pass
    std::vector<float> predictions;
    forward_internal(X, predictions);
    
    // Copy labels to GPU
    cudaMemcpy(d_Y, Y.data(), Y.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute loss gradient: CL2 = L2 - Y
    cuda_subtract(d_L2, d_Y, d_temp1, OUTPUT_SIZE * batch_size);
    
    // Backward pass
    backward();
    
    // Update weights
    update_weights(learning_rate);
}

float SignClassifierNetwork::calculate_accuracy(const std::vector<float>& predictions, const std::vector<float>& labels) {
    int correct = 0;
    
    for (int sample = 0; sample < batch_size; sample++) {
        // For binary classification: prediction > 0.5 => class 1, otherwise class 0
        int pred_class = predictions[sample] > 0.5f ? 1 : 0;
        int true_class = labels[sample] > 0.5f ? 1 : 0;
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / batch_size;
}

void SignClassifierNetwork::train(const std::vector<float>& X, const std::vector<float>& Y, int epochs, float learning_rate) {
    std::cout << "Starting training for " << epochs << " epochs with learning rate " << learning_rate << std::endl;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        train_step(X, Y, learning_rate);
        
        if (epoch % 100 == 0) {
            std::vector<float> predictions;
            forward_internal(X, predictions);
            float accuracy = calculate_accuracy(predictions, Y);
            
            std::cout << "Epoch " << epoch << ", Accuracy: " << accuracy << std::endl;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
}

std::pair<std::vector<float>, std::vector<float>> SignClassifierNetwork::generate_training_data(int n_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> X(INPUT_SIZE * n_samples);
    std::vector<float> Y(OUTPUT_SIZE * n_samples);
    
    for (int i = 0; i < n_samples; i++) {
        // Generate two numbers from [-1, 1], different from zero
        float x1, x2;
        do {
            x1 = dist(gen);
        } while (std::abs(x1) < 0.01f);  // Avoid values very close to zero
        
        do {
            x2 = dist(gen);
        } while (std::abs(x2) < 0.01f);
        
        // Store in (features, samples) format
        X[0 * n_samples + i] = x1;  // first feature, i-th sample
        X[1 * n_samples + i] = x2;  // second feature, i-th sample
        
        // Label: 1 if same sign, 0 if different signs
        bool same_sign = (x1 > 0 && x2 > 0) || (x1 < 0 && x2 < 0);
        Y[i] = same_sign ? 1.0f : 0.0f;
    }
    
    std::cout << "Generated " << n_samples << " training samples" << std::endl;
    std::cout << "X size: " << X.size() << ", Y size: " << Y.size() << std::endl;
    
    return std::make_pair(X, Y);
}

void SignClassifierNetwork::freeMemory() {
    // Free GPU memory
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_dW1); cudaFree(d_db1); cudaFree(d_dW2); cudaFree(d_db2);
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_L1u); cudaFree(d_L1); cudaFree(d_L2u); cudaFree(d_L2);
    cudaFree(d_temp1); cudaFree(d_temp2); cudaFree(d_temp3);
}
