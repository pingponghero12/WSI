#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import fetch_openml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_mnist_sklearn():
    print("Loading MNIST dataset from sklearn...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    train_images = mnist.data[:60000]
    train_labels = mnist.target[:60000].astype(int)
    test_images = mnist.data[60000:]
    test_labels = mnist.target[60000:].astype(int)
    
    print(f"Training data shape: {train_images.shape}")
    print(f"Test data shape: {test_images.shape}")
    print(f"Training labels range: {train_labels.min()} to {train_labels.max()}")
    
    return (train_images, train_labels), (test_images, test_labels)

def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((num_classes, len(labels)), dtype=np.float32)
    for i, label in enumerate(labels):
        encoded[int(label), i] = 1.0
    return encoded

def main():
    print("MNIST Neural Network Training with CUDA")
    
    try:
        import cuda_neural_network as cnn
        print("CUDA module imported")
    except ImportError as e:
        print(f"Import failed: {e}")
        return
    
    # Load data exactly like the first script
    (train_images, train_labels), (test_images, test_labels) = load_mnist_sklearn()
    
    # Use exact same data preparation as first script
    train_images = train_images.reshape(60000, 784).T.astype('float32') / 255
    n_train_samples = 2000
    
    X = train_images[:, :n_train_samples]  # Shape: (784, 2000)
    Y = one_hot_encode(train_labels[:n_train_samples], 10)  # Shape: (10, 2000)
    
    print("Starting training...")
    print("Training data shape:", X.shape)
    
    # Initialize network with same architecture as first script
    net = cnn.CudaNeuralNetwork(784, 128, 10, n_train_samples)
    
    # Train with exact same parameters as first script
    net.train(X.flatten().tolist(), Y.flatten().tolist(), 2000, 0.35)
    
    # Test on same data as training (like first script does during training)
    predictions = net.forward(X.flatten().tolist())
    predictions_array = np.array(predictions).reshape(10, n_train_samples)
    pred_classes = np.argmax(predictions_array, axis=0)
    final_accuracy = np.mean(pred_classes == train_labels[:n_train_samples])
    
    print(f"\nFinal training accuracy: {final_accuracy:.4f}")
    
    # FIXED: Test in batches using the SAME trained network
    print("\nPreparing test data...")
    test_images = test_images.reshape(10000, 784).T.astype('float32') / 255
    
    print("Running inference on test data...")
    
    # Test in smaller batches to use the same trained network
    batch_size = 1000  # Use smaller batches
    all_predictions = []
    
    for i in range(0, 10000, batch_size):
        end_idx = min(i + batch_size, 10000)
        batch_test_images = test_images[:, i:end_idx]
        
        # Pad to match training batch size if needed
        current_batch_size = end_idx - i
        if current_batch_size < n_train_samples:
            # Pad with zeros to match expected input size
            padding_needed = n_train_samples - current_batch_size
            padding = np.zeros((784, padding_needed), dtype=np.float32)
            batch_test_images_padded = np.concatenate([batch_test_images, padding], axis=1)
        else:
            batch_test_images_padded = batch_test_images
        
        # Use the TRAINED network
        batch_predictions = net.forward(batch_test_images_padded.flatten().tolist())
        batch_predictions_array = np.array(batch_predictions).reshape(10, n_train_samples)
        
        # Only take predictions for actual samples (not padding)
        actual_predictions = batch_predictions_array[:, :current_batch_size]
        all_predictions.append(actual_predictions)
    
    # Combine all predictions
    all_predictions = np.concatenate(all_predictions, axis=1)
    test_pred_classes = np.argmax(all_predictions, axis=0)
    test_accuracy = np.mean(test_pred_classes == test_labels)
    
    print("Test Accuracy:", test_accuracy)

if __name__ == "__main__":
    main()
