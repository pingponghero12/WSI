#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import fetch_openml
import sys
import os

# Add current directory to Python path for module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_mnist_sklearn():
    """Load MNIST data using sklearn instead of TensorFlow"""
    print("Loading MNIST dataset from sklearn...")
    
    # Load MNIST data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Split into train and test (sklearn MNIST comes as one dataset)
    train_images = mnist.data[:60000]
    train_labels = mnist.target[:60000].astype(int)
    test_images = mnist.data[60000:]
    test_labels = mnist.target[60000:].astype(int)
    
    print(f"Training data shape: {train_images.shape}")
    print(f"Test data shape: {test_images.shape}")
    print(f"Training labels range: {train_labels.min()} to {train_labels.max()}")
    
    return (train_images, train_labels), (test_images, test_labels)

def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding in (num_classes, num_samples) format"""
    encoded = np.zeros((num_classes, len(labels)), dtype=np.float32)
    for i, label in enumerate(labels):
        encoded[int(label), i] = 1.0
    return encoded

def main():
    print("MNIST Neural Network Training with CUDA (Debug Version)")
    print("=" * 60)
    
    # Try to import CUDA module (should be in same directory)
    try:
        import cuda_neural_network as cnn
        print("✅ CUDA Neural Network module imported successfully!")
    except ImportError as e:
        print(f"❌ Failed to import CUDA Neural Network: {e}")
        return
    
    # Load data
    (train_images, train_labels), (test_images, test_labels) = load_mnist_sklearn()
    
    # Use smaller subset for debugging
    n_train_samples = 100  # Much smaller for debugging
    train_images_subset = train_images[:n_train_samples]
    train_labels_subset = train_labels[:n_train_samples]
    
    # Prepare data in the format expected by CUDA implementation
    X = train_images_subset.T.astype('float32') / 255.0  # Shape: (784, 100)
    Y = one_hot_encode(train_labels_subset, 10)          # Shape: (10, 100)
    
    print(f"Training input shape: {X.shape}")
    print(f"Training labels shape: {Y.shape}")
    
    # Initialize CUDA Neural Network
    print("\nInitializing CUDA Neural Network...")
    try:
        net = cnn.CudaNeuralNetwork(
            input_size=784,
            hidden_size=128,
            output_size=10,
            batch_size=n_train_samples
        )
        print("✅ Neural network initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize neural network: {e}")
        return
    
    # Test forward pass BEFORE training
    print("\n=== Testing forward pass BEFORE training ===")
    try:
        predictions = []
        print(f"Input data size: {len(X.flatten().tolist())}")
        print("Calling forward...")
        
        net.forward(X.flatten().tolist(), predictions)
        
        print(f"Predictions returned: {len(predictions)} elements")
        if len(predictions) > 0:
            print(f"First few predictions: {predictions[:10]}")
            predictions_array = np.array(predictions).reshape(10, n_train_samples)
            print(f"Reshaped predictions shape: {predictions_array.shape}")
        else:
            print("❌ No predictions returned!")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Train for just a few epochs
    print(f"\n=== Training for 10 epochs ===")
    try:
        net.train(
            X.flatten().tolist(), 
            Y.flatten().tolist(), 
            epochs=10,  # Just 10 epochs for debugging
            learning_rate=0.1
        )
        print("✅ Training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Test forward pass AFTER training
    print("\n=== Testing forward pass AFTER training ===")
    try:
        predictions = []
        print("Calling forward after training...")
        
        net.forward(X.flatten().tolist(), predictions)
        
        print(f"Predictions returned: {len(predictions)} elements")
        if len(predictions) > 0:
            print(f"First few predictions: {predictions[:10]}")
            predictions_array = np.array(predictions).reshape(10, n_train_samples)
            print(f"Reshaped predictions shape: {predictions_array.shape}")
            
            # Calculate accuracy
            pred_classes = np.argmax(predictions_array, axis=0)
            true_classes = train_labels_subset
            accuracy = np.mean(pred_classes == true_classes)
            
            print(f"Training Accuracy: {accuracy:.4f}")
            print("✅ Debug test completed successfully!")
        else:
            print("❌ Still no predictions returned after training!")
            
    except Exception as e:
        print(f"❌ Post-training forward pass failed: {e}")

if __name__ == "__main__":
    main()
