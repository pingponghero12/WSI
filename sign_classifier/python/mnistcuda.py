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
    
    return (train_images, train_labels), (test_images, test_labels)

def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((num_classes, len(labels)), dtype=np.float32)
    for i, label in enumerate(labels):
        encoded[int(label), i] = 1.0
    return encoded

def main():
    print("MNIST Neural Network Training with CUDA (Final)")
    print("=" * 55)
    
    try:
        import cuda_neural_network as cnn
        print("✅ CUDA Neural Network module imported successfully!")
    except ImportError as e:
        print(f"❌ Failed to import CUDA Neural Network: {e}")
        return
    
    # Load data
    (train_images, train_labels), (test_images, test_labels) = load_mnist_sklearn()
    
    # Use consistent batch size
    n_train_samples = 2000
    train_images_subset = train_images[:n_train_samples]
    train_labels_subset = train_labels[:n_train_samples]
    
    # Prepare data
    X = train_images_subset.T.astype('float32') / 255.0  # Shape: (784, 2000)
    Y = one_hot_encode(train_labels_subset, 10)          # Shape: (10, 2000)
    
    print(f"Training input shape: {X.shape}")
    print(f"Training labels shape: {Y.shape}")
    
    # Initialize CUDA Neural Network
    print("\nInitializing CUDA Neural Network...")
    net = cnn.CudaNeuralNetwork(784, 128, 10, n_train_samples)
    
    # Test forward pass BEFORE training
    print("\n=== Testing Forward Pass BEFORE Training ===")
    predictions = net.forward(X.flatten().tolist())  # Now returns predictions directly
    
    print(f"Returned predictions: {len(predictions)} elements")
    if len(predictions) > 0:
        predictions_array = np.array(predictions).reshape(10, n_train_samples)
        print(f"✅ Reshaped to: {predictions_array.shape}")
        print(f"Sample predictions (first column): {predictions_array[:, 0]}")
        print(f"Sum of first column: {np.sum(predictions_array[:, 0])}")
    else:
        print("❌ No predictions returned")
        return
    
    # Train the network
    print(f"\n=== Training for 100 epochs ===")
    try:
        net.train(X.flatten().tolist(), Y.flatten().tolist(), 100, 0.1)
        print("✅ Training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Test forward pass AFTER training
    print("\n=== Testing Forward Pass AFTER Training ===")
    predictions = net.forward(X.flatten().tolist())
    
    print(f"Returned predictions: {len(predictions)} elements")
    if len(predictions) > 0:
        predictions_array = np.array(predictions).reshape(10, n_train_samples)
        
        # Calculate accuracy
        pred_classes = np.argmax(predictions_array, axis=0)
        true_classes = train_labels_subset
        accuracy = np.mean(pred_classes == true_classes)
        
        print(f"✅ Training Accuracy: {accuracy:.4f}")
        print("🎉 CUDA Neural Network working perfectly!")
    else:
        print("❌ No predictions returned after training")

if __name__ == "__main__":
    main()
