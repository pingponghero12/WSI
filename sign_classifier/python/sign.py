import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("CUDA Neural Network - Sign Classification")
    print("Problem: Same sign detection for two numbers in [-1,1]")
    print("Architecture: 2 -> 4 (ReLU) -> 1 (Sigmoid)")
    print()
    
    try:
        import sign_classifier as sc
        print("✅ Module imported")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Generate data
    n_samples = 1000
    print(f"Generating {n_samples} samples...")
    X, Y = sc.SignClassifierNetwork.generate_training_data(n_samples)
    
    if len(X) == 0 or len(Y) == 0:
        print("❌ No data generated")
        return
    
    # Initialize network
    net = sc.SignClassifierNetwork(n_samples)
    
    # Test before training
    predictions = net.forward(X)
    accuracy_before = net.calculate_accuracy(predictions, Y)
    print(f"Accuracy before training: {accuracy_before:.4f}")
    
    # Train
    print("Training...")
    net.train(X, Y, epochs=1000, learning_rate=1.0)
    
    # Test after training
    predictions = net.forward(X)
    accuracy_after = net.calculate_accuracy(predictions, Y)
    print(f"Accuracy after training:  {accuracy_after:.4f}")
    print(f"Improvement: +{accuracy_after - accuracy_before:.4f}")

if __name__ == "__main__":
    main()
