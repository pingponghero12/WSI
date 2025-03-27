import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, confusion_matrix, classification_report

def load_images_from_folder(folder):
    """Load all images from a folder structure where each subfolder is a digit class"""
    images = []
    labels = []
    
    # Loop through each digit folder (0-9)
    for digit in range(10):
        digit_path = os.path.join(folder, str(digit))
        if not os.path.exists(digit_path):
            continue
            
        # Loop through each image in the digit folder
        for file in os.listdir(digit_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(digit_path, file)
                try:
                    # Open and preprocess image
                    image = Image.open(file_path).convert('L')  # Convert to grayscale
                    image = image.resize((28, 28))              # Resize to MNIST format
                    
                    # Convert to numpy array and normalize
                    img_array = np.array(image)
                    
                    # Invert colors if needed (MNIST has white digits on black)
                    if np.mean(img_array) > 127:
                        img_array = 255 - img_array
                        
                    img_array = img_array / 255.0
                    
                    # Add to dataset
                    images.append(img_array)
                    labels.append(digit)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    if not images:
        return None, None
        
    return np.array(images), np.array(labels)

def evaluate_model():
    # Define model path (directory without .keras extension)
    model_path = 'mnist_random_forest_ydf'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
        
    # Check if images directory exists
    if not os.path.exists('my_digits'):
        print("Image directory 'my_digits' not found. Please create it and add your handwritten digits.")
        return
    
    # Import YDF first to ensure it's available
    print("Importing YDF library...")
    try:
        import ydf
    except ImportError:
        print("ERROR: ydf module not found. Please install it with: pip install ydf")
        return
    
    # Load the YDF model
    print(f"Loading model from {model_path}...")
    try:
        model = ydf.load_model(model_path)
        print("Model loaded successfully with ydf.load_model")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load custom dataset
    print("Loading images...")
    images, labels = load_images_from_folder('my_digits')
    
    if images is None or len(images) == 0:
        print("No images found in the 'my_digits' directory")
        return
        
    print(f"Loaded {len(images)} images")
    
    # Prepare inputs for model - flatten images for YDF
    images_flat = images.reshape(len(images), 28*28)
    
    # Create a DataFrame with column names that match the training data
    feature_names = [f'pixel_{i}' for i in range(28*28)]
    test_df = pd.DataFrame(images_flat, columns=feature_names)
    
    # Make predictions using YDF model
    print("Making predictions...")
    try:
        # First try with probability output
        predictions_proba = model.predict(test_df, output="probability")
        print(f"Prediction shape: {predictions_proba.shape}")
        
        # Check if we got probabilities (which would be 2D) or class predictions (1D)
        if len(predictions_proba.shape) > 1 and predictions_proba.shape[1] > 1:
            # We got probabilities, need to convert to class indices
            pred_classes = np.argmax(predictions_proba, axis=1)
            print("Converted probabilities to class predictions")
        else:
            # We already have class predictions
            pred_classes = predictions_proba
            print("Model returned class predictions directly")
            
    except Exception as e:
        print(f"Prediction with probability output failed: {e}")
        
        # Try standard prediction method
        try:
            pred_classes = model.predict(test_df)
            print("Used standard prediction method")
        except Exception as e2:
            print(f"Standard prediction method also failed: {e2}")
            return
    
    # Print shapes for debugging
    print(f"pred_classes shape: {pred_classes.shape}")
    print(f"labels shape: {labels.shape}")
    
    # Make sure pred_classes is 1D to match labels
    if len(pred_classes.shape) > 1:
        if pred_classes.shape[1] == 1:
            # If shape is (n, 1), flatten to (n,)
            pred_classes = pred_classes.flatten()
        else:
            # If shape is (n, k) where k > 1, convert to class indices
            pred_classes = np.argmax(pred_classes, axis=1)
    
    # Double-check shapes match
    print(f"Final pred_classes shape: {pred_classes.shape}")
    
    # Calculate metrics
    accuracy = np.mean(pred_classes == labels)
    
    # For precision, handle case where some classes might not be present
    classes_present = np.unique(labels)
    precision = precision_score(labels, pred_classes, average='macro', labels=classes_present, zero_division=0)
    
    # Print results
    print("\n===== EVALUATION RESULTS =====")
    print(f"Total images: {len(images)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, pred_classes)
    print(cm)
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(labels, pred_classes, zero_division=0))
    
    # Visualize some predictions
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(images))):
        plt.subplot(5, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"True: {labels[i]}\nPred: {pred_classes[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
