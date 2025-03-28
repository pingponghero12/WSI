import os
import numpy as np
import tensorflow as tf
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
                    images.append(img_array.reshape(28, 28, 1))
                    labels.append(digit)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    if not images:
        return None, None
        
    return np.array(images), np.array(labels)

def evaluate_model():
    # Check if model exists
    if not os.path.exists('simplenet_mnist.keras'):
        print("Model not found. Please train the model first.")
        return
        
    # Check if images directory exists
    if not os.path.exists('my_digits'):
        print("Image directory 'my_digits' not found. Please create it and add your handwritten digits.")
        return
        
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model('simplenet_mnist.keras')
    model.summary()
    
    # Load custom dataset
    print("Loading images...")
    images, labels = load_images_from_folder('my_digits')
    
    if images is None or len(images) == 0:
        print("No images found in the 'my_digits' directory")
        return
        
    print(f"Loaded {len(images)} images")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(images)
    pred_classes = np.argmax(predictions, axis=1)
    
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
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {labels[i]}\nPred: {pred_classes[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
