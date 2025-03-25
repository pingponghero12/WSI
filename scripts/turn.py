import os
from PIL import Image

def rotate_images_in_directory(root_dir):
    """
    Rotate all images in a directory structure 90 degrees counter-clockwise.
    
    Args:
        root_dir: Root directory containing subdirectories with images
    """
    print(f"Looking for images in {root_dir}...")
    
    # Track statistics
    total_images = 0
    rotated_images = 0
    
    # Process each digit directory
    for digit in range(10):
        digit_dir = os.path.join(root_dir, str(digit))
        
        # Skip if directory doesn't exist
        if not os.path.exists(digit_dir):
            print(f"Directory for digit {digit} not found. Skipping.")
            continue
        
        print(f"Processing digit {digit}...")
        
        # Process each image in the digit directory
        for filename in os.listdir(digit_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(digit_dir, filename)
                total_images += 1
                
                try:
                    # Open the image
                    with Image.open(file_path) as img:
                        # Rotate 90 degrees counter-clockwise
                        rotated = img.rotate(90, expand=True)
                        
                        # Save the rotated image, overwriting the original
                        rotated.save(file_path)
                        rotated_images += 1
                        print(f"  Rotated: {file_path}")
                except Exception as e:
                    print(f"  Error processing {file_path}: {e}")
    
    print(f"\nRotation complete!")
    print(f"Total images found: {total_images}")
    print(f"Successfully rotated: {rotated_images}")

if __name__ == "__main__":
    # Directory containing digit subdirectories
    image_root = "my_digits"
    
    # Check if directory exists
    if not os.path.exists(image_root):
        print(f"Error: Directory '{image_root}' not found.")
        print("Please create the directory structure first.")
        exit(1)
    
    # Confirm before proceeding
    print(f"This will rotate ALL images in '{image_root}' and its subdirectories 90° counter-clockwise.")
    confirm = input("Do you want to continue? (y/n): ")
    
    if confirm.lower() == 'y':
        rotate_images_in_directory(image_root)
    else:
        print("Operation cancelled.")
