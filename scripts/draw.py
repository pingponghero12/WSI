import pygame
import os
import numpy as np
from PIL import Image
import sys

pygame.init()

# Settings
WIDTH, HEIGHT = 280, 280
BACKGROUND_COLOR = (255, 255, 255)
DRAWING_COLOR = (0, 0, 0)
CURSOR_SIZE = 20

# Set up the drawing window
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Draw a Digit (0-9)")

# Create a surface for drawing
drawing_surface = pygame.Surface((WIDTH, HEIGHT))
drawing_surface.fill(BACKGROUND_COLOR)

def save_drawing(digit):
    # Create directory if it doesn't exist
    os.makedirs(f'my_digits/{digit}', exist_ok=True)
    
    # Count existing files to generate a unique filename
    count = len([f for f in os.listdir(f'my_digits/{digit}') 
                if f.startswith(f'{digit}_') and (f.endswith('.png') or f.endswith('.jpg'))])
    
    # Save the pygame surface as an image
    pygame_img = pygame.surfarray.array3d(drawing_surface)
    img = Image.fromarray(pygame_img)
    filename = f'my_digits/{digit}/{digit}_{count+1}.png'
    img.save(filename)
    print(f"Saved to {filename}")

# Main loop
running = True
digit = None

# Ask for digit
print("Enter the digit you're about to draw (0-9):")
digit = input()
if not digit.isdigit() or len(digit) != 1:
    print("Invalid input. Please enter a single digit (0-9).")
    pygame.quit()
    sys.exit()

print("Use mouse to draw. Press S to save, C to clear, Q to quit.")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:  # Save
                save_drawing(digit)
            elif event.key == pygame.K_c:  # Clear
                drawing_surface.fill(BACKGROUND_COLOR)
            elif event.key == pygame.K_q:  # Quit
                running = False
    
    # Draw when mouse button is pressed
    if pygame.mouse.get_pressed()[0]:
        x, y = pygame.mouse.get_pos()
        # Draw a circle at cursor position
        pygame.draw.circle(drawing_surface, DRAWING_COLOR, (x, y), CURSOR_SIZE // 2)
    
    # Copy the drawing surface to the screen
    screen.blit(drawing_surface, (0, 0))
    
    # Update the display
    pygame.display.flip()

pygame.quit()

# Ensure current drawing is saved before exiting
save_choice = input("Save current drawing before exit? (y/n): ")
if save_choice.lower() == 'y':
    save_drawing(digit)
