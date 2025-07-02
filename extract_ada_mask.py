import cv2
import numpy as np
import os

# -------------------------
# Ask user for input/output folders
# -------------------------
input_folder = input("Enter the path to the folder with overlaid images: ").strip('"').strip("'")
output_folder = input("Enter the path to save binary masks: ").strip('"').strip("'")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image extensions
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']

# -------------------------
# Process all image files
# -------------------------
for filename in os.listdir(input_folder):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in valid_exts:
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, name + '.png')

    image = cv2.imread(input_path)
    if image is None:
        print(f"Skipping unreadable image: {filename}")
        continue

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red color range (two halves)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Create binary mask: red → black (0), rest → white (255)
    binary_mask = np.full_like(red_mask, 255)
    binary_mask[red_mask > 0] = 0

    cv2.imwrite(output_path, binary_mask)
    print(f"Saved: {output_path}")
