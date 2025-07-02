import os
import cv2
import numpy as np

# -------------------------
# Ask user for folder paths
# -------------------------
image_folder = input("Enter the path to the image folder: ").strip('"').strip("'")
mask_folder = input("Enter the path to the mask folder: ").strip('"').strip("'")
output_folder = input("Enter the path to the output folder: ").strip('"').strip("'")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# -------------------------
# Process each image-mask pair
# -------------------------
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    mask_path = os.path.join(mask_folder, filename)

    if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
        print(f"Skipping {filename}: corresponding file not found.")
        continue

    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Skipping {filename}: failed to load image or mask.")
        continue

    # Resize mask if dimensions don't match
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create object mask from black regions in mask
    object_mask = (mask == 0).astype(np.uint8)
    object_mask_3ch = np.repeat(object_mask[:, :, np.newaxis], 3, axis=2)

    # Red overlay
    red_overlay = np.zeros_like(image)
    red_overlay[:] = [0, 0, 255]

    # Apply red overlay where object is
    result = image.copy()
    result[object_mask_3ch == 1] = red_overlay[object_mask_3ch == 1]

    # Save result
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")
