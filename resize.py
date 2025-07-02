import cv2
import os

# -------------------------
# Ask user for input/output folder and target size
# -------------------------
input_folder = input("Enter the path to the input image folder: ").strip('"').strip("'")
output_folder = input("Enter the path to save resized images: ").strip('"').strip("'")
width = int(input("Enter the desired width (e.g., 512): "))
height = int(input("Enter the desired height (e.g., 512): "))

# Supported image extensions
valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# -------------------------
# Walk through all subfolders
# -------------------------
for root, dirs, files in os.walk(input_folder):
    for file in files:
        name, ext = os.path.splitext(file)
        if ext.lower() not in valid_exts:
            continue

        input_path = os.path.join(root, file)

        # Construct the relative path to preserve folder structure
        rel_path = os.path.relpath(input_path, input_folder)
        output_path = os.path.join(output_folder, rel_path)

        # Create output subfolder if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read and resize
        image = cv2.imread(input_path)
        if image is None:
            print(f"Skipping unreadable image: {input_path}")
            continue

        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        # Save resized image
        cv2.imwrite(output_path, resized)
        print(f"Saved: {output_path}")
