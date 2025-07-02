import numpy as np
import cv2
import os
from glob import glob

def raw3d_to_jpg_resized(raw_path, output_dir, width, height, slice_index=0, out_width=1024, out_height=512):
    try:
        raw_data = np.fromfile(raw_path, dtype=np.uint16)
    except Exception as e:
        print(f"Error reading file {raw_path}: {e}")
        return

    total_slices = raw_data.size // (width * height)
    if total_slices == 0:
        print(f"File {raw_path} seems empty or wrong dimensions.")
        return

    if slice_index >= total_slices or slice_index < 0:
        print(f"slice_index {slice_index} out of range for file {raw_path}. Max slice: {total_slices-1}")
        return

    volume = raw_data.reshape((total_slices, height, width))
    slice_2d = volume[slice_index]

    norm_img = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resized_img = cv2.resize(norm_img, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(raw_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.jpg")  # Use original filename

    cv2.imwrite(output_path, resized_img)
    print(f"Saved resized slice {slice_index} of {raw_path} as {output_path}")

if __name__ == "__main__":
    input_folder = input("Enter the input folder path containing raw files: ").strip()
    output_dir = input("Enter the output folder path to save JPGs: ").strip()

    try:
        width = int(input("Enter width of each slice (e.g. 512): ").strip())
        height = int(input("Enter height of each slice (e.g. 512): ").strip())
    except ValueError:
        print("Invalid number input. Please enter integers for width and height.")
        exit(1)

    raw_files = glob(os.path.join(input_folder, '*.raw'))
    if not raw_files:
        print(f"No raw files found in {input_folder}")
        exit(1)

    for raw_file in raw_files:
        raw3d_to_jpg_resized(raw_file, output_dir, width, height, slice_index=0)
