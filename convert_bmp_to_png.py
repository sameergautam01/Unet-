# convert_bmp_to_png_prompt.py
import os
import cv2
from glob import glob

def convert_bmp_to_png(input_dir, output_dir):
    bmp_files = glob(os.path.join(input_dir, '**', '*.bmp'), recursive=True)

    if not bmp_files:
        print("No BMP files found.")
        return

    for bmp_path in bmp_files:
        rel_path = os.path.relpath(bmp_path, input_dir)
        png_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.png')

        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        img = cv2.imread(bmp_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Failed to read {bmp_path}")
            continue

        cv2.imwrite(png_path, img)
        print(f"Converted: {bmp_path} -> {png_path}")

if __name__ == '__main__':
    print("=== BMP to PNG Converter ===")
    input_dir = input("Enter the path to the input folder (containing BMP files): ").strip('"').strip("'")
    output_dir = input("Enter the path to the output folder (where PNGs will be saved): ").strip('"').strip("'")

    if not os.path.isdir(input_dir):
        print(f"Error: Input path '{input_dir}' does not exist or is not a directory.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        convert_bmp_to_png(input_dir, output_dir)
