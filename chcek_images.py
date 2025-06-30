import os
import cv2

img_dir = r"C:\test\Image Library\Image Library\Standard Testing Library-Asphalt\3D_jpg"
bad_images = []

for fname in os.listdir(img_dir):
    if fname.lower().endswith(".jpg"):
        fpath = os.path.join(img_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"[BAD] Cannot read image: {fpath}")
            bad_images.append(fpath)

print("\nSummary:")
print(f"Total files checked: {len(os.listdir(img_dir))}")
print(f"Bad images found: {len(bad_images)}")
