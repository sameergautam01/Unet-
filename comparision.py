import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Ask user to input paths
# -------------------------
def ask_path(prompt):
    path = input(f"{prompt}: ").strip('"').strip("'")
    if not os.path.isdir(path):
        print(f"❌ Error: '{path}' is not a valid directory.")
        exit()
    return path

# -------------------------
# Folder Selection via input
# -------------------------
image_dir = ask_path("Enter path to folder with original JPG images")
gt_mask_dir = ask_path("Enter path to folder with ground truth PNG masks")
pred_mask_dir = ask_path("Enter path to folder with predicted masks (PNG or JPG)")
output_dir = ask_path("Enter path to folder to save output dashboards")
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Get all image files (JPG only)
# -------------------------
images = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

if not images:
    print("❌ No .jpg images found in the selected folder.")
    exit()

results = []

# -------------------------
# Process each image
# -------------------------
for image_name in images:
    image_base = os.path.splitext(image_name)[0]

    image_path = os.path.join(image_dir, f"{image_base}.jpg")

    # Try predicted mask as PNG, fallback to JPG
    pred_path_png = os.path.join(pred_mask_dir, f"{image_base}.png")
    pred_path_jpg = os.path.join(pred_mask_dir, f"{image_base}.jpg")
    if os.path.exists(pred_path_png):
        pred_path = pred_path_png
    elif os.path.exists(pred_path_jpg):
        pred_path = pred_path_jpg
    else:
        print(f"⚠️ Skipping {image_name} (missing predicted mask)")
        continue

    gt_path = os.path.join(gt_mask_dir, f"{image_base}.png")
    if not os.path.exists(gt_path):
        print(f"⚠️ Skipping {image_name} (missing ground truth mask)")
        continue

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if image is None or pred_mask is None or gt_mask is None:
        print(f"⚠️ Skipping {image_name} (could not load one or more files)")
        continue

    # -------------------------
    # Convert masks to binary: cracks = 1 (black pixels), background = 0 (white pixels)
    # -------------------------
    gt_mask_bin = (gt_mask == 0).astype(np.uint8)
    pred_mask_bin = (pred_mask == 0).astype(np.uint8)

    # -------------------------
    # Metric Calculation (crack = 1)
    # -------------------------
    tp = np.logical_and(gt_mask_bin == 1, pred_mask_bin == 1).sum()
    fp = np.logical_and(gt_mask_bin == 0, pred_mask_bin == 1).sum()
    fn = np.logical_and(gt_mask_bin == 1, pred_mask_bin == 0).sum()
    intersection = np.logical_and(gt_mask_bin == 1, pred_mask_bin == 1).sum()
    union = np.logical_or(gt_mask_bin == 1, pred_mask_bin == 1).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = intersection / union if union > 0 else 0

    results.append({
        'Image': image_name,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'IoU': iou,
        'TP': tp,
        'FP': fp,
        'FN': fn
    })

    # -------------------------
    # Visualization
    # -------------------------
    diff_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    diff_map[np.logical_and(gt_mask_bin == 1, pred_mask_bin == 1)] = [0, 255, 0]   # TP (green)
    diff_map[np.logical_and(gt_mask_bin == 0, pred_mask_bin == 1)] = [255, 0, 0]   # FP (red)
    diff_map[np.logical_and(gt_mask_bin == 1, pred_mask_bin == 0)] = [0, 0, 255]   # FN (blue)

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay[pred_mask_bin == 1] = [0, 255, 0]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"{image_name}\n"
        f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f} | IoU: {iou:.3f}",
        fontsize=14
    )

    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(gt_mask_bin, cmap='gray')
    axs[0, 1].set_title("Ground Truth (Black = Crack)")
    axs[0, 1].axis('off')

    axs[0, 2].imshow(diff_map)
    axs[0, 2].set_title("Difference Map\n(G=TP, R=FP, B=FN)")
    axs[0, 2].axis('off')

    axs[1, 0].imshow(overlay)
    axs[1, 0].set_title("Overlay (Green = Predicted Crack)")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(pred_mask_bin, cmap='gray')
    axs[1, 1].set_title("Predicted Mask (Black = Crack)")
    axs[1, 1].axis('off')

    gauge_ax = axs[1, 2]
    gauge_ax.axis('equal')
    gauge_ax.axis('off')
    theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    gauge_ax.plot(x, y, color='black')

    gauge_ax.fill_between(x, 0, y, where=(x <= 0), color='lightcoral', alpha=0.5)
    gauge_ax.fill_between(x, 0, y, where=(x > 0) & (x <= 0.5), color='khaki', alpha=0.5)
    gauge_ax.fill_between(x, 0, y, where=(x > 0.5), color='lightgreen', alpha=0.5)

    p_angle = (precision * np.pi) - (np.pi / 2)
    r_angle = (recall * np.pi) - (np.pi / 2)
    f_angle = (f1 * np.pi) - (np.pi / 2)

    gauge_ax.arrow(0, 0, np.cos(p_angle) * 0.75, np.sin(p_angle) * 0.75,
                   head_width=0.05, head_length=0.1, fc='orange', ec='orange')
    gauge_ax.arrow(0, 0, np.cos(r_angle) * 0.6, np.sin(r_angle) * 0.6,
                   head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    gauge_ax.arrow(0, 0, np.cos(f_angle) * 0.5, np.sin(f_angle) * 0.5,
                   head_width=0.05, head_length=0.1, fc='purple', ec='purple')

    gauge_ax.set_title("Gauge: 🟧 Precision, 🔵 Recall, 🟪 F1-score", fontsize=11)
    gauge_ax.text(0, -0.3, f'Precision: {precision:.2f}', ha='center', fontsize=10, color='orange')
    gauge_ax.text(0, -0.45, f'Recall: {recall:.2f}', ha='center', fontsize=10, color='blue')
    gauge_ax.text(0, -0.6, f'F1-score: {f1:.2f}', ha='center', fontsize=10, color='purple')

    save_path = os.path.join(output_dir, f"{image_base}_dashboard.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")

# -------------------------
# Save metrics to Excel
# -------------------------
df = pd.DataFrame(results)
excel_path = os.path.join(output_dir, "crack_segmentation_summary.xlsx")
df.to_excel(excel_path, index=False)
print(f"\n📊 Excel summary saved to: {excel_path}")
print("\n🎉 All images processed successfully!")
