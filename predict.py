# predict.py (CPU-only version for Intel Core Ultra 7)

import os
import argparse
import yaml
import cv2
import numpy as np
import torch
from albumentations import Compose, Resize, Normalize
from tqdm import tqdm

import archs
from dataset import Dataset

def load_config(model_name):
    with open(f'models/{model_name}/config.yml') as f:
        return yaml.safe_load(f)

def get_transform(h, w):
    return Compose([
        Resize(h, w),
        Normalize()
    ])

def load_model(config):
    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision']
    )
    model.load_state_dict(torch.load(f"models/{config['name']}/model.pth", map_location='cpu'))
    model.eval()
    return model  # Remains on CPU

def predict(model, image, transform):
    augmented = transform(image=image)
    img = augmented['image'].astype('float32') / 255
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W) - stays on CPU

    with torch.no_grad():
        output = model(img)
        if isinstance(output, list):  # Deep supervision
            output = output[-1]
        output = torch.sigmoid(output)
        output = output.squeeze(0).numpy()  # (num_classes, H, W)
    return output

def save_masks(preds, out_dir, img_id, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    preds_bin = (preds > threshold).astype(np.uint8) * 255
    for cls_idx in range(preds.shape[0]):
        out_path = os.path.join(out_dir, f"{img_id}_class{cls_idx}.png")
        cv2.imwrite(out_path, preds_bin[cls_idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, help='Name of the trained model (folder in models/)')
    parser.add_argument('--input_dir', required=True, help='Directory of input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save predicted masks')
    parser.add_argument('--img_ext', default='.jpg')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.model_name)
    model = load_model(config)
    transform = get_transform(config['input_h'], config['input_w'])

    img_paths = [p for p in os.listdir(args.input_dir) if p.endswith(args.img_ext)]

    for img_name in tqdm(img_paths):
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(args.input_dir, img_name)
        image = cv2.imread(img_path)
        preds = predict(model, image, transform)
        save_masks(preds, args.output_dir, img_id, threshold=args.threshold)

if __name__ == '__main__':
    main()
