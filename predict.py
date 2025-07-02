# predict.py (CUDA-enabled version with system resource logging)

import os
import argparse
import yaml
import cv2
import numpy as np
import torch
import time
import psutil
from albumentations import Compose, Resize, Normalize
from tqdm import tqdm
import GPUtil

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


def log_system_usage(stage=''):
    print(f"\n[{stage}] System Usage:")
    print(f"  CPU Usage: {psutil.cpu_percent()}%")
    print(f"  RAM Usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"  GPU {gpu.id} - {gpu.name}: {gpu.load * 100:.1f}% load, {gpu.memoryUsed}MB used of {gpu.memoryTotal}MB")


def load_model(config, device):
    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision']
    )
    model.load_state_dict(torch.load(f"models/{config['name']}/model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict(model, image, transform, device):
    augmented = transform(image=image)
    img = augmented['image'].astype('float32') / 255
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        if isinstance(output, list):
            output = output[-1]
        output = torch.sigmoid(output)
        output = output.squeeze(0).cpu().numpy()
    return output


def save_mask(pred, out_dir, original_filename, threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    pred_bin = (pred > threshold).astype(np.uint8) * 255

    if pred_bin.ndim == 3:
        pred_bin = pred_bin[0]

    out_filename = os.path.splitext(original_filename)[0] + '.png'
    out_path = os.path.join(out_dir, out_filename)
    cv2.imwrite(out_path, pred_bin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, help='Name of the trained model (folder in models/)')
    parser.add_argument('--input_dir', required=True, help='Directory of input images')
    parser.add_argument('--output_dir', required=True, help='Directory to save predicted masks')
    parser.add_argument('--img_ext', default='.jpg')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    log_system_usage(stage="Startup")

    config = load_config(args.model_name)
    model = load_model(config, device)
    transform = get_transform(config['input_h'], config['input_w'])

    img_paths = [p for p in os.listdir(args.input_dir) if p.endswith(args.img_ext)]

    for img_name in tqdm(img_paths, desc="Predicting"):
        img_path = os.path.join(args.input_dir, img_name)
        image = cv2.imread(img_path)

        start_time = time.time()
        preds = predict(model, image, transform, device)
        elapsed = time.time() - start_time

        save_mask(preds, args.output_dir, img_name, threshold=args.threshold)

        print(f"\n[Prediction: {img_name}] Time: {elapsed:.3f}s")
        log_system_usage(stage="After Prediction")

if __name__ == '__main__':
    main()
