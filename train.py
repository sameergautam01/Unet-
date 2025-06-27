import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import psutil
import GPUtil
from sklearn.metrics import f1_score
from albumentations import (
    RandomRotate90,
    HorizontalFlip,
    HueSaturationValue,
    RandomBrightnessContrast,
    Resize,
    Normalize,
    Compose,
    OneOf
)
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('--arch', '-a', default='NestedUNet', choices=ARCH_NAMES)
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=1024, type=int)
    parser.add_argument('--input_h', default=512, type=int)
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)
    parser.add_argument('--dataset', default='cracks')
    parser.add_argument('--img_ext', default='.jpg')
    parser.add_argument('--mask_ext', default='.png')
    parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser.parse_args()


def compute_f1(outputs, targets, threshold=0.5):
    probs = torch.sigmoid(outputs)  # logits to probabilities
    preds_bin = (probs > threshold).int()
    targets_bin = targets.int()  # Convert targets to int (0 or 1)
    return f1_score(targets_bin.view(-1).cpu().numpy(), preds_bin.view(-1).cpu().numpy(), zero_division=1)

def get_system_stats():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    gpus = GPUtil.getGPUs()
    gpu_stats = [{"id": gpu.id, "name": gpu.name, "load": gpu.load * 100, "memoryUtil": gpu.memoryUtil * 100} for gpu in gpus]
    return cpu_percent, ram_percent, gpu_stats


def train_one_epoch(config, train_loader, model, cri terion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'f1': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        f1 = compute_f1(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['f1'].update(f1, input.size(0))
        pbar.set_postfix({k: v.avg for k, v in avg_meters.items()})
        pbar.update(1)
    pbar.close()
    return {k: v.avg for k, v in avg_meters.items()}


def validate_one_epoch(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'f1': AverageMeter()}
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            f1 = compute_f1(output, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['f1'].update(f1, input.size(0))
            pbar.set_postfix({k: v.avg for k, v in avg_meters.items()})
            pbar.update(1)
        pbar.close()
    return {k: v.avg for k, v in avg_meters.items()}


def main():
    config = vars(parse_args())
    config['name'] = config['name'] or f"{config['dataset']}_{config['arch']}_woDS"
    os.makedirs(f"models/{config['name']}", exist_ok=True)
    with open(f"models/{config['name']}/config.yml", 'w') as f:
        yaml.dump(config, f)

    # Criterion
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # Model
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision']).cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])

    # Scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    else:
        scheduler = None

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))]
    train_ids, val_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(),
        OneOf([
            HueSaturationValue(),
            RandomBrightnessContrast(),
        ], p=1),
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    train_dataset = Dataset(train_ids,
                            f"inputs/{config['dataset']}/images",
                            f"inputs/{config['dataset']}/masks",
                            config['img_ext'], config['mask_ext'], config['num_classes'],
                            train_transform)
    val_dataset = Dataset(val_ids,
                          f"inputs/{config['dataset']}/images",
                          f"inputs/{config['dataset']}/masks",
                          config['img_ext'], config['mask_ext'], config['num_classes'],
                          val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                               num_workers=config['num_workers'], drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                             num_workers=config['num_workers'])

    log = OrderedDict([(k, []) for k in ['epoch', 'lr', 'loss', 'iou', 'f1', 'val_loss', 'val_iou', 'val_f1']])
    best_iou = 0

    for epoch in range(config['epochs']):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        train_log = train_one_epoch(config, train_loader, model, criterion, optimizer)
        val_log = validate_one_epoch(config, val_loader, model, criterion)

        # Scheduler step
        if scheduler:
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])
            else:
                scheduler.step()

        cpu_percent, ram_percent, gpu_stats = get_system_stats()

        for k in ['loss', 'iou', 'f1']:
            log[k].append(train_log[k])
            log[f"val_{k}"].append(val_log[k])
        log['epoch'].append(epoch + 1)
        log['lr'].append(optimizer.param_groups[0]['lr'])

        # Save log CSV
        pd.DataFrame(log).to_csv(f"models/{config['name']}/log.csv", index=False)

        # Append to epoch log text file
        with open(f"models/{config['name']}/epoch_log.txt", "a") as f:
            f.write(f"Epoch {epoch + 1}/{config['epochs']}\n")
            f.write(f"Train Loss: {train_log['loss']:.4f}, IoU: {train_log['iou']:.4f}, F1: {train_log['f1']:.4f}\n")
            f.write(f"Val   Loss: {val_log['loss']:.4f}, IoU: {val_log['iou']:.4f}, F1: {val_log['f1']:.4f}\n")
            f.write(f"CPU Usage: {cpu_percent:.1f}%, RAM Usage: {ram_percent:.1f}%\n")
            for g in gpu_stats:
                f.write(f"GPU {g['id']} ({g['name']}): Load {g['load']:.1f}%, Mem {g['memoryUtil']:.1f}%\n")
            f.write("-" * 40 + "\n")

        # Save best model
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f"models/{config['name']}/model.pth")
            best_iou = val_log['iou']
            print("=> saved best model")

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
