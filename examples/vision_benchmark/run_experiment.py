"""Vision Architecture Comparison Experiment using raxpy."""

import argparse
import csv
import json
import os
import ssl
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing_extensions import Annotated

import raxpy

try:
    from models import build_resnet34, build_vit_small, count_parameters
except ImportError:
    from experiment.models import build_resnet34, build_vit_small, count_parameters


@dataclass
class ResNetConfig:
    stem_kernel: Annotated[int, raxpy.Integer(lb=3, ub=7)]
    dropout_rate: Annotated[float, raxpy.Float(lb=0.0, ub=0.3)]


@dataclass
class ViTConfig:
    patch_size: Annotated[int, raxpy.Integer(lb=4, ub=7)]
    dropout_rate: Annotated[float, raxpy.Float(lb=0.0, ub=0.2)]


@dataclass
class AugmentationConfig:
    rotation_deg: Annotated[float, raxpy.Float(lb=5.0, ub=15.0)]
    crop_padding: Annotated[int, raxpy.Integer(lb=1, ub=2)]


_TRAIN_DATASET = None
_VAL_DATASET = None
_RUN_CONFIG = {}


def _get_data_loaders(
    batch_size: int,
    augmentation: Optional[AugmentationConfig],
) -> Tuple[DataLoader, DataLoader]:
    global _TRAIN_DATASET, _VAL_DATASET

    if augmentation is not None:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=augmentation.rotation_deg),
            transforms.RandomCrop(28, padding=augmentation.crop_padding),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    _TRAIN_DATASET.dataset.transform = train_transform
    _VAL_DATASET.dataset.transform = val_transform

    train_loader = DataLoader(_TRAIN_DATASET, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(_VAL_DATASET, batch_size=128, shuffle=False)
    return train_loader, val_loader


def evaluate_trial(
    model_arch: Union[ResNetConfig, ViTConfig],
    lr: Annotated[float, raxpy.Float(lb=1e-4, ub=2e-3)],
    weight_decay: Annotated[float, raxpy.Float(lb=1e-5, ub=2e-3)],
    batch_size: Annotated[int, raxpy.Integer(lb=32, ub=128)],
    augmentation: Optional[AugmentationConfig],
) -> Dict:
    device = torch.device(_RUN_CONFIG.get("device", "cpu"))
    epochs = _RUN_CONFIG.get("epochs", 1)

    if isinstance(model_arch, ResNetConfig):
        arch_type = "ResNet-34"
        model = build_resnet34(stem_kernel=int(model_arch.stem_kernel), dropout_rate=float(model_arch.dropout_rate))
        arch_meta = {"stem_kernel": int(model_arch.stem_kernel), "patch_size": None, "dropout_rate": float(model_arch.dropout_rate)}
    elif isinstance(model_arch, ViTConfig):
        arch_type = "ViT-Small"
        model = build_vit_small(patch_size=int(model_arch.patch_size), dropout_rate=float(model_arch.dropout_rate))
        arch_meta = {"stem_kernel": None, "patch_size": int(model_arch.patch_size), "dropout_rate": float(model_arch.dropout_rate)}
    else:
        raise TypeError(f"Unknown architecture branch: {type(model_arch)}")

    model.to(device)
    param_count = count_parameters(model)
    train_loader, val_loader = _get_data_loaders(batch_size=int(batch_size), augmentation=augmentation)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    t_start = time.time()
    model.train()
    running_loss, total_steps = 0.0, 0

    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_steps += 1

    train_duration = time.time() - t_start
    avg_train_loss = running_loss / max(1, total_steps)

    model.eval()
    correct, total, val_loss_sum, val_steps = 0, 0, 0.0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            val_loss_sum += val_loss.item()
            val_steps += 1
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_accuracy = (correct / total * 100.0) if total > 0 else 0.0
    avg_val_loss = val_loss_sum / max(1, val_steps)

    aug_info = {
        "has_augmentation": augmentation is not None,
        "rotation_deg": float(augmentation.rotation_deg) if augmentation else None,
        "crop_padding": int(augmentation.crop_padding) if augmentation else None,
    }

    result = {
        "architecture": arch_type,
        "parameters": param_count,
        "learning_rate": float(lr),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        **arch_meta,
        **aug_info,
        "train_loss": round(avg_train_loss, 4),
        "val_loss": round(avg_val_loss, 4),
        "val_accuracy": round(val_accuracy, 2),
        "train_duration_sec": round(train_duration, 2),
    }

    print(f"[{arch_type}] Params: {param_count:,} | LR: {lr:.5f} | BS: {int(batch_size)} | Aug: {aug_info['has_augmentation']} | Val Acc: {val_accuracy:.2f}% | Time: {train_duration:.2f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run raxpy vision architecture exploration experiment.")
    parser.add_argument("--n_points", type=int, default=20, help="Number of design points (default: 20)")
    parser.add_argument("--train_subset", type=int, default=3000, help="Number of MNIST training samples (default: 3000)")
    parser.add_argument("--val_subset", type=int, default=500, help="Number of MNIST validation samples (default: 500)")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per trial (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset directory (default: ./data)")
    parser.add_argument("--output_dir", type=str, default="./experiment", help="Output directory (default: ./experiment)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)

    os.makedirs(args.data_dir, exist_ok=True)
    raw_train = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    raw_test = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transforms.ToTensor())

    global _TRAIN_DATASET, _VAL_DATASET, _RUN_CONFIG
    _TRAIN_DATASET = Subset(raw_train, list(range(min(args.train_subset, len(raw_train)))))
    _VAL_DATASET = Subset(raw_test, list(range(min(args.val_subset, len(raw_test)))))
    _RUN_CONFIG = {"epochs": args.epochs, "device": "cpu"}

    doe, arg_sets, results = raxpy.perform_experiment(
        evaluate_trial,
        n_points=args.n_points,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "results.json")
    csv_path = os.path.join(args.output_dir, "results.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"Results saved to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()

# this code was debugged and optimized with assistance from Gemini 3.8 Flash (medium) within Google Antigravity, September 2026