import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _subset_dataset(dataset, size: int, seed: int) -> Subset:
    if size is None or size <= 0 or size >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:size].tolist()
    return Subset(dataset, indices)


def _resolve_root(root: str, project_root: str) -> str:
    if root is None:
        return project_root
    if not root or root == ".":
        return project_root
    if os.path.isabs(root):
        return root
    return os.path.join(project_root, root)


def build_datasets(cfg: Dict) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    data_cfg = cfg.get("data", {})
    project_root = cfg.get("project_root", ".")
    root = _resolve_root(data_cfg.get("root", "data"), project_root)
    use_aug = bool(data_cfg.get("use_augmentation", True))

    if use_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    seed = int(cfg.get("seed", 0))
    train_subset = data_cfg.get("train_subset")
    test_subset = data_cfg.get("test_subset")

    train_set = _subset_dataset(train_set, train_subset, seed)
    test_set = _subset_dataset(test_set, test_subset, seed + 1)

    return train_set, test_set


def build_loaders(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    cfg: Dict,
) -> Tuple[DataLoader, DataLoader]:
    batch_size = int(cfg.get("batch_size", 128))
    test_batch_size = int(cfg.get("test_batch_size", 256))
    num_workers = int(cfg.get("num_workers", 2))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, test_loader
