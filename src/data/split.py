from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset


def _subset(dataset, size: int, seed: int) -> Subset:
    if size is None or size <= 0 or size >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:size].tolist()
    return Subset(dataset, indices)


def build_mia_loaders(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    cfg: Dict,
) -> Tuple[DataLoader, DataLoader]:
    mia_cfg = cfg.get("mia", {})
    batch_size = int(mia_cfg.get("batch_size", cfg.get("test_batch_size", 256)))
    num_workers = int(cfg.get("num_workers", 2))
    seed = int(cfg.get("seed", 0))

    member_size = mia_cfg.get("member_size")
    nonmember_size = mia_cfg.get("nonmember_size")
    use_test = bool(mia_cfg.get("use_test_as_nonmember", True))

    members = _subset(train_set, member_size, seed + 10)
    if use_test:
        nonmembers = _subset(test_set, nonmember_size, seed + 20)
    else:
        # Fallback: use test set if no explicit non-member pool is configured.
        nonmembers = _subset(test_set, nonmember_size, seed + 20)

    member_loader = DataLoader(
        members,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    nonmember_loader = DataLoader(
        nonmembers,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return member_loader, nonmember_loader
