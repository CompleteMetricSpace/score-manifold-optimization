#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset saving utilities.
"""

import torch
from pathlib import Path
from typing import Union


def save_dataset(dataset_bundle, path: Union[str, Path]):
    """
    Save dataset bundle to file.

    Saves in enhanced PyTorch format with metadata:
    {
        "train_data": tensor,
        "test_data": tensor,
        "val_data": tensor,
        "metadata": {...}
    }

    Args:
        dataset_bundle: DatasetBundle instance
        path: Path to save file (absolute or relative)

    Example:
        >>> save_dataset(bundle, "data/stiefel_n10p10.pt")
    """
    from .loader import DatasetBundle

    if not isinstance(dataset_bundle, DatasetBundle):
        raise TypeError(f"Expected DatasetBundle, got {type(dataset_bundle)}")

    # Ensure path is Path object
    path = Path(path)

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build save dictionary
    save_dict = {
        "train_data": dataset_bundle.train_data,
        "metadata": dataset_bundle.metadata,
    }

    # Add optional components
    if dataset_bundle.test_data is not None:
        save_dict["test_data"] = dataset_bundle.test_data

    if dataset_bundle.val_data is not None:
        save_dict["val_data"] = dataset_bundle.val_data

    # Save to file
    torch.save(save_dict, path)

    print(f"Dataset saved to {path}")
    print(f"  Train samples: {len(dataset_bundle.train_data)}")
    if dataset_bundle.test_data is not None:
        print(f"  Test samples: {len(dataset_bundle.test_data)}")
    if dataset_bundle.val_data is not None:
        print(f"  Val samples: {len(dataset_bundle.val_data)}")
