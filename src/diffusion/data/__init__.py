"""
Data generation and loading module for diffusion models.

This module provides a unified interface for:
- Generating datasets from constraints (manifolds, systems, regions)
- Saving datasets with comprehensive metadata
- Loading datasets with automatic constraint reconstruction

Main API:
    >>> from diffusion.data import generate_dataset, load_dataset
    >>>
    >>> # Generate data
    >>> dataset = generate_dataset(
    ...     generator_type="manifold",
    ...     constraint=Stiefel(n=10, p=10),
    ...     n_train=5000,
    ...     n_test=1000,
    ...     save_path="stiefel_data.pt"
    ... )
    >>>
    >>> # Load data
    >>> dataset = load_dataset("stiefel_data.pt")
    >>> space = dataset.get_space()
    >>> dataloader = dataset.get_dataloader(split="train", batch_size=32)
"""

from .generator import (
    DataGenerator,
    ManifoldDataGenerator,
    DynamicsDataGenerator,
    RegionDataGenerator,
    generate_dataset,
)
from .loader import load_dataset, DatasetBundle
from .saver import save_dataset

__all__ = [
    # Generation
    "DataGenerator",
    "ManifoldDataGenerator",
    "DynamicsDataGenerator",
    "RegionDataGenerator",
    "generate_dataset",

    # Loading
    "load_dataset",
    "DatasetBundle",

    # Saving
    "save_dataset",
]
