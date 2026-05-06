"""
Diffusion and optimizer factory utilities.

This module provides factory functions to create diffusion processes and optimizers
from plain Python configs (dict-like objects).
"""

from collections.abc import Mapping
from typing import Any
import torch

from ..core.diffusion import VPDiffusion, VEDiffusion


def _get_value(config: Any, key: str, default=None):
    """
    Generic nested getter for dict-like configs.

    Args:
        config: Configuration object
        key: Dot-notation key (e.g., 'model.hidden_dim')
        default: Default value if key not found

    Returns:
        Value at key or default
    """
    value: Any = config
    for part in key.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return default
            value = value[part]
            continue
        if hasattr(value, part):
            value = getattr(value, part)
            continue
        return default
    return value


def create_diffusion(space, config):
    """
    Create diffusion process based on configuration.

    Args:
        space: Mathematical space
        config: Configuration object with diffusion.type, diffusion.beta_min, etc.

    Returns:
        Diffusion process (VPDiffusion or VEDiffusion)
    """
    diffusion_type = str(_get_value(config, "diffusion.type")).lower()

    if diffusion_type == 'vp':
        return VPDiffusion(
            _get_value(config, 'diffusion.beta_min'),
            _get_value(config, 'diffusion.beta_max'),
            _get_value(config, 'diffusion.T'),
            space
        )
    elif diffusion_type == 've':
        return VEDiffusion(
            _get_value(config, 'diffusion.sigma_min'),
            _get_value(config, 'diffusion.sigma_max'),
            _get_value(config, 'diffusion.T'),
            space
        )
    else:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}")


def create_optimizer(model, config):
    """
    Create optimizer from configuration.

    Args:
        model: PyTorch model
        config: Configuration object with optimizer.type, optimizer.lr, etc.

    Returns:
        PyTorch optimizer
    """
    optimizer_type = str(_get_value(config, "optimizer.type")).lower()

    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=float(_get_value(config, "optimizer.lr")),
            betas=tuple(_get_value(config, "optimizer.betas", [0.9, 0.999])),
            weight_decay=float(_get_value(config, "optimizer.weight_decay", 0.0)),
        )
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(_get_value(config, "optimizer.lr")),
            betas=tuple(_get_value(config, "optimizer.betas", [0.9, 0.999])),
            weight_decay=float(_get_value(config, "optimizer.weight_decay", 0.0)),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
