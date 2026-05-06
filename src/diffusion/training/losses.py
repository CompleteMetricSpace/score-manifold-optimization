#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions for diffusion model training.
"""

import torch
from typing import Any, Callable, Dict, Tuple, Union


def stratified_uniform(t_min: float, t_max: float, batch_size: int, device: str = 'cpu') -> torch.Tensor:
    """
    Stratified uniform sampling in [t_min, t_max].

    Divides the interval into batch_size equal bins and samples
    uniformly within each bin. This ensures better coverage of the
    time domain compared to naive uniform sampling.

    Args:
        t_min: Minimum time
        t_max: Maximum time
        batch_size: Number of samples
        device: Device to create tensor on

    Returns:
        t: (batch_size,) tensor of time values
    """
    # Create bin edges
    bins = torch.linspace(t_min, t_max, batch_size + 1, device=device)

    # Sample uniformly within each bin
    u = torch.rand(batch_size, device=device)
    t = bins[:-1] + u * (bins[1:] - bins[:-1])

    return t


def denoising_score_matching_loss(
    score_model: Callable,
    diffusion_process,
    x0: torch.Tensor,
    t: torch.Tensor,
    weight_by_var: bool = True,
    reduction: str = 'mean',
    return_debug: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Denoising score matching loss (conditional score matching).

    Computes:
        L = E_{t,x_0,ε} [ w(t) ||s_θ(t, x_t) - ∇log p_t(x_t|x_0)||² ]

    where:
    - x_t ~ p_t(·|x_0) is sampled from the conditional distribution
    - ∇log p_t(x_t|x_0) = -ε / sqrt(var) is the true conditional score
    - w(t) = sqrt(var) if weight_by_var=True, else 1.0

    Args:
        score_model: Score model s_θ(t, x)
        diffusion_process: DiffusionProcess instance
        x0: (batch, *dims) clean data samples
        t: (batch,) time values
        weight_by_var: Whether to weight by variance (importance sampling)
        reduction: 'mean', 'sum', or 'none'
        return_debug: Whether to return a diagnostics dict together with loss

    Returns:
        loss: Scalar loss value (if reduction != 'none')
    """
    batch_size = x0.shape[0]
    debug: Dict[str, Any] = {
        "ot_enabled": False,
        "ot_applied": False,
    }

    # Get conditional distribution p_t(·|x_0)
    mean, var = diffusion_process.conditional_mean_var(t, x0)

    # Sample noise
    noise = torch.randn_like(x0)

    # Sample x_t ~ p_t(·|x_0)
    # x_t = mean + sqrt(var) * noise
    var_view = var.view(batch_size, *[1] * len(diffusion_process.data_dims))
    x_t = mean + torch.sqrt(var_view) * noise

    score_pred = score_model(t, x_t)
    score_true = -(x_t - mean) / var_view

    # Compute loss
    diff = score_pred - score_true

    # Weight by variance (importance sampling)
    if weight_by_var:
        # Multiply by sqrt(var) to get ||sqrt(var) * (score_pred - score_true)||²
        # This equals ||ε_pred - ε||² where ε_pred = -sqrt(var) * score_pred
        weight = torch.sqrt(var_view)
        diff = diff * weight

    # Square and reduce
    loss = diff.pow(2)

    if reduction == 'mean':
        loss_out = loss.mean()
    elif reduction == 'sum':
        loss_out = loss.sum()
    elif reduction == 'none':
        loss_out = loss.mean(dim=list(range(1, loss.ndim)))  # Average over spatial dims
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    if return_debug:
        return loss_out, debug
    return loss_out


def score_matching_loss_simple(
    score_model: Callable,
    diffusion_process,
    x0: torch.Tensor,
    t: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Simplified denoising score matching loss (epsilon prediction).

    Predicts noise directly: ε_θ(t, x_t) instead of score.
    Equivalent to score matching but with simpler formulation:
        L = E_{t,x_0,ε} [ ||ε_θ(t, x_t) - ε||² ]

    Args:
        score_model: Model that predicts noise (not score)
        diffusion_process: DiffusionProcess instance
        x0: (batch, *dims) clean data samples
        t: (batch,) time values
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Scalar loss value (if reduction != 'none')
    """
    batch_size = x0.shape[0]

    # Get conditional distribution p_t(·|x_0)
    mean, var = diffusion_process.conditional_mean_var(t, x0)

    # Sample noise
    noise = torch.randn_like(x0)

    # Sample x_t ~ p_t(·|x_0)
    var_view = var.view(batch_size, *[1] * len(diffusion_process.data_dims))
    x_t = mean + torch.sqrt(var_view) * noise

    # Predict noise (model outputs noise, not score)
    noise_pred = score_model(t, x_t)

    # MSE loss
    loss = (noise_pred - noise).pow(2)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss.mean(dim=list(range(1, loss.ndim)))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
