#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training configuration options.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingOptions:
    """
    Configuration for diffusion model training.

    Time Sampling:
        t_start: Minimum time value for training (avoid t=0 singularities)
        t_stop: Maximum time value for training
        T_stop: Alternative name for t_stop (backward compatibility)
        stratified: Whether to use stratified time sampling

    Loss Configuration:
        weight_by_var: Whether to weight loss by variance (importance sampling)
        loss_type: Type of loss ('dsm' for denoising score matching)

    Optimization:
        clip_grad: Whether to clip gradients
        clip_value: Maximum gradient norm

    Distributed Training:
        rank: Process rank for distributed training
        local_rank: Local rank on current node
        world_size: Total number of processes
        ddp: Whether using DistributedDataParallel

    Logging:
        log_interval: Epochs between logging (deprecated, not used)
        save_interval: Epochs between checkpoints
        eval_interval: Epochs between sample-metric evaluations
    """

    # Time sampling
    t_start: float = 1e-5
    t_stop: float = 1.0
    T_stop: Optional[float] = None  # Alias for t_stop (backward compatibility)
    stratified: bool = True

    # Loss configuration
    weight_by_var: bool = True
    loss_type: str = "dsm"  # Denoising score matching

    # Optimization
    clip_grad: bool = True
    clip_value: float = 1.0

    # Distributed training
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    ddp: bool = False

    # Logging
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500

    # Data loading
    batch_size: int = 32
    num_workers: int = 0

    # Evaluation
    num_eval_samples: int = 100
    eval_batch_size: int = 32

    # Sample-metric evaluation via optional trainer log_fn
    log_sample_metrics: bool = True   # Enable sample metric logging
    eval_log_fn_samples: int = 50     # Number of generated samples for log_fn
    eval_log_fn_steps: int = 100      # Sampling steps for log_fn evaluation

    def __post_init__(self):
        """Handle backward compatibility."""
        if self.T_stop is not None:
            self.t_stop = self.T_stop
