"""
Training infrastructure for diffusion models.

This module provides:
- DiffusionTrainer: Unified trainer for (DiffusionProcess, ScoreModel) pairs
- TrainingOptions: Configuration dataclass for training hyperparameters
- Loss functions: Denoising score matching and variants
- Factory helpers: create_diffusion and create_optimizer
"""

from .options import TrainingOptions
from .factory_utils import create_diffusion, create_optimizer
from .losses import denoising_score_matching_loss, stratified_uniform, score_matching_loss_simple
from .trainer import DiffusionTrainer

__all__ = [
    "DiffusionTrainer",
    "TrainingOptions",
    "create_diffusion",
    "create_optimizer",
    "denoising_score_matching_loss",
    "score_matching_loss_simple",
    "stratified_uniform",
]
