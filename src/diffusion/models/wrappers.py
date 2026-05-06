#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score model wrapper classes for output transformations.

This module provides wrapper classes that compose with any ScoreModel
to apply transformations to the score output (e.g., scaling by diffusion coefficient,
adding residual connections, log-time transformations, etc.).
"""

import torch
import torch.nn as nn
from typing import Callable
from .base import ScoreModel


class DiffusionScalingWrapper(ScoreModel):
    """
    Wrapper that scales score output by diffusion coefficient.

    This implements the common pattern where the model predicts noise ε
    instead of score ∇log p_t(x), related by: score = -ε/σ(t)

    Args:
        base_model: The score model to wrap
        sigma_fn: Function (t) -> value, interpretation depends on mode:
            - mode="sigma": sigma_fn should return σ(t), output scaled by 1/σ(t)
            - mode="sqrt_var": sigma_fn should return σ²(t), output scaled by 1/√σ²(t) = 1/σ(t)
            - mode="var": sigma_fn should return σ²(t), output scaled by 1/σ²(t)
        clip_eps: Minimum value to clamp (prevents division by zero)
        mode: Scaling mode - "sigma", "sqrt_var", or "var"
    """

    def __init__(
        self,
        base_model: ScoreModel,
        sigma_fn: Callable,
        clip_eps: float = 1e-7,
        mode: str = "sigma"
    ):
        # Inherit space and time embedder from base model
        super().__init__(base_model.space, base_model.time_embedder)

        self.base_model = base_model
        self.sigma_fn = sigma_fn
        self.clip_eps = clip_eps
        self.mode = mode

        assert mode in ["sigma", "sqrt_var", "var"], f"Unknown mode: {mode}"

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Get base score prediction
        score = self.base_model(t, x)

        # Get value from sigma_fn
        sigma_value = self.sigma_fn(t)

        # Apply scaling based on mode
        if self.mode == "sigma":
            # sigma_fn returns σ(t), scale by 1/σ(t)
            scaling = 1.0 / torch.clamp(sigma_value, min=self.clip_eps)
        elif self.mode == "sqrt_var":
            # sigma_fn returns σ²(t), scale by 1/√σ²(t) = 1/σ(t)
            scaling = 1.0 / torch.clamp(torch.sqrt(sigma_value), min=self.clip_eps)
        elif self.mode == "var":
            # sigma_fn returns σ²(t), scale by 1/σ²(t)
            scaling = 1.0 / torch.clamp(sigma_value, min=self.clip_eps**2)

        # Reshape scaling to broadcast correctly: (batch,) -> (batch, 1, 1, ...)
        scaling_view = scaling.view(-1, *([1] * len(self.data_dims)))

        return score * scaling_view

    def get_num_parameters(self) -> int:
        """Return number of parameters (from base model only)"""
        return self.base_model.get_num_parameters()


class ResidualWrapper(ScoreModel):
    """
    Wrapper that adds a residual connection: output = base_model(t, x) ± x

    Args:
        base_model: The score model to wrap
        positive: If True, adds x; if False, subtracts x
    """

    def __init__(self, base_model: ScoreModel, positive: bool = True):
        super().__init__(base_model.space, base_model.time_embedder)

        self.base_model = base_model
        self.positive = positive

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        score = self.base_model(t, x)

        if self.positive:
            return score + x
        else:
            return score - x

    def get_num_parameters(self) -> int:
        return self.base_model.get_num_parameters()


class LogTimeWrapper(ScoreModel):
    """
    Wrapper that applies log transform to time before passing to base model.

    Useful for emphasizing small time values.

    Args:
        base_model: The score model to wrap
    """

    def __init__(self, base_model: ScoreModel):
        super().__init__(base_model.space, base_model.time_embedder)

        self.base_model = base_model

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Apply log transform to time
        log_t = torch.log(t)

        return self.base_model(log_t, x)

    def get_num_parameters(self) -> int:
        return self.base_model.get_num_parameters()


class VParamWrapper(ScoreModel):
    """
    Wrapper for V-parameterization of score-based diffusion.

    V-parameterization uses a different target than standard score matching,
    which can improve training stability.

    Formula (matching models.py VParamWrapper):
        sigma_sq_t = sigma_fn(t)          # σ²(t)
        sigma_t = sqrt(sigma_sq_t)        # σ(t)
        sigma_sqrt =  sqrt(1 + sigma_sq_t)   #√(1+σ²(t))
        output = - x/(1.0 + sigma_sq_t_batch) - score/(sigma_t_batch * sigma_sqrt_batch)

    Simplified: output = - x/(1+σ²) - base(t,x) /(σ·√(1+σ²)) 

    Args:
        base_model: The score model to wrap
        sigma_fn: Function (t) -> σ²(t) returning variance
        clip_eps: Minimum value for clipping σ(t) for numerical stability
    """

    def __init__(
        self,
        base_model: ScoreModel,
        sigma_fn: Callable,
        clip_eps: float = 1e-7
    ):
        super().__init__(base_model.space, base_model.time_embedder)

        self.base_model = base_model
        self.sigma_fn = sigma_fn
        self.clip_eps = clip_eps

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        score = self.base_model(t, x)

        # sigma_fn returns σ²(t) (variance)
        sigma_sq_t = self.sigma_fn(t)
        sigma_t = torch.sqrt(sigma_sq_t).clip(min=self.clip_eps)

        # sigma_sqrt = √(1 + σ²(t))
        sigma_sqrt = torch.sqrt(1.0 + sigma_sq_t)

        # Reshape for broadcasting: (batch,) -> (batch, 1, 1, ...)
        ndim = [1] * len(self.data_dims)
        sigma_sq_t_batch = sigma_sq_t.view(-1, *ndim)
        sigma_t_batch = sigma_t.view(-1, *ndim)
        sigma_sqrt_batch = sigma_sqrt.view(-1, *ndim)

        # Formula: - x/(1+σ²) - base(t,x) /(σ·√(1+σ²)) 
        return - x/(1.0 + sigma_sq_t_batch) - score/(sigma_t_batch * sigma_sqrt_batch)

    def get_num_parameters(self) -> int:
        return self.base_model.get_num_parameters()


class SequentialWrapper(ScoreModel):
    """
    Wrapper that chains multiple score models/wrappers sequentially.

    Useful for composing multiple transformations.

    Args:
        *models: Variable number of ScoreModel instances to chain
    """

    def __init__(self, *models: ScoreModel):
        assert len(models) > 0, "Must provide at least one model"

        # All models must operate on the same space
        base_space = models[0].space
        for model in models[1:]:
            assert model.space == base_space, "All models must operate on same space"

        super().__init__(base_space, models[0].time_embedder)

        self.models = nn.ModuleList(models)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Pass through each model sequentially
        output = x
        for model in self.models:
            output = model(t, output)

        return output

    def get_num_parameters(self) -> int:
        return sum(model.get_num_parameters() for model in self.models)
