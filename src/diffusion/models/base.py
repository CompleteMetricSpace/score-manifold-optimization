#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base score model class.

This module defines the abstract ScoreModel interface that all
score-based diffusion models should implement.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Optional
import warnings

from diffusion.spaces import Space
from diffusion.utils.embeddings import TimeEmbedder


class ScoreModel(nn.Module, ABC):
    """
    Abstract base class for score-based diffusion models.

    A ScoreModel encapsulates:
    - The mathematical Space it operates on
    - A neural network architecture
    - Time embedding strategy
    - Forward pass: (t, x) -> score

    The model should be architecture-agnostic and work with any
    compatible Space (VectorSpace, MatrixSpace, TrajectorySpace, etc.).

    Attributes:
        space: The Space this model operates on
        data_dims: Shape of data in the space (tuple)
        time_embedder: Strategy for embedding time values (optional)
    """

    def __init__(
        self,
        space: Space,
        time_embedder: Optional[TimeEmbedder] = None
    ):
        """
        Initialize score model.

        Args:
            space: Mathematical space where model operates
            time_embedder: Time embedding strategy (defaults to SinCosEmbedder(128))
        """
        super().__init__()
        self.space = space
        self.data_dims = space.get_dims()
        assert time_embedder is not None, "time_embedder is required"
        self.time_embedder = time_embedder

    @abstractmethod
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute score function s_θ(t, x).

        This is the core method that neural networks must implement.
        It computes the score (gradient of log density) at time t for data x.

        Args:
            t: (batch,) time values in [0, T]
            x: (batch, *space.dims) data points in the space

        Returns:
            score: (batch, *space.dims) score estimates
                   Same shape as input x
        """
        pass

    def get_space(self) -> Space:
        """Return the space this model operates on."""
        return self.space

    def get_data_dims(self) -> tuple:
        """Return shape of data in the space."""
        return self.data_dims

    def get_time_embedder(self) -> TimeEmbedder:
        """Return the time embedding strategy."""
        return self.time_embedder

    def get_num_parameters(self) -> int:
        """
        Count trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"{self.__class__.__name__}(space={self.space}, "
                f"params={self.get_num_parameters()})")


class WrappedScoreModel(ScoreModel):
    """
    Wrapper for existing model architectures to make them ScoreModels.

    This class allows you to wrap any existing PyTorch model that has
    a forward(t, x) signature into a ScoreModel with proper Space awareness.

    This is useful for integrating legacy models or models from diffusion.models.models
    into the new architecture without rewriting them.

    Attributes:
        model: The wrapped PyTorch module
    """

    def __init__(
        self,
        space: Space,
        model: nn.Module,
        time_embedder: Optional[TimeEmbedder] = None
    ):
        """
        Initialize wrapped model.

        Args:
            space: Space where model operates
            model: Existing PyTorch module with forward(t, x) method
            time_embedder: Time embedding strategy (optional)
        """
        super().__init__(space, time_embedder)
        self.model = model

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through wrapped model.

        Args:
            t: (batch,) time values
            x: (batch, *space.dims) data points

        Returns:
            score: (batch, *space.dims) score estimates
        """
        return self.model(t, x)

    def __repr__(self):
        return (f"WrappedScoreModel(space={self.space}, "
                f"model={self.model.__class__.__name__}, "
                f"params={self.get_num_parameters()})")


# ============================================================================
# Helper functions for creating ScoreModels from existing architectures
# ============================================================================

def wrap_existing_model(model: nn.Module, space: Space) -> ScoreModel:
    """
    Wrap an existing model into a ScoreModel.

    This is a convenience function for quickly wrapping models from
    diffusion.models.models.

    Args:
        model: Existing PyTorch module with forward(t, x) signature
        space: Space where the model operates

    Returns:
        ScoreModel wrapping the given model

    Example:
        >>> import torch
        >>> from torch import nn
        >>> from diffusion.core import VectorSpace
        >>>
        >>> class CustomScoreNet(nn.Module):
        ...     def forward(self, t, x):
        ...         return x
        >>> old_model = CustomScoreNet()
        >>>
        >>> # Wrap it
        >>> space = VectorSpace(10)
        >>> score_model = wrap_existing_model(old_model, space)
    """
    return WrappedScoreModel(space, model)


def create_score_model_from_config(
    space: Space,
    model_type: str,
    **kwargs
) -> ScoreModel:
    """
    Legacy compatibility factory for ScoreModel creation.

    Active code should call diffusion.models.model_factory.create_model().
    This helper is retained for one release cycle for older call sites.

    Args:
        space: Space where model operates
        model_type: String identifier for model type
        **kwargs: Model-specific parameters

    Returns:
        ScoreModel instance

    Example:
        >>> space = VectorSpace(50)
        >>> model = create_score_model_from_config(
        ...     space=space,
        ...     model_type="MLP",
        ...     hidden_dim=256,
        ...     num_layers=4
        ... )
    """
    warnings.warn(
        "create_score_model_from_config() is deprecated and will be removed in a future release. "
        "Use diffusion.models.model_factory.create_model() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Legacy aliases accepted by older scripts/checkpoints.
    legacy_aliases = {
        "MLP": "MLPScoreModel",
        "FullyConnected": "MLPScoreModel",
    }
    resolved_model_type = legacy_aliases.get(model_type, model_type)

    # Import here to avoid circular dependency at module import time.
    from .model_factory import create_model

    return create_model(
        model_type=resolved_model_type,
        space=space,
        **kwargs,
    )


# ============================================================================
# Common model wrappers (from models.py) adapted for ScoreModel
# ============================================================================

class ResidualWrapper(ScoreModel):
    """
    Adds residual connection to a score model.

    output = model(t, x) ± x

    This is the ScoreModel-compatible version of ResNetTime from models.py.
    """

    def __init__(self, model: ScoreModel, positive: bool = True):
        """
        Initialize residual wrapper.

        Args:
            model: Base score model
            positive: If True, output = model(t,x) + x, else output = model(t,x) - x
        """
        super().__init__(model.space, model.time_embedder)
        self.model = model
        self.positive = positive

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection."""
        if self.positive:
            return self.model(t, x) + x
        else:
            return self.model(t, x) - x


class DiffusionScalingWrapper(ScoreModel):
    """
    Scales output by diffusion coefficient.

    output = model(t, x) / σ(t)

    This is the ScoreModel-compatible version of OutputDiffusionWrapper.
    Useful for parameterizing as ∇log p = -ε/σ instead of directly predicting score.
    """

    def __init__(
        self,
        model: ScoreModel,
        sigma_fn: callable,
        clip_eps: Optional[float] = None
    ):
        """
        Initialize diffusion scaling wrapper.

        Args:
            model: Base score model
            sigma_fn: Function (t) -> σ(t) computing diffusion coefficient
            clip_eps: If provided, clip σ(t) to avoid division by zero
        """
        super().__init__(model.space, model.time_embedder)
        self.model = model
        self.sigma_fn = sigma_fn
        self.clip_eps = clip_eps

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply diffusion scaling."""
        # Compute model output
        output = self.model(t, x)

        # Compute sigma(t)
        sigma_t = self.sigma_fn(t)

        # Clip if needed
        if self.clip_eps is not None:
            sigma_t = torch.clamp(sigma_t, min=self.clip_eps)

        # Scale output: shape (batch, *data_dims)
        # sigma_t is (batch,), need to broadcast
        ndim = [1 for _ in self.data_dims]
        sigma_view = sigma_t.view(-1, *ndim)

        return output / sigma_view


class LogTimeWrapper(ScoreModel):
    """
    Applies log transform to time before passing to model.

    output = model(log(t), x)

    Useful when model should focus more on small time values.
    """

    def __init__(self, model: ScoreModel):
        """
        Initialize log time wrapper.

        Args:
            model: Base score model
        """
        super().__init__(model.space, model.time_embedder)
        self.model = model

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply log time transform."""
        return self.model(torch.log(t), x)


class ChannelSqueezeWrapper(ScoreModel):
    """
    Squeezes/unsqueezes a dimension for compatibility.

    Useful for models expecting (batch, height, width) but receiving
    (batch, 1, height, width).

    This is the ScoreModel version of ChannelWrapper from models.py.
    """

    def __init__(self, model: ScoreModel, squeeze_dim: int = 1):
        """
        Initialize channel squeeze wrapper.

        Args:
            model: Base score model
            squeeze_dim: Dimension to squeeze/unsqueeze
        """
        super().__init__(model.space, model.time_embedder)
        self.model = model
        self.squeeze_dim = squeeze_dim

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Squeeze, apply model, unsqueeze."""
        x_squeezed = x.squeeze(self.squeeze_dim)
        output = self.model(t, x_squeezed)
        return output.unsqueeze(self.squeeze_dim)


# ============================================================================
# Utility functions
# ============================================================================

def validate_score_output(
    x: torch.Tensor,
    score: torch.Tensor,
    space: Space
) -> bool:
    """
    Validate that score output has correct shape.

    Args:
        x: Input data (batch, *space.dims)
        score: Model output (should be batch, *space.dims)
        space: Space definition

    Returns:
        True if shapes match, False otherwise
    """
    if x.shape != score.shape:
        return False
    if score.shape[1:] != space.get_dims():
        return False
    return True
