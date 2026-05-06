#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal/sequence score models.

These models operate on TrajectorySpace (or MatrixSpace with shape (horizon, dim))
and preserve the sequence structure using 1D convolutions or temporal attention.
"""

import torch
from torch import nn
from typing import Optional, Tuple

from .base import ScoreModel
from diffusion.spaces import Space, TrajectorySpace, MatrixSpace
from diffusion.utils.embeddings import TimeEmbedder, SinCosEmbedder


class UNet1DTimeScoreModel(ScoreModel):
    """
    1D UNet score model for trajectory/sequence data.

    Wraps UNet1DTime from diffusion.models.models as a ScoreModel.
    Operates on TrajectorySpace or MatrixSpace with shape (horizon, channels).

    The underlying UNet1DTime expects input (B, N, d) and outputs (B, N, d),
    where N is the sequence length and d is the feature dimension.

    Args:
        space: TrajectorySpace or MatrixSpace defining (horizon, channels)
        model_channels: Base channel width for UNet
        channel_mults: Per-level channel multipliers (controls depth)
        num_res_blocks: Number of residual blocks per level
        time_emb_dim: Dimension of sinusoidal time embedding
        dropout: Dropout rate
        time_embedder: Not used (UNet1DTime has its own time embedding)
    """

    def __init__(
        self,
        space: Space,
        model_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
        time_embedder: Optional[TimeEmbedder] = None,
    ):
        super().__init__(space, time_embedder)

        dims = space.get_dims()
        assert len(dims) == 2, f"UNet1DTimeScoreModel requires 2D space (horizon, channels), got {dims}"
        horizon, in_channels = dims

        # Import from active consolidated model primitives.
        from diffusion.models.models import UNet1DTime
        self.module = UNet1DTime(
            in_channels=in_channels,
            model_channels=model_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) time values
            x: (B, horizon, channels)

        Returns:
            score: (B, horizon, channels)
        """
        return self.module(t, x)


class TemporalUnetScoreModel(ScoreModel):
    """
    Temporal UNet score model for trajectory data (Janner et al. architecture).

    Wraps TemporalUnet from diffusion.models.models as a ScoreModel.
    Operates on TrajectorySpace with shape (horizon, transition_dim).

    Args:
        space: TrajectorySpace defining (horizon, input_dim + output_dim)
        dim: Base channel dimension for UNet
        dim_mults: Channel dimension multipliers per level
        attention: Whether to use linear attention
        time_embedder: Not used (TemporalUnet has its own time embedding)
    """

    def __init__(
        self,
        space: Space,
        dim: int = 32,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        attention: bool = False,
        time_embedder: Optional[TimeEmbedder] = None,
    ):
        super().__init__(space, time_embedder)

        dims = space.get_dims()
        assert len(dims) == 2, f"TemporalUnetScoreModel requires 2D space (horizon, channels), got {dims}"
        horizon, transition_dim = dims

        # cond_dim is the output_dim if TrajectorySpace, else transition_dim
        if isinstance(space, TrajectorySpace):
            cond_dim = space.output_dim
        else:
            cond_dim = transition_dim

        # Import from active consolidated model primitives.
        from diffusion.models.models import TemporalUnet
        self.module = TemporalUnet(
            horizon=horizon,
            transition_dim=transition_dim,
            cond_dim=cond_dim,
            dim=dim,
            dim_mults=dim_mults,
            attention=attention,
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) time values
            x: (B, horizon, transition_dim)

        Returns:
            score: (B, horizon, transition_dim)
        """
        return self.module(t, x)
