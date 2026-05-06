#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix-specialized score models.

These models operate on MatrixSpace and preserve matrix structure
(i.e., they don't flatten the input).
"""

import torch
from torch import nn
from typing import Optional

from .base import ScoreModel
from diffusion.spaces import Space, MatrixSpace
from diffusion.utils.embeddings import TimeEmbedder, SinCosEmbedder


class TraceMLPScoreModel(ScoreModel):
    """
    Score model using trace-probe + per-column MLP with time embedding.

    Same as TraceMLPScoreModel but uses a TimeEmbedder for time conditioning
    instead of raw scalar t.

    Args:
        space: MatrixSpace(d, d) - must be square
        trace_probe: Number of trace probes
        hidden_dim: Hidden layer dimension
        num_layers: Number of MLP layers
        time_embedder: Time embedding strategy (required)
    """

    def __init__(
        self,
        space: MatrixSpace,
        trace_probes: Optional[int] = None,
        hidden_dim=Optional[int],
        num_layers=Optional[int],
        time_embedder: Optional[TimeEmbedder] = None,
        activation: Optional[str] = None
    ):
        assert time_embedder is not None, "time_embedder is required"
        super().__init__(space, time_embedder)

        assert isinstance(space, MatrixSpace), "TraceMLPScoreModel requires MatrixSpace"
        assert space.m == space.n, "Requires square matrices"

        # Import from active consolidated model primitives.
        from diffusion.models.models import TraceMLPTimeModuleTimeEmbedding
        self.module = TraceMLPTimeModuleTimeEmbedding(
            d=space.n, m=trace_probes, hidden_dim=hidden_dim, num_layers=num_layers,
            time_emb=time_embedder, activation=activation
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.module(t, x)
