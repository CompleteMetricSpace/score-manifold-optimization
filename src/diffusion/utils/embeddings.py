#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time embedding strategies for diffusion models.

This module defines various time embedding approaches:
- SinCosEmbedder: Sinusoidal positional embeddings (standard)
- LinearEmbedder: Direct time value
- LogEmbedder: Log-transformed time
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


class TimeEmbedder(ABC):
    """
    Abstract base class for time embedding strategies.

    A TimeEmbedder transforms scalar time values into vector embeddings
    that can be fed into neural networks.
    """

    @abstractmethod
    def embed(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed time values.

        Args:
            t: (batch,) scalar time values

        Returns:
            embeddings: (batch, time_emb_dim) embedded time values
        """
        pass

    @abstractmethod
    def get_time_emb_dim(self) -> int:
        """
        Return dimensionality of time embeddings.

        Returns:
            Integer dimension of embedding space
        """
        pass


class SinCosEmbedder(TimeEmbedder):
    """
    Sinusoidal positional embedding (Vaswani et al., 2017).

    Standard time embedding used in diffusion models and transformers.
    Maps t ∈ R to [sin(ω_1 t), cos(ω_1 t), ..., sin(ω_d/2 t), cos(ω_d/2 t)]
    where ω_k = 10000^(-2k/d).

    Attributes:
        time_emb_dim: Dimension of embedding (must be even)
    """

    def __init__(self, time_emb_dim: int):
        """
        Initialize sinusoidal embedder.

        Args:
            time_emb_dim: Dimension of embedding (must be even and > 2)
        """
        assert time_emb_dim % 2 == 0 and time_emb_dim > 2, \
            "time_emb_dim must be even and > 2"
        self.time_emb_dim = time_emb_dim

    def embed(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal embedding.

        Args:
            t: (batch,) time values

        Returns:
            embeddings: (batch, time_emb_dim)
        """
        half_dim = self.time_emb_dim // 2

        # Compute frequency scale
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)

        # Broadcast: t (batch,) -> (batch, 1), emb (half_dim,) -> (1, half_dim)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # (batch, half_dim)

        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (batch, time_emb_dim)

        return emb

    def get_time_emb_dim(self) -> int:
        """Return embedding dimension."""
        return self.time_emb_dim

    def __repr__(self):
        return f"SinCosEmbedder(dim={self.time_emb_dim})"


class LinearEmbedder(TimeEmbedder):
    """
    Linear (identity) time embedding.

    Simply returns t as a 1D embedding (no transformation).

    Attributes:
        None
    """

    def __init__(self):
        """Initialize linear embedder."""
        self.time_emb_dim = 1

    def embed(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return time as 1D embedding.

        Args:
            t: (batch,) time values

        Returns:
            embeddings: (batch, 1)
        """
        return t.unsqueeze(1)

    def get_time_emb_dim(self) -> int:
        """Return embedding dimension (always 1)."""
        return self.time_emb_dim

    def __repr__(self):
        return "LinearEmbedder"


class LogEmbedder(TimeEmbedder):
    """
    Logarithmic time embedding.

    Embeds t as -log(t), useful for emphasizing small time values.

    Attributes:
        None
    """

    def __init__(self):
        """Initialize logarithmic embedder."""
        self.time_emb_dim = 1

    def embed(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return -log(t) as 1D embedding.

        Args:
            t: (batch,) time values (must be positive)

        Returns:
            embeddings: (batch, 1)
        """
        return -torch.log(t).unsqueeze(1)

    def get_time_emb_dim(self) -> int:
        """Return embedding dimension (always 1)."""
        return self.time_emb_dim

    def __repr__(self):
        return "LogEmbedder"


class LinearLogSigmaEmbedder(TimeEmbedder):
    """
    Combined linear + log + sigma embedding.

    Embeds t as [t, -log(t), σ²(t)] where σ²(t) is provided.

    Attributes:
        sigma_sq: Function mapping t to σ²(t)
    """

    def __init__(self, sigma_sq: callable):
        """
        Initialize combined embedder.

        Args:
            sigma_sq: Function (batch,) -> (batch,) computing variance schedule
        """
        self.time_emb_dim = 3
        self.sigma_sq = sigma_sq

    def embed(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return [t, -log(t), σ²(t)] as 3D embedding.

        Args:
            t: (batch,) time values

        Returns:
            embeddings: (batch, 3)
        """
        return torch.cat([
            t.unsqueeze(1),
            -torch.log(t).unsqueeze(1),
            self.sigma_sq(t).unsqueeze(1)
        ], dim=1)

    def get_time_emb_dim(self) -> int:
        """Return embedding dimension (always 3)."""
        return self.time_emb_dim

    def __repr__(self):
        return "LinearLogSigmaEmbedder"
