#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP-based score models.

This module implements score models based on multi-layer perceptrons (MLPs).
These models flatten the input, process through fully-connected layers,
and reshape back to the original space.
"""

import torch
from torch import nn
from typing import Optional, List, Callable

from .base import ScoreModel
from diffusion.spaces import Space, EuclideanSpace, MatrixSpace
from diffusion.utils.embeddings import TimeEmbedder, SinCosEmbedder


class MLPScoreModel(ScoreModel):
    """
    MLP-based score model for flat/vector spaces.

    Architecture:
    1. Flatten input to vector
    2. Embed time
    3. Concatenate [flattened_x, time_embedding]
    4. Pass through MLP layers
    5. Reshape to original space shape

    This model works with any EuclideanSpace (VectorSpace, MatrixSpace, etc.)
    by flattening and reshaping appropriately.

    Attributes:
        flatten_dim: Total flattened dimension
        net: The MLP network
    """

    def __init__(
        self,
        space: EuclideanSpace,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        activation: str = "relu",
        time_embedder: Optional[TimeEmbedder] = None,
        reconcat_time_per_layer: bool = False,
        residual_connections: bool = False,
        layer_norm: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize MLP score model.

        Args:
            space: Euclidean space (VectorSpace, MatrixSpace, etc.)
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers (including input and output)
            activation: Activation function ("relu", "gelu", "silu")
            time_embedder: Time embedding strategy (defaults to SinCosEmbedder(128))
            reconcat_time_per_layer: Re-concatenate time embedding after each hidden layer
            residual_connections: Add hidden residual skip connections when dimensions match
            layer_norm: Whether to use layer normalization
            dropout: Dropout probability (0 = no dropout)
        """
        assert time_embedder is not None
        assert hidden_dim is not None
        assert num_layers is not None

        super().__init__(space, time_embedder)

        assert isinstance(space, EuclideanSpace), "MLPScoreModel requires EuclideanSpace"
        assert num_layers >= 2, "Need at least 2 layers (input + output)"

        self.flatten_dim = space.get_total_dim()
        time_dim = time_embedder.get_time_emb_dim()
        self.activation_name = activation
        self.reconcat_time_per_layer = reconcat_time_per_layer
        self.residual_connections = residual_connections
        self.layer_norm_enabled = layer_norm
        self.dropout_prob = dropout
        self.num_layers = num_layers
        self.time_dim = time_dim

        # Build MLP layers with optional per-layer time re-concatenation.
        self.layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        self.hidden_activation = self._get_activation(activation)
        self.hidden_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        input_dim = self.flatten_dim + time_dim

        for i in range(num_layers):
            # Determine output dimension
            if i == num_layers - 1:
                # Last layer: output to data dimension
                out_dim = self.flatten_dim
            else:
                # Hidden layers
                out_dim = hidden_dim

            self.layers.append(nn.Linear(input_dim, out_dim))

            if i < num_layers - 1:
                self.hidden_norms.append(nn.LayerNorm(out_dim) if layer_norm else nn.Identity())
                input_dim = out_dim + (time_dim if reconcat_time_per_layer else 0)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute score.

        Args:
            t: (batch,) time values
            x: (batch, *space.dims) data points

        Returns:
            score: (batch, *space.dims) score estimates
        """
        batch_size = x.shape[0]

        # Flatten spatial dimensions
        x_flat = x.reshape(batch_size, -1)

        # Embed time
        t_emb = self.time_embedder.embed(t)

        # Concatenate and forward
        h = torch.cat([x_flat, t_emb], dim=-1)
        for i, layer in enumerate(self.layers):
            skip = None
            if i < self.num_layers - 1 and self.residual_connections:
                if self.reconcat_time_per_layer and h.shape[-1] > self.time_dim:
                    # Residual skip uses hidden channels only (exclude concatenated time channels).
                    skip = h[:, :-self.time_dim]
                else:
                    skip = h

            h = layer(h)
            if i < self.num_layers - 1:
                h = self.hidden_norms[i](h)
                h = self.hidden_activation(h)
                h = self.hidden_dropout(h)
                if skip is not None and skip.shape == h.shape:
                    h = h + skip
                if self.reconcat_time_per_layer:
                    h = torch.cat([h, t_emb], dim=-1)
        score_flat = h

        # Reshape to original space
        return score_flat.reshape(batch_size, *self.data_dims)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "tanh": nn.Tanh()
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        return activations[name]

    def __repr__(self):
        return (f"MLPScoreModel(space={self.space}, "
                f"hidden_dim={self.layers[0].out_features if len(self.layers) > 1 else 'N/A'}, "
                f"num_layers={self.num_layers}, "
                f"activation={self.activation_name}, "
                f"reconcat_time_per_layer={self.reconcat_time_per_layer}, "
                f"residual_connections={self.residual_connections}, "
                f"params={self.get_num_parameters()})")


class ResidualMLPScoreModel(ScoreModel):
    """
    MLP with residual blocks (ResNet-style).

    Each block: LayerNorm -> Linear -> Activation -> Linear -> Add residual

    This architecture can learn deeper representations while maintaining
    gradient flow through skip connections.
    """

    def __init__(
        self,
        space: EuclideanSpace,
        hidden_dim: Optional[int] = None,
        num_blocks: Optional[int] = None,
        activation: str = "relu",
        time_embedder: Optional[TimeEmbedder] = None,
        dropout: float = 0.0
    ):
        """
        Initialize residual MLP score model.

        Args:
            space: Euclidean space
            hidden_dim: Hidden dimension (constant throughout)
            num_blocks: Number of residual blocks
            activation: Activation function
            time_embedder: Time embedding strategy
            dropout: Dropout probability
        """
        assert time_embedder is not None, "time_embedder is required"

        super().__init__(space, time_embedder)

        assert isinstance(space, EuclideanSpace), "ResidualMLPScoreModel requires EuclideanSpace"

        self.flatten_dim = space.get_total_dim()
        time_dim = time_embedder.get_time_emb_dim()

        # Input projection
        self.input_proj = nn.Linear(self.flatten_dim + time_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.flatten_dim)

        self.activation = self._get_activation(activation)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute score through residual blocks."""
        batch_size = x.shape[0]

        # Flatten and embed time
        x_flat = x.reshape(batch_size, -1)
        t_emb = self.time_embedder.embed(t)

        # Input projection
        h = self.input_proj(torch.cat([x_flat, t_emb], dim=-1))
        h = self.activation(h)

        # Residual blocks
        for block in self.blocks:
            h = block(h)

        # Output projection
        score_flat = self.output_proj(h)

        return score_flat.reshape(batch_size, *self.data_dims)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        return MLPScoreModel._get_activation(name)


class ResidualBlock(nn.Module):
    """Single residual block with LayerNorm."""

    def __init__(self, dim: int, activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = MLPScoreModel._get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through residual block."""
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class TimeConditionedMLPScoreModel(ScoreModel):
    """
    MLP with time conditioning via FiLM (Feature-wise Linear Modulation).

    The time embedding modulates (scales and shifts) hidden layer activations:
        h_modulated = γ(t) * h + β(t)

    This allows the model to adapt its behavior more flexibly based on time.
    """

    def __init__(
        self,
        space: EuclideanSpace,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        activation: str = "relu",
        time_embedder: Optional[TimeEmbedder] = None
    ):
        """
        Initialize FiLM-conditioned MLP.

        Args:
            space: Euclidean space
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            activation: Activation function
            time_embedder: Time embedding strategy
        """
        assert time_embedder is not None, "time_embedder is required"

        super().__init__(space, time_embedder)

        assert isinstance(space, EuclideanSpace), "TimeConditionedMLPScoreModel requires EuclideanSpace"
        assert num_layers >= 2

        self.flatten_dim = space.get_total_dim()
        time_dim = time_embedder.get_time_emb_dim()

        # Input layer
        self.input_fc = nn.Linear(self.flatten_dim, hidden_dim)

        # Hidden layers with FiLM conditioning
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers - 2)
        ])

        # FiLM parameters (scale and shift for each hidden layer)
        self.film_layers = nn.ModuleList([
            nn.Linear(time_dim, 2 * hidden_dim)  # Outputs [γ, β]
            for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_fc = nn.Linear(hidden_dim, self.flatten_dim)

        self.activation = MLPScoreModel._get_activation(activation)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward with FiLM conditioning."""
        batch_size = x.shape[0]

        # Flatten
        x_flat = x.reshape(batch_size, -1)

        # Embed time
        t_emb = self.time_embedder.embed(t)

        # Input layer
        h = self.input_fc(x_flat)
        h = self.activation(h)

        # Apply FiLM to input layer
        film_params = self.film_layers[0](t_emb)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        h = gamma * h + beta

        # Hidden layers with FiLM
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            h = self.activation(h)

            # Apply FiLM
            film_params = self.film_layers[i + 1](t_emb)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            h = gamma * h + beta

        # Output
        score_flat = self.output_fc(h)

        return score_flat.reshape(batch_size, *self.data_dims)


class _LayerNormTimeResidualBlock(nn.Module):
    """Residual block with LayerNorm/Linear/SiLU sublayers and additive time biases."""

    def __init__(self, hidden_dim: int, time_cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.time_bias1 = nn.Linear(time_cond_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.time_bias2 = nn.Linear(time_cond_dim, hidden_dim)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, time_ctx: torch.Tensor) -> torch.Tensor:
        residual = x

        h = self.fc1(self.norm1(x))
        h = h + self.time_bias1(time_ctx)
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc2(self.norm2(h))
        h = h + self.time_bias2(time_ctx)
        h = self.act(h)
        h = self.dropout(h)

        return residual + h


class LayerNormMLPScoreModel(ScoreModel):
    """
    Matrix score model with LayerNorm residual MLP blocks and log-SNR Fourier time conditioning.

    Architecture:
    1. Flatten matrix input X ∈ R^{m×n} to x ∈ R^{mn}
    2. Project x to hidden dimension
    3. Build time context from Fourier(log-SNR(t)) via two-layer MLP (fixed output dim 128)
    4. Process hidden features with residual blocks:
       (LayerNorm -> Linear -> +time_bias -> SiLU) x2, then residual add
    5. Project back to R^{mn} and reshape to matrix
    """

    def __init__(
        self,
        space: MatrixSpace,
        hidden_dim: int,
        num_blocks: int,
        fourier_dim: int = 128,
        logsnr_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dropout: float = 0.0,
        time_embedder: Optional[TimeEmbedder] = None
    ):
        """
        Initialize LayerNormMLPScoreModel.

        Args:
            space: MatrixSpace(m, n)
            hidden_dim: Backbone hidden width
            num_blocks: Number of residual blocks
            fourier_dim: Fourier feature dimension for log-SNR (must be even)
            logsnr_fn: Optional mapping t -> log-SNR(t). Defaults to clipped logit transform.
            dropout: Dropout probability inside residual blocks
            time_embedder: Kept for ScoreModel compatibility; not used for time conditioning here
        """
        if time_embedder is None:
            # Required by ScoreModel API; this model uses an internal time-conditioning path.
            time_embedder = SinCosEmbedder(4)
        super().__init__(space, time_embedder)

        assert isinstance(space, MatrixSpace), "LayerNormMLPScoreModel requires MatrixSpace"
        assert hidden_dim > 0, "hidden_dim must be positive"
        assert num_blocks > 0, "num_blocks must be positive"
        assert fourier_dim % 2 == 0 and fourier_dim > 0, "fourier_dim must be a positive even integer"

        self.flatten_dim = space.get_total_dim()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.fourier_dim = fourier_dim
        self.time_cond_dim = 128
        self.logsnr_fn = logsnr_fn
        self.logsnr_eps = 1e-5

        # Reuse existing Fourier and 2-layer time MLP primitives.
        from diffusion.models.models import SinusoidalTimeEmbedding, TimeMLP
        self.logsnr_fourier = SinusoidalTimeEmbedding(time_dim=fourier_dim)
        self.time_mlp = TimeMLP(in_dim=fourier_dim, out_dim=self.time_cond_dim)

        self.input_proj = nn.Linear(self.flatten_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            _LayerNormTimeResidualBlock(hidden_dim, self.time_cond_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, self.flatten_dim)

    def _compute_logsnr(self, t: torch.Tensor) -> torch.Tensor:
        """Map t to log-SNR scalar values with a stable default."""
        if self.logsnr_fn is not None:
            logsnr = self.logsnr_fn(t)
            if logsnr.ndim == 2 and logsnr.shape[1] == 1:
                logsnr = logsnr[:, 0]
            assert logsnr.shape == t.shape, "logsnr_fn must return shape (batch,)"
            return logsnr

        t_clamped = torch.clamp(t, min=self.logsnr_eps, max=1.0 - self.logsnr_eps)
        return torch.log(t_clamped) - torch.log1p(-t_clamped)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute score with internal log-SNR Fourier conditioning."""
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)

        t = t.to(device=x.device, dtype=x.dtype)
        logsnr = self._compute_logsnr(t)
        t_fourier = self.logsnr_fourier(logsnr)
        t_ctx = self.time_mlp(t_fourier)

        h = self.input_proj(x_flat)
        for block in self.blocks:
            h = block(h, t_ctx)

        score_flat = self.output_proj(h)
        return score_flat.reshape(batch_size, *self.data_dims)
