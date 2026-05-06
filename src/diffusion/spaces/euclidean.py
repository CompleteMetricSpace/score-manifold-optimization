#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Euclidean spaces.

This module defines concrete Euclidean space implementations:
- VectorSpace: R^n
- MatrixSpace: R^{m×n}
- TrajectorySpace: R^{T×D} for time-series/control data
"""

import torch
from typing import Tuple
from .base import Space


class EuclideanSpace(Space):
    """
    Base class for flat Euclidean spaces R^{d1 × d2 × ... × dn}.

    Attributes:
        dims: Tuple of dimensions defining the space shape
    """

    def __init__(self, dims: Tuple[int, ...]):
        """
        Initialize Euclidean space.

        Args:
            dims: Tuple of positive integers defining the space shape
        """
        assert all(d > 0 for d in dims), "All dimensions must be positive"
        self.dims = tuple(dims)

    def get_dims(self) -> Tuple[int, ...]:
        """Return shape of elements in this space."""
        return self.dims

    def sample_uniform(self, n: int) -> torch.Tensor:
        """
        Sample from standard normal distribution N(0, I).

        Args:
            n: Number of samples

        Returns:
            Tensor of shape (n, *self.dims)
        """
        return torch.randn(n, *self.dims)

    def validate_shape(self, x: torch.Tensor) -> bool:
        """
        Check if x has correct shape.

        Args:
            x: Tensor to validate

        Returns:
            True if x.shape[1:] == self.dims
        """
        return x.shape[1:] == self.dims

    def __repr__(self):
        return f"{self.__class__.__name__}{self.dims}"


class VectorSpace(EuclideanSpace):
    """
    R^n vector space.

    Elements have shape (n,)

    Attributes:
        dim: Dimension n of the vector space
    """

    def __init__(self, dim: int):
        """
        Initialize vector space.

        Args:
            dim: Dimension of the vector space
        """
        super().__init__((dim,))
        self.dim = dim

    def __repr__(self):
        return f"R^{self.dim}"


class MatrixSpace(EuclideanSpace):
    """
    R^{m×n} matrix space.

    Elements have shape (m, n)

    Attributes:
        m: Number of rows
        n: Number of columns
    """

    def __init__(self, m: int, n: int):
        """
        Initialize matrix space.

        Args:
            m: Number of rows
            n: Number of columns
        """
        super().__init__((m, n))
        self.m = m
        self.n = n

    def __repr__(self):
        return f"R^({self.m}×{self.n})"


class TrajectorySpace(EuclideanSpace):
    """
    Space of trajectories: R^{T×D} where D = input_dim + output_dim.

    Used for time-series and control system data.
    Each trajectory is a sequence of T timesteps, where each timestep
    contains both input (control) and output (state/measurement) components.

    Attributes:
        horizon: Number of time steps T
        input_dim: Dimension of input/control at each timestep
        output_dim: Dimension of output/state at each timestep
    """

    def __init__(self, horizon: int, input_dim: int, output_dim: int):
        """
        Initialize trajectory space.

        Args:
            horizon: Number of time steps T
            input_dim: Dimension of input/control
            output_dim: Dimension of output/state
        """
        super().__init__((horizon, input_dim + output_dim))
        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim

    def split_input_output(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split trajectory into (input, output) components.

        Args:
            X: (batch, horizon, input_dim + output_dim) trajectories

        Returns:
            u: (batch, horizon, input_dim) inputs
            y: (batch, horizon, output_dim) outputs
        """
        return X[..., :self.input_dim], X[..., self.input_dim:]

    def __repr__(self):
        return f"TrajectorySpace(T={self.horizon}, u_dim={self.input_dim}, y_dim={self.output_dim})"
