#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base classes for spaces and constraints.

This module defines the abstract interfaces for:
- Space: Ambient mathematical spaces
- Constraint: Subsets/constraints within spaces
- Region: General region constraints
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Tuple


class Space(ABC):
    """
    Abstract base class for mathematical spaces.

    A Space defines the ambient mathematical space where data lives.
    Examples: R^n, R^{m×n}, R^{T×D} (trajectories)
    """

    @abstractmethod
    def get_dims(self) -> Tuple[int, ...]:
        """
        Return shape of elements in this space.

        Returns:
            Tuple of dimensions, e.g., (n,) for R^n, (m, n) for R^{m×n}
        """
        pass

    @abstractmethod
    def sample_uniform(self, n: int) -> torch.Tensor:
        """
        Sample n points uniformly (or with natural measure) from the space.

        Args:
            n: Number of samples

        Returns:
            Tensor of shape (n, *self.get_dims())
        """
        pass

    @abstractmethod
    def validate_shape(self, x: torch.Tensor) -> bool:
        """
        Check if tensor x has correct shape for this space.

        Args:
            x: Tensor to validate

        Returns:
            True if shape is valid, False otherwise
        """
        pass

    def get_total_dim(self) -> int:
        """
        Return total flattened dimension.

        Returns:
            Product of all dimensions
        """
        return int(np.prod(self.get_dims()))

    def __eq__(self, other) -> bool:
        """Check equality based on type and dimensions."""
        if not isinstance(other, Space):
            return False
        return type(self) == type(other) and self.get_dims() == other.get_dims()

    def __hash__(self):
        """Make Space hashable based on type and dimensions."""
        return hash((type(self).__name__, self.get_dims()))


class Constraint(ABC):
    """
    Abstract constraint: defines a subset S ⊂ Space.

    Every constraint has:
    - A reference to the ambient Space it lives in
    - Methods to measure violation and project onto the constraint

    Note: Constraints are used for evaluation/metrics only in this framework.
    They do NOT influence training or sampling.
    """

    def __init__(self, space: Space):
        """
        Initialize constraint.

        Args:
            space: The ambient Space this constraint lives in
        """
        self._space = space

    @abstractmethod
    def violation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Measure constraint violation.

        Args:
            x: (batch, *space.dims) points in ambient space

        Returns:
            violations: (batch,) non-negative scalars (0 = satisfied)
        """
        pass

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x onto constraint set.

        Args:
            x: (batch, *space.dims) points in ambient space

        Returns:
            x_proj: (batch, *space.dims) closest points in constraint set
        """
        pass

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n points from the constraint set.

        Args:
            n: Number of samples

        Returns:
            samples: (n, *space.dims)
        """
        pass

    def closest(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for project (for backward compatibility).

        Args:
            x: Points in ambient space

        Returns:
            Closest points in constraint set
        """
        return self.project(x)

    def distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance to constraint set.

        Args:
            x: (batch, *space.dims) points in ambient space

        Returns:
            distances: (batch,) distances to constraint
        """
        x_proj = self.project(x)
        diff = (x - x_proj).reshape(x.shape[0], -1)
        return diff.norm(dim=1)

    def get_space(self) -> Space:
        """Return the ambient space."""
        return self._space

    def get_dims(self) -> Tuple[int, ...]:
        """Return dimensions of ambient space."""
        return self.get_space().get_dims()


class Region(Constraint):
    """
    General region constraint (not necessarily smooth).

    Examples: boxes, balls, unions of sets
    Base class for both simple regions and smooth manifolds.
    """
    pass
