#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Region constraints.

This module defines concrete region implementations:
- UnitHypercube: [0,1]^n
- UnitSquare: [0,1]^2
- ShiftBall: Unit ball shifted by (0.5, 0.5)
"""

import torch
import math
from .base import Region
from .euclidean import VectorSpace


class UnitHypercube(Region):
    """
    Unit hypercube [0,1]^n.

    Attributes:
        n: Dimension of the hypercube
    """

    def __init__(self, n: int):
        """
        Initialize unit hypercube.

        Args:
            n: Dimension
        """
        super().__init__(VectorSpace(n))
        self.n = n

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto [0,1]^n by clamping."""
        return torch.clamp(x, 0, 1)

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        """Violation is Euclidean distance to cube."""
        return self.distance(x)

    def sample(self, n: int) -> torch.Tensor:
        """Sample uniformly from [0,1]^n."""
        return torch.rand(n, self.n)


class UnitSquare(Region):
    """
    Unit square [0,1]^2.

    Special case of UnitHypercube with methods for boundary sampling.

    Attributes:
        None (inherits from Region with VectorSpace(2))
    """

    def __init__(self):
        """Initialize unit square."""
        super().__init__(VectorSpace(2))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto [0,1]^2 by clamping."""
        return torch.clamp(x, 0, 1)

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        """Violation is Euclidean distance to square."""
        return self.distance(x)

    def sample(self, n: int) -> torch.Tensor:
        """Sample uniformly from [0,1]^2."""
        return torch.rand(n, 2)

    def boundary(self, n: int) -> torch.Tensor:
        """
        Sample n points on the boundary of the unit square.

        Args:
            n: Number of boundary points (will be rounded to multiple of 4)

        Returns:
            Points on boundary: (n, 2)
        """
        n_per_side = n // 4
        t = torch.linspace(0, 1, n_per_side)
        ones = torch.ones(n_per_side)
        zeros = torch.zeros(n_per_side)

        # Four sides of the square
        side1 = torch.stack([t, zeros], dim=1)  # Bottom
        side2 = torch.stack([ones, t], dim=1)   # Right
        side3 = torch.stack([t.flip(0), ones], dim=1)  # Top
        side4 = torch.stack([zeros, t.flip(0)], dim=1)  # Left

        return torch.cat([side1, side2, side3, side4], dim=0)


class ShiftBall(Region):
    """
    Unit ball shifted by (0.5, 0.5) in R^2.

    This is the set {(x,y) : (x+0.5)^2 + (y+0.5)^2 <= 1}

    Attributes:
        None (inherits from Region with VectorSpace(2))
    """

    def __init__(self):
        """Initialize shifted ball."""
        super().__init__(VectorSpace(2))
        self.center = torch.tensor([[0.5, 0.5]])

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project onto shifted ball.

        Args:
            x: (batch, 2) points

        Returns:
            Projected points: (batch, 2)
        """
        # Shift to origin
        z = x + self.center.to(x.device)

        # Compute norms
        norms = z.norm(dim=1, keepdim=True)

        # Project to unit ball and shift back
        p = torch.where(norms > 1, z / norms, z)
        return p - self.center.to(x.device)

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        """Violation is Euclidean distance to ball."""
        return self.distance(x)

    def sample(self, n: int) -> torch.Tensor:
        """
        Sample uniformly from shifted ball.

        Uses rejection sampling from the unit sphere.

        Args:
            n: Number of samples

        Returns:
            Points in ball: (n, 2)
        """
        # Sample from 2D normal distribution
        rand_pts = torch.randn(n, 2)

        # Uniform radius with correct 2D distribution
        unif = torch.rand(n, 1)
        r = unif.pow(0.5)  # r^2 ~ Uniform[0,1] for 2D

        # Normalize and scale by radius
        data = r * rand_pts / rand_pts.norm(dim=1, keepdim=True)

        # Shift
        return data - self.center

    def boundary(self, n: int) -> torch.Tensor:
        """
        Sample n points on the boundary (circle).

        Args:
            n: Number of boundary points

        Returns:
            Points on boundary: (n, 2)
        """
        theta = torch.linspace(0, 2 * math.pi, n)
        x = torch.cos(theta).unsqueeze(1)
        y = torch.sin(theta).unsqueeze(1)
        return torch.cat([x, y], dim=1) - self.center
