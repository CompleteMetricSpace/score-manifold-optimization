"""
Mathematical spaces and constraints.

This module defines:
- Space hierarchy: Ambient mathematical spaces
- Constraint hierarchy: Subsets/constraints within spaces
- Concrete implementations: Manifolds, regions, trajectory constraints
"""

from .base import Space, Constraint, Region
from .euclidean import EuclideanSpace, VectorSpace, MatrixSpace, TrajectorySpace
from .manifolds import Manifold, AffineSubspace, SliceSubspace, Sphere, Stiefel
from .regions import UnitHypercube, UnitSquare, ShiftBall

__all__ = [
    # Base classes
    'Space', 'Constraint', 'Region', 'Manifold',
    # Euclidean spaces
    'EuclideanSpace', 'VectorSpace', 'MatrixSpace', 'TrajectorySpace',
    # Manifolds
    'AffineSubspace', 'SliceSubspace', 'Sphere', 'Stiefel',
    # Regions
    'UnitHypercube', 'UnitSquare', 'ShiftBall',
]
