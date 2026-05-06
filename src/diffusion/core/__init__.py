"""
Core diffusion components.

This module contains the fundamental building blocks for diffusion models:
- spaces: Mathematical spaces (R^n, R^{m×n}, trajectory spaces)
- diffusion: Diffusion process definitions (VP, VE)
- sde: SDE solvers and utilities
"""

from diffusion.spaces import (
    Space, Constraint, Region, Manifold,
    EuclideanSpace, VectorSpace, MatrixSpace, TrajectorySpace,
    AffineSubspace, SliceSubspace, Sphere, Stiefel,
    UnitHypercube, UnitSquare, ShiftBall
)
from .diffusion import DiffusionProcess, VPDiffusion, VEDiffusion, create_diffusion
from .sde import SDE, sde_euler_maruyama_scalar, build_reverse_flow

__all__ = [
    # Spaces
    'Space', 'Constraint', 'Region', 'Manifold',
    'EuclideanSpace', 'VectorSpace', 'MatrixSpace', 'TrajectorySpace',
    'AffineSubspace', 'SliceSubspace', 'Sphere', 'Stiefel',
    'UnitHypercube', 'UnitSquare', 'ShiftBall',
    # Diffusion
    'DiffusionProcess', 'VPDiffusion', 'VEDiffusion', 'create_diffusion',
    # SDE
    'SDE', 'sde_euler_maruyama_scalar', 'build_reverse_flow',
]
