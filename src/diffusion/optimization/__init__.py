#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public DLF/DRGD optimization surface for the paper repository."""

from .types import (
    OptimizationResult,
    RiemannianConfig,
    RiemannianMetrics,
    RiemannianState,
    ScoreTangentConfig,
)
from .projector_base import Projector
from .constraint_projector import ConstraintProjector, build_constraint_projector
from .score_projector import ScoreProjector, build_score_projector
from .riemannian import (
    riemannian_flow_step,
    riemannian_step,
    run_riemannian_flow,
    run_riemannian_optimization,
)

__all__ = [
    "Projector",
    "ConstraintProjector",
    "build_constraint_projector",
    "ScoreProjector",
    "build_score_projector",
    "ScoreTangentConfig",
    "RiemannianConfig",
    "RiemannianState",
    "RiemannianMetrics",
    "OptimizationResult",
    "riemannian_flow_step",
    "riemannian_step",
    "run_riemannian_flow",
    "run_riemannian_optimization",
]
