#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
import torch


@dataclass
class ScoreTangentConfig:
    method: Literal["jvp", "vjp", "jacobian", "jacobian_truncated", "empirical_basis", "jvp_separate_time"] = "jvp"
    n_samples: Optional[int] = None
    eps_samples: float = 5e-2
    truncation_rtol: float = 1e-6
    remove_first_sv: bool = True
    max_jacobian_dim: Optional[int] = None
    tangent_separate_time: Optional[float] = None


@dataclass
class RiemannianConfig:
    step_size: float
    n_steps: int
    tangent_mode: str = "projector"  # none|projector|tangent_projector|project-at-start
    landing_gain: Optional[float] = None
    x_eval_fns: Optional[Any] = None


@dataclass
class RiemannianState:
    x: torch.Tensor
    step: int = 0


@dataclass
class RiemannianMetrics:
    objective: float
    grad_norm: float
    proj_grad_norm: float
    extras: Optional[Dict[str, float]] = None


@dataclass
class OptimizationResult:
    final_x: torch.Tensor
    trajectory: List[torch.Tensor]
    metrics: List[Any]
    extra: Optional[Dict[str, Any]] = None
