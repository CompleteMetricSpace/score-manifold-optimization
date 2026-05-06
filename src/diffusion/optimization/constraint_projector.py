#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import torch

try:
    from torch.func import jvp as torch_jvp
except Exception:
    from functorch import jvp as torch_jvp

from diffusion.spaces import Manifold
from .projector_base import Projector


@dataclass
class ConstraintProjector(Projector):
    constraint: object

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.constraint.project(x)

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if isinstance(self.constraint, Manifold) and hasattr(self.constraint, "project_tangent"):
            x_on = self.constraint.project(x)
            return self.constraint.project_tangent(x_on, v)

        if hasattr(self.constraint, "project_tangent") and callable(getattr(self.constraint, "project_tangent")):
            # Use native tangent projection when available (e.g. dynamics constraints)
            # without forcing an expensive point projection first.
            return self.constraint.project_tangent(x, v)


        # Generic differentiable fallback: tangent from Jacobian of projection map.
        _, v_tan = torch_jvp(self.project, (x,), (v,))
        return v_tan


def build_constraint_projector(constraint) -> ConstraintProjector:
    return ConstraintProjector(constraint=constraint)


def coerce_to_projector(obj, name: str) -> Projector:
    if isinstance(obj, Projector):
        return obj
    if hasattr(obj, "project") and callable(getattr(obj, "project")):
        return build_constraint_projector(obj)
    raise TypeError(f"{name} must be a Projector or implement project(x)")
