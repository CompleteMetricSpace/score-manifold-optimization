#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DRGD reference-tracking helper built on run_riemannian_optimization."""

from __future__ import annotations

from typing import Any, Callable, List

import torch

from diffusion.optimization import RiemannianConfig, run_riemannian_optimization

from .objectives import build_reference_tracking_objective


def run_reference_tracking(
    *,
    reference_trajectory: torch.Tensor,
    slice_spec: List[Any],
    projector: Any,
    cfg: RiemannianConfig,
    x0: torch.Tensor,
    loss_type="L2",
    log_fn: Callable[[torch.Tensor], Any] | None = None,
) -> dict[str, Any]:
    """Optimize a trajectory to track a reference using DRGD."""
    reference = torch.as_tensor(reference_trajectory)
    objective_fn = build_reference_tracking_objective(reference_trajectory=reference,slice_spec=slice_spec, loss_type=loss_type)
    x0 = x0.detach().clone()

    result = run_riemannian_optimization(
        x0=x0,
        objective_fn=objective_fn,
        grad_objective_fn=None,
        projector=projector,
        cfg=cfg,
    )

    history = [
        float(objective_fn(x).mean().detach().item()) for x in result.trajectory
    ]
    if log_fn is None:
        log_output: list[Any] = []
    else:
        with torch.no_grad():
            log_output = [log_fn(x) for x in result.trajectory]

    payload: dict[str, Any] = {
        "result": result,
        "objective_history": history,
        "reference": reference.detach().clone(),
        "log_output": log_output,
    }

    return payload


__all__ = ["run_reference_tracking"]
