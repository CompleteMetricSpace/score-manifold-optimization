#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Dict, Optional
import torch

from diffusion.utils import compute_grad
from diffusion.utils.eval_hooks import evaluate_iterate, normalize_eval_fns

try:
    from torch.func import linearize as torch_linearize
except Exception:
    from functorch import linearize as torch_linearize

from .types import (
    RiemannianConfig,
    RiemannianState,
    RiemannianMetrics,
    OptimizationResult,
)
def riemannian_step(
    state: RiemannianState,
    objective_fn: Callable,
    grad_objective_fn: Optional[Callable],
    projector,
    cfg: RiemannianConfig,
):
    tangent_mode = (cfg.tangent_mode or "projector").lower()
    if tangent_mode not in {"none", "projector", "tangent_projector", "project-at-start"}:
        raise ValueError(
            f"Invalid tangent_mode='{cfg.tangent_mode}'. "
            "Use 'none', 'projector', 'tangent_projector', or 'project-at-start'."
        )
    x = state.x

    if tangent_mode == "project-at-start":
        tangent_cfg = getattr(projector, "tangent_cfg", None)
        method = getattr(tangent_cfg, "method", None) if tangent_cfg is not None else None
        if method is None:
            raise NotImplementedError(
                "tangent_mode='project-at-start' requires projector.tangent_cfg.method == 'jvp' or 'vjp'."
            )
        method = str(method)
        if method == "vjp":
            x_req = x.detach().clone().requires_grad_(True)
            x_proj = projector.project(x_req)
            g = compute_grad(
                x_proj.detach(),
                objective_fn,
                grad_f=grad_objective_fn,
                reduction="sum",
                create_graph=False,
            )
            pg = torch.autograd.grad((x_proj * g.detach()).sum(), x_req)[0]
            x_next = x_proj.detach() - cfg.step_size * pg
        elif method == "jvp":
            x_proj, jvp_fn = torch_linearize(projector.project, x)
            g = compute_grad(
                x_proj,
                objective_fn,
                grad_f=grad_objective_fn,
                reduction="sum",
                create_graph=False,
            )
            pg = jvp_fn(g)
            x_next = x_proj - cfg.step_size * pg
        else:
            raise NotImplementedError(
                "tangent_mode='project-at-start' is only supported when projector.tangent_cfg.method "
                "is 'jvp' or 'vjp'. "
                f"Got method='{method}'."
            )
    else:
        g = compute_grad(
            x,
            objective_fn,
            grad_f=grad_objective_fn,
            reduction="sum",
            create_graph=False,
        )

        if tangent_mode == "none":
            pg = g
        else:
            pg = projector.project_tangent(x, g)
        x_next = x - cfg.step_size * pg
        x_next = projector.project(x_next)

    objective_val = objective_fn(x_next)
    objective_tensor = torch.as_tensor(objective_val, device=x_next.device, dtype=x_next.dtype)
    objective_scalar = float(objective_tensor.mean().item())
    metrics = RiemannianMetrics(
        objective=objective_scalar,
        grad_norm=float(g.norm().item()),
        proj_grad_norm=float(pg.norm().item()),
        extras=None,
    )
    return RiemannianState(x=x_next.detach(), step=state.step + 1), metrics


def riemannian_flow_step(
    state: RiemannianState,
    objective_fn: Callable,
    grad_objective_fn: Optional[Callable],
    projector,
    cfg: RiemannianConfig,
):
    tangent_mode = (cfg.tangent_mode or "projector").lower()
    if tangent_mode != "projector":
        raise ValueError(
            "run_riemannian_flow only supports tangent_mode='projector' because "
            "denoising landing flow always uses the projector tangent action."
        )
    if cfg.landing_gain is None:
        raise ValueError(
            "run_riemannian_flow requires cfg.landing_gain to be set explicitly. "
            "Use landing_gain=0.0 if you intentionally want no landing term."
        )

    x = state.x
    x_proj = projector.project(x)
    g = compute_grad(
        x_proj,
        objective_fn,
        grad_f=grad_objective_fn,
        reduction="sum",
        create_graph=False,
    )

    pg = projector.project_tangent(x, g)

    landing_residual = x_proj - x
    x_next = x - cfg.step_size * pg + cfg.step_size * float(cfg.landing_gain) * landing_residual

    extras: Dict[str, float] = {
        "landing_residual_norm": float(landing_residual.norm().item()),
    }

    objective_val = objective_fn(x_proj)
    objective_tensor = torch.as_tensor(objective_val, device=x_proj.device, dtype=x_proj.dtype)
    objective_scalar = float(objective_tensor.mean().item())
    metrics = RiemannianMetrics(
        objective=objective_scalar,
        grad_norm=float(g.norm().item()),
        proj_grad_norm=float(pg.norm().item()),
        extras=extras,
    )
    return RiemannianState(x=x_next.detach(), step=state.step + 1), metrics


def run_riemannian_optimization(
    x0: torch.Tensor,
    objective_fn: Callable,
    grad_objective_fn: Optional[Callable],
    projector,
    cfg: RiemannianConfig,
    callback: Optional[Callable] = None,
) -> OptimizationResult:
    x_eval_fns = normalize_eval_fns(cfg.x_eval_fns, default_name="eval")
    state = RiemannianState(x=x0.detach().clone(), step=0)
    trajectory = [state.x]
    metrics = []

    for _ in range(cfg.n_steps):
        state, metric = riemannian_step(
            state=state,
            objective_fn=objective_fn,
            grad_objective_fn=grad_objective_fn,
            projector=projector,
            cfg=cfg,
        )
        eval_extras = evaluate_iterate(x_eval_fns, state.x, prefix="x")
        if eval_extras:
            metric.extras = {**(metric.extras or {}), **eval_extras}
        trajectory.append(state.x)
        metrics.append(metric)
        if callback is not None:
            callback(state, metric)

    return OptimizationResult(
        final_x=state.x,
        trajectory=trajectory,
        metrics=metrics,
        extra=None,
    )


def run_riemannian_flow(
    x0: torch.Tensor,
    objective_fn: Callable,
    grad_objective_fn: Optional[Callable],
    projector,
    cfg: RiemannianConfig,
    callback: Optional[Callable] = None,
) -> OptimizationResult:
    x_eval_fns = normalize_eval_fns(cfg.x_eval_fns, default_name="eval")
    state = RiemannianState(x=x0.detach().clone(), step=0)
    trajectory = [state.x]
    metrics = []

    for _ in range(cfg.n_steps):
        state, metric = riemannian_flow_step(
            state=state,
            objective_fn=objective_fn,
            grad_objective_fn=grad_objective_fn,
            projector=projector,
            cfg=cfg,
        )
        eval_extras = evaluate_iterate(x_eval_fns, state.x, prefix="x")
        if eval_extras:
            metric.extras = {**(metric.extras or {}), **eval_extras}
        trajectory.append(state.x)
        metrics.append(metric)
        if callback is not None:
            callback(state, metric)

    return OptimizationResult(
        final_x=state.x,
        trajectory=trajectory,
        metrics=metrics,
        extra=None,
    )
