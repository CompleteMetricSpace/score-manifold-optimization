#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass, field
from math import prod
from typing import Callable, Literal, Optional, Tuple

import torch


try:
    from torch.func import jvp as torch_jvp
except Exception:
    from functorch import jvp as torch_jvp

from diffusion.core import DiffusionProcess, sde_euler_maruyama_scalar

from .projector_base import Projector
from .types import ScoreTangentConfig
from .tangent import (
    truncate_small_singular_values,
    normal_tangent_decomposition,
    project_with_basis,
)


_MIN_MEAN_COEF = 1e-12


def _require_diffusion_process(diffusion: object, *, name: str = "diffusion") -> DiffusionProcess:
    if not isinstance(diffusion, DiffusionProcess):
        raise TypeError(f"{name} must be a DiffusionProcess, got {type(diffusion).__name__}.")
    return diffusion


def _mean_coef_var_from_diffusion(
    diffusion: DiffusionProcess,
    t: torch.Tensor,
    data_dims,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dims = tuple(data_dims)
    x_ref = torch.ones((t.shape[0], *dims), device=t.device, dtype=t.dtype)
    mean_ref, var = diffusion.conditional_mean_var(t, x_ref)

    if mean_ref.shape != x_ref.shape:
        raise ValueError(
            "conditional_mean_var(t, x0) must return mean with the same shape as x0, "
            f"got mean.shape={tuple(mean_ref.shape)} and x0.shape={tuple(x_ref.shape)}."
        )
    if var.shape != t.shape:
        raise ValueError(
            "conditional_mean_var(t, x0) must return scalar variance per batch element, "
            f"got var.shape={tuple(var.shape)} and t.shape={tuple(t.shape)}."
        )

    mean_ref_flat = mean_ref.reshape(mean_ref.shape[0], -1)
    mean_coef = mean_ref_flat[:, 0]
    expected_mean = mean_coef.unsqueeze(1).expand_as(mean_ref_flat)

    return mean_coef, var


@dataclass
class ScoreProjector(Projector):
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    mean_coef_fn: Callable[[torch.Tensor], torch.Tensor]
    var_fn: Callable[[torch.Tensor], torch.Tensor]
    score_time: float
    data_ndim: int
    n_proj: int = 1
    tangent_cfg: ScoreTangentConfig = field(default_factory=ScoreTangentConfig)

    def __post_init__(self):
        if int(self.n_proj) < 1:
            raise ValueError(f"n_proj must be >= 1, got {self.n_proj}.")
        if float(self.score_time) < 0.0:
            raise ValueError(f"score_time must be >= 0, got {self.score_time}.")

        base_methods_param = ["jvp", "jvp_weeny"]
        base_methods_static = ["vjp", "jacobian", "jacobian_truncated", "empirical_basis", "jvp_separate_time"]
        method = self.tangent_cfg.method
        if method in base_methods_static:
            self.base_method = method
            self.method_params = {}
        elif method in base_methods_param:
            self.base_method = method
            self.method_params = {"n": 1} #Both "jvp", "jvp_weeny"
            #Add some special handling
        else: #Add some special handling
            m_jvp = re.fullmatch(r"jvp_(\d+)", method) #Parse jvp_n method
            m_jvp_weeny = re.fullmatch(r"jvp_weeny_(\d+)", method) #Parse jvp_null n method
            if m_jvp:
                self.base_method = "jvp"
                self.method_params = {"n": int(m_jvp.group(1))}
            elif m_jvp_weeny:
                self.base_method = "jvp_weeny"
                self.method_params = {"n": int(m_jvp_weeny.group(1))}
            else:
                raise ValueError(f"Unknown tangent method {method}")

    def _project_impl(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        t = self._time_batch(x)
        x_proj = x
        mean_coef = self.mean_coef_fn(t)
        if mean_coef.shape != t.shape:
            raise ValueError(
                f"mean_coef_fn(t) must return shape {tuple(t.shape)}, got {tuple(mean_coef.shape)}."
            )
        mean_sign = torch.where(mean_coef >= 0, torch.ones_like(mean_coef), -torch.ones_like(mean_coef))
        mean_coef_safe = torch.where(
            mean_coef.abs() < _MIN_MEAN_COEF,
            mean_sign * _MIN_MEAN_COEF,
            mean_coef,
        )
        var = self.var_fn(t)
        if var.shape != t.shape:
            raise ValueError(f"var_fn(t) must return shape {tuple(t.shape)}, got {tuple(var.shape)}.")
        mean_coef_view = mean_coef_safe.view(x.shape[0], *([1] * self.data_ndim))
        var_view = var.view(x.shape[0], *([1] * self.data_ndim))
        for _ in range(int(self.n_proj)):
            score = self.score_fn(t, x_proj)
            x0_hat = (x_proj + var_view * score) / mean_coef_view
            x_proj = x_proj + float(scale) * (x0_hat - x_proj)
        return x_proj

    def _time_batch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), float(self.score_time), device=x.device, dtype=x.dtype)

    def project(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Tweedie projection:
            x <- x + scale * (E[x0 | x_t=x] - x)
        Optional kwargs are kept for backward compatibility with existing analysis code.
        """
        return self._project_impl(x, scale=scale)

    def _project_null(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Tweedie projection:
            x <- x + scale * (E[x0 | x_t=x] - x)
        Optional kwargs are kept for backward compatibility with existing analysis code.
        """
        t = self._time_batch(x)
        mean_coef = self.mean_coef_fn(t)
        if mean_coef.shape != t.shape:
            raise ValueError(
                f"mean_coef_fn(t) must return shape {tuple(t.shape)}, got {tuple(mean_coef.shape)}."
            )
        mean_sign = torch.where(mean_coef >= 0, torch.ones_like(mean_coef), -torch.ones_like(mean_coef))
        mean_coef_safe = torch.where(
            mean_coef.abs() < _MIN_MEAN_COEF,
            mean_sign * _MIN_MEAN_COEF,
            mean_coef,
        )
        var = self.var_fn(t)
        if var.shape != t.shape:
            raise ValueError(f"var_fn(t) must return shape {tuple(t.shape)}, got {tuple(var.shape)}.")
        mean_coef_view = mean_coef_safe.view(x.shape[0], *([1] * self.data_ndim))
        var_view = var.view(x.shape[0], *([1] * self.data_ndim))
        score = self.score_fn(t, x)
        x0_hat = (x + var_view * score) / mean_coef_view
        return float(scale) * (x0_hat - x)

    def _project_tangent_jvp(self, x: torch.Tensor, v: torch.Tensor, n: int = 1) -> torch.Tensor:
        v_tan = v
        for _ in range(n):
            _, v_tan = torch_jvp(self.project, (x,), (v_tan,))
        return v_tan

    def _project_tangent_jvp_weeny(self, x: torch.Tensor, v: torch.Tensor, n: int = 1) -> torch.Tensor:
        v_tan = v
        for _ in range(n):
           v_tan = self._mc_weeny(x, v_tan)
        return v_tan
    
    def _mc_weeny(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        _, v1 = torch_jvp(self.project, (x,), (v,))
        _, v2 = torch_jvp(self.project, (x,), (v1,))
        _, v3 = torch_jvp(self.project, (x,), (v2,))
        return 3*v2 - 2*v3 #McWeeny spectral sharpening


    def pushforward_jvp(self, x: torch.Tensor, v: torch.Tensor):
        """
        Backward-compatible helper used by legacy scripts.
        """
        return torch_jvp(self.project, (x,), (v,))

    def _project_tangent_jacobian(self, x: torch.Tensor, v: torch.Tensor, truncated: bool = False) -> torch.Tensor:
        batch = x.shape[0]
        out = []
        flat_dim = int(prod(x.shape[1:]))
        if self.tangent_cfg.max_jacobian_dim is not None and flat_dim > self.tangent_cfg.max_jacobian_dim:
            raise ValueError(
                f"Jacobian tangent requested with flat_dim={flat_dim}, "
                f"exceeding max_jacobian_dim={self.tangent_cfg.max_jacobian_dim}."
            )

        for b in range(batch):
            xb = x[b:b + 1].detach().clone().requires_grad_(True)
            vb = v[b:b + 1]
            yb = self.project(xb)
            yb_flat = yb.flatten(start_dim=1)[0]
            jac_cols = []
            for i in range(yb_flat.numel()):
                grad = torch.autograd.grad(
                    yb_flat[i],
                    xb,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                jac_cols.append(grad.flatten(start_dim=1)[0])
            jac = torch.stack(jac_cols, dim=0)  # (d_out, d_in)
            if truncated:
                jac = truncate_small_singular_values(jac, rtol=self.tangent_cfg.truncation_rtol)
            vb_flat = vb.flatten(start_dim=1)[0]
            out_flat = jac @ vb_flat
            out.append(out_flat.unflatten(0, x.shape[1:]).unsqueeze(0))
        return torch.cat(out, dim=0)

    def _project_tangent_empirical_basis(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        _, t_basis_flat = normal_tangent_decomposition(
            x=x,
            proj=self.project,
            n_samples=self.tangent_cfg.n_samples,
            eps_samples=self.tangent_cfg.eps_samples,
            remove_first_sv=self.tangent_cfg.remove_first_sv,
        )
        return project_with_basis(v, t_basis_flat)

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        method = self.base_method
        method_params = self.method_params
        if method == "vjp":
            raise NotImplementedError(
                "ScoreProjector tangent method 'vjp' is only supported with tangent_mode='project-at-start'. "
                "project_tangent(x, v) always denotes Jv, not J^T v."
            )
        if method == "jvp":
            n = method_params["n"]
            return self._project_tangent_jvp(x, v, n)
        if method == "jvp_weeny":
            n = method_params["n"]
            return self._project_tangent_jvp_weeny(x, v, n)
        if method == "jacobian":
            return self._project_tangent_jacobian(x, v, truncated=False)
        if method == "jacobian_truncated":
            return self._project_tangent_jacobian(x, v, truncated=True)
        if method == "empirical_basis":
            if x.shape[0] > 1:
                raise NotImplementedError(
                    "empirical_basis tangent projection currently supports batch size 1 only."
                )
            return self._project_tangent_empirical_basis(x, v)
        if method == "jvp_separate_time":
            t = torch.full((x.shape[0],), float(self.tangent_cfg.tangent_separate_time), device=x.device, dtype=x.dtype)
            mean_coef = self.mean_coef_fn(t)
            if mean_coef.shape != t.shape:
                raise ValueError(
                    f"mean_coef_fn(t) must return shape {tuple(t.shape)}, got {tuple(mean_coef.shape)}."
                )
            if torch.any(mean_coef.abs() < _MIN_MEAN_COEF):
                raise ValueError(
                    "mean_coef_fn(t) returned values too close to zero for stable Tweedie inversion."
                )
            var = self.var_fn(t)
            if var.shape != t.shape:
                raise ValueError(f"var_fn(t) must return shape {tuple(t.shape)}, got {tuple(var.shape)}.")
            mean_coef_view = mean_coef.view(x.shape[0], *([1] * self.data_ndim))
            var_view = var.view(x.shape[0], *([1] * self.data_ndim))

            def _project_step(p: torch.Tensor) -> torch.Tensor:
                score = self.score_fn(t, p)
                x0_hat = (p + var_view * score) / mean_coef_view
                return p + (x0_hat - p)

            _, v_tan = torch_jvp(_project_step, (x,), (v,))
            return v_tan
        raise ValueError(f"Unknown tangent method: {method}")

def build_score_projector(
    score_model,
    diffusion: DiffusionProcess,
    score_time: float,
    data_dims,
    n_proj: int = 1,
    tangent_cfg: Optional[ScoreTangentConfig] = None,
) -> ScoreProjector:
    diffusion = _require_diffusion_process(diffusion)
    if float(score_time) > float(diffusion.T):
        raise ValueError(f"score_time={score_time} exceeds diffusion.T={diffusion.T}.")

    data_dims = tuple(data_dims)
    data_ndim = len(data_dims)

    def score_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return score_model(t, x)

    def mean_coef_fn(t: torch.Tensor) -> torch.Tensor:
        mean_coef, _ = _mean_coef_var_from_diffusion(diffusion, t, data_dims)
        return mean_coef

    def var_fn(t: torch.Tensor) -> torch.Tensor:
        _, var = _mean_coef_var_from_diffusion(diffusion, t, data_dims)
        return var

    return ScoreProjector(
        score_fn=score_fn,
        mean_coef_fn=mean_coef_fn,
        var_fn=var_fn,
        score_time=score_time,
        data_ndim=data_ndim,
        n_proj=n_proj,
        tangent_cfg=tangent_cfg if tangent_cfg is not None else ScoreTangentConfig(),
    )


@dataclass
class ScoreDiffuserProjector(Projector):
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    diffusion: DiffusionProcess
    score_time: float
    data_ndim: int
    n_proj: int = 1
    flow_type: Literal["SDE", "ODE"] = "ODE"
    tangent_cfg: ScoreTangentConfig = field(default_factory=ScoreTangentConfig)

    def __post_init__(self):
        _require_diffusion_process(self.diffusion)
        if int(self.n_proj) < 1:
            raise ValueError(f"n_proj must be >= 1, got {self.n_proj}.")
        if float(self.score_time) < 0:
            raise ValueError(f"score_time must be >= 0, got {self.score_time}.")
        if float(self.score_time) > float(self.diffusion.T):
            raise ValueError(
                f"score_time={self.score_time} exceeds diffusion.T={self.diffusion.T}."
            )
        flow_type = str(self.flow_type).upper()
        if flow_type not in {"SDE", "ODE"}:
            raise ValueError(f"Unknown flow_type: {self.flow_type}. Must be 'SDE' or 'ODE'.")
        self.flow_type = flow_type

    def _time_batch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), float(self.score_time), device=x.device, dtype=x.dtype)

    def project(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Reverse-flow projection over [0, score_time] (forward-time interval).

        This integrates reverse SDE/ODE from t_rev = T - score_time to t_rev = T.
        """
        if float(self.score_time) == 0.0:
            return x

        t_rev_start = float(self.diffusion.T) - float(self.score_time)
        t_rev_end = float(self.diffusion.T)
        n_grid_pts = int(self.n_proj) + 1

        scaled_score_fn = lambda t, y: float(scale) * self.score_fn(t, y)
        reverse_sde = self.diffusion.get_reverse_sde(scaled_score_fn, flow_type=self.flow_type)

        x_proj = sde_euler_maruyama_scalar(
            f=reverse_sde.f_drift,
            g=reverse_sde.g_diff,
            n_grid_pts=n_grid_pts,
            t_start=t_rev_start,
            t_end=t_rev_end,
            init_list=x,
            end_only=True,
            device=x.device,
        )
        return x_proj

    def _project_tangent_jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        _, v_tan = torch_jvp(self.project, (x,), (v,))
        return v_tan

    def pushforward_jvp(self, x: torch.Tensor, v: torch.Tensor):
        """
        Backward-compatible helper used by legacy scripts.
        """
        return torch_jvp(self.project, (x,), (v,))

    def _project_tangent_jacobian(self, x: torch.Tensor, v: torch.Tensor, truncated: bool = False) -> torch.Tensor:
        batch = x.shape[0]
        out = []
        flat_dim = int(prod(x.shape[1:]))
        if self.tangent_cfg.max_jacobian_dim is not None and flat_dim > self.tangent_cfg.max_jacobian_dim:
            raise ValueError(
                f"Jacobian tangent requested with flat_dim={flat_dim}, "
                f"exceeding max_jacobian_dim={self.tangent_cfg.max_jacobian_dim}."
            )

        for b in range(batch):
            xb = x[b:b + 1].detach().clone().requires_grad_(True)
            vb = v[b:b + 1]
            yb = self.project(xb)
            yb_flat = yb.flatten(start_dim=1)[0]
            jac_cols = []
            for i in range(yb_flat.numel()):
                grad = torch.autograd.grad(
                    yb_flat[i],
                    xb,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                jac_cols.append(grad.flatten(start_dim=1)[0])
            jac = torch.stack(jac_cols, dim=0)  # (d_out, d_in)
            if truncated:
                jac = truncate_small_singular_values(jac, rtol=self.tangent_cfg.truncation_rtol)
            vb_flat = vb.flatten(start_dim=1)[0]
            out_flat = jac @ vb_flat
            out.append(out_flat.unflatten(0, x.shape[1:]).unsqueeze(0))
        return torch.cat(out, dim=0)

    def _project_tangent_empirical_basis(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        _, t_basis_flat = normal_tangent_decomposition(
            x=x,
            proj=self.project,
            n_samples=self.tangent_cfg.n_samples,
            eps_samples=self.tangent_cfg.eps_samples,
            remove_first_sv=self.tangent_cfg.remove_first_sv,
        )
        return project_with_basis(v, t_basis_flat)

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        method = self.tangent_cfg.method
        if method == "vjp":
            raise NotImplementedError(
                "ScoreDiffuserProjector tangent method 'vjp' is only supported with tangent_mode='project-at-start'. "
                "project_tangent(x, v) always denotes Jv, not J^T v."
            )
        if method == "jvp":
            return self._project_tangent_jvp(x, v)
        if method == "jacobian":
            return self._project_tangent_jacobian(x, v, truncated=False)
        if method == "jacobian_truncated":
            return self._project_tangent_jacobian(x, v, truncated=True)
        if method == "empirical_basis":
            if x.shape[0] > 1:
                raise NotImplementedError(
                    "empirical_basis tangent projection currently supports batch size 1 only."
                )
            return self._project_tangent_empirical_basis(x, v)
        if method == "jvp_separate_time":
            t = torch.full((x.shape[0],), float(self.tangent_cfg.tangent_separate_time), device=x.device, dtype=x.dtype)
            sigma_sq_view = self._sigma_sq_view(t, x)
            _, v_tan = torch_jvp(lambda p: p + sigma_sq_view * self.score_fn(t, p), (x,), (v,))
            return v_tan
        raise ValueError(f"Unknown tangent method: {method}")

def build_score_diffuser_projector(
    score_model,
    diffusion: DiffusionProcess,
    score_time: float,
    data_dims,
    n_proj: int = 1,
    tangent_cfg: Optional[ScoreTangentConfig] = None,
    flow_type: Literal["SDE", "ODE"] = "ODE",
) -> ScoreDiffuserProjector:
    diffusion = _require_diffusion_process(diffusion)
    data_ndim = len(tuple(data_dims))

    def score_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return score_model(t, x)

    return ScoreDiffuserProjector(
        score_fn=score_fn,
        diffusion=diffusion,
        score_time=score_time,
        data_ndim=data_ndim,
        n_proj=n_proj,
        flow_type=flow_type,
        tangent_cfg=tangent_cfg if tangent_cfg is not None else ScoreTangentConfig(),
    )
