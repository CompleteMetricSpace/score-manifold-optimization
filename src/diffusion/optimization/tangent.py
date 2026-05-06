#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import prod
from typing import Callable, Optional, Tuple
import torch

try:
    from torch.func import jvp as torch_jvp
except Exception:
    from functorch import jvp as torch_jvp


def truncate_small_singular_values(mat: torch.Tensor, rtol: float = 1e-6) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    if s.numel() == 0:
        return mat
    keep = s > (rtol * s.max())
    if keep.sum() == 0:
        return torch.zeros_like(mat)
    s_trunc = torch.zeros_like(s)
    s_trunc[keep] = s[keep]
    return (u * s_trunc.unsqueeze(0)) @ vh


def project_with_basis(v: torch.Tensor, t_basis_flat: torch.Tensor) -> torch.Tensor:
    v_flat = v.flatten(start_dim=1)
    proj_flat = (v_flat @ t_basis_flat) @ t_basis_flat.T
    return proj_flat.unflatten(1, v.shape[1:])


def normal_tangent_decomposition(
    x: torch.Tensor,
    proj: Callable[[torch.Tensor], torch.Tensor],
    n_samples: Optional[int] = None,
    eps_samples: float = 0.05,
    remove_first_sv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data_dim = tuple(x.shape[1:])
    if n_samples is None:
        n_samples = 2 * prod(data_dim)
    z_noise = torch.randn(n_samples, *data_dim, device=x.device, dtype=x.dtype)
    z_norm = z_noise.flatten(start_dim=1).norm(dim=1, keepdim=True).clamp(min=1e-12)
    z_noise = eps_samples * z_noise / z_norm.view(n_samples, *([1] * len(data_dim)))
    x0 = x[0:1]
    n_space = (x0 + z_noise) - proj(x0 + z_noise)
    n_space_flat = n_space.flatten(start_dim=1)
    u_pca, s_pca, _ = torch.linalg.svd(n_space_flat.T, full_matrices=True)

    if s_pca.numel() <= 1:
        max_idx = 1
    else:
        s_delta = s_pca[:-1] - s_pca[1:]
        if remove_first_sv and s_delta.numel() > 0:
            s_delta[0] = 0
        max_idx = int(s_delta.argmax().item()) + 1

    n_basis_flat = u_pca[:, :max_idx]
    t_basis_flat = u_pca[:, max_idx:]
    return n_basis_flat, t_basis_flat


def tangent_projection(
    g: torch.Tensor,
    x: torch.Tensor,
    proj: Callable[[torch.Tensor], torch.Tensor],
    jacobian: bool = True,
    trunc_jacobian: bool = False,
    n_samples: Optional[int] = None,
    eps_samples: float = 0.05,
):
    if jacobian:
        px, pg = torch_jvp(proj, (x,), (g,))
        if trunc_jacobian:
            g_flat = g.flatten(start_dim=1)
            pg_flat = pg.flatten(start_dim=1)
            if g_flat.shape[0] == 1:
                denom = (g_flat @ g_flat.T).clamp(min=1e-12)
                a = (pg_flat.T @ g_flat) / denom
                a = truncate_small_singular_values(a)
                pg = (g_flat @ a.T).unflatten(1, g.shape[1:])
        return pg, px

    if trunc_jacobian:
        _, t_basis_flat = normal_tangent_decomposition(
            x, proj, n_samples=n_samples, eps_samples=eps_samples
        )
        pg = project_with_basis(g, t_basis_flat)
        return pg, None

    return g, None


def second_fundamental_form_from_projector(
    projector,
    p: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    project_to_normal: bool = False,
    ensure_tangent_inputs: bool = False,
    project_point: bool = False,
    use_project: bool = False,
) -> torch.Tensor:
    """
    Compute the second fundamental form II(u, v) from a closest-point projector.

    Geometry/sign convention:
        II(u, v) = -D(dπ)_p[u] v = -D^2π(p)[u, v],
    where π is the nearest-point projection map and p is on the manifold.

    By default this uses one differentiation of ``project_tangent``:
        II(u, v) = - d/dt|_{t=0} project_tangent(p + t u, v).
    If ``use_project=True``, it uses a nested-JVP Hessian action of ``project``.
    """
    if not isinstance(p, torch.Tensor) or not isinstance(u, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError("p, u, v must be torch.Tensor.")
    if p.shape != u.shape or p.shape != v.shape:
        raise ValueError(
            f"p, u, v must have identical shapes; got p={tuple(p.shape)}, "
            f"u={tuple(u.shape)}, v={tuple(v.shape)}."
        )
    if p.numel() == 0:
        raise ValueError("p must be non-empty.")

    p_eval = projector.project(p) if project_point else p
    u_eval = u
    v_eval = v

    if ensure_tangent_inputs:
        u_eval = projector.project_tangent(p_eval, u_eval)
        v_eval = projector.project_tangent(p_eval, v_eval)

    if use_project:
        def dproj_times_v(x: torch.Tensor) -> torch.Tensor:
            _, jv = torch_jvp(projector.project, (x,), (v_eval,))
            return jv

        _, h_uv = torch_jvp(dproj_times_v, (p_eval,), (u_eval,))
        ii = -h_uv
    else:
        def tangent_map(x: torch.Tensor) -> torch.Tensor:
            return projector.project_tangent(x, v_eval)

        _, dtangent = torch_jvp(tangent_map, (p_eval,), (u_eval,))
        ii = -dtangent

    if project_to_normal:
        ii = ii - projector.project_tangent(p_eval, ii)

    return ii


def estimate_mean_curvature(
    projector,
    p: torch.Tensor,
    codim: int,
    eps: float,
    num_samples: int,
) -> torch.Tensor:
    """
    Estimate mean curvature vector H(p) using only Jacobian-vector products.

    This implements the basis-free symmetric Monte Carlo estimator

        H(p) ≈ k E_{n,v}[ <v, (Dπ(p+εn)-Dπ(p-εn))v> / (2ε) * n ],

    where k=codim(M), n is a random unit normal direction, and v is a random
    tangent probe. We only use ``project_tangent(x, v)=Dπ(x)v`` and never build
    full Jacobians/Hessians or matrix inverses.
    """
    if not isinstance(p, torch.Tensor):
        raise TypeError("p must be torch.Tensor.")
    if p.ndim < 2:
        raise ValueError(f"p must include a batch dimension and ambient coordinates, got shape {tuple(p.shape)}.")
    if p.numel() == 0:
        raise ValueError("p must be non-empty.")
    if int(codim) <= 0:
        raise ValueError(f"codim must be > 0, got {codim}.")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}.")

    p_eval = p
    batch = p_eval.shape[0]
    data_shape = tuple(p_eval.shape[1:])
    tiny = 1e-12
    max_resample_attempts = 8

    acc = torch.zeros_like(p_eval)
    valid_counts = torch.zeros(batch, dtype=torch.int64, device=p_eval.device)
    codim_float = float(int(codim))
    eps_float = float(eps)
    expand = (1,) * len(data_shape)

    for _ in range(int(num_samples)):
        sample_contrib = torch.zeros_like(p_eval)
        sample_valid = torch.zeros(batch, dtype=torch.bool, device=p_eval.device)

        for _ in range(max_resample_attempts):
            pending = ~sample_valid
            if not bool(pending.any()):
                break

            pending_idx = torch.nonzero(pending, as_tuple=False).flatten()
            p_pending = p_eval[pending_idx]
            g = torch.randn_like(p_pending)
            h = torch.randn_like(p_pending)

            # Normal probe: n = P_N g / ||P_N g||, with P_N = I - P_T.
            pt_g = projector.project_tangent(p_pending, g)
            pn_g = g - pt_g
            pn_norm = pn_g.flatten(start_dim=1).norm(dim=1)
            keep_n = pn_norm > tiny
            if not bool(keep_n.any()):
                continue

            keep_idx = pending_idx[keep_n]
            p_keep = p_pending[keep_n]
            n = pn_g[keep_n] / pn_norm[keep_n].view(-1, *expand)

            # Tangent probe: v = P_T h.
            v = projector.project_tangent(p_keep, h[keep_n])
            v_norm = v.flatten(start_dim=1).norm(dim=1)
            keep_v = v_norm > tiny
            if not bool(keep_v.any()):
                continue

            final_idx = keep_idx[keep_v]
            p_final = p_keep[keep_v]
            n_final = n[keep_v]
            v_final = v[keep_v]

            # Symmetric finite difference of Dπ along normal direction n.
            dplus_v = projector.project_tangent(p_final + eps_float * n_final, v_final)
            dminus_v = projector.project_tangent(p_final - eps_float * n_final, v_final)
            fd_v = (dplus_v - dminus_v) / (2.0 * eps_float)

            scalar = (v_final.flatten(start_dim=1) * fd_v.flatten(start_dim=1)).sum(dim=1)
            # Our curvature convention is II(u,v) = -D(dπ)[u]v, so the MC trace
            # estimator carries a leading minus sign.
            contrib = -codim_float * scalar.view(-1, *expand) * n_final

            sample_contrib[final_idx] = contrib
            sample_valid[final_idx] = True

        if bool(sample_valid.any()):
            acc[sample_valid] += sample_contrib[sample_valid]
            valid_counts[sample_valid] += 1

    out = torch.zeros_like(p_eval)
    has_valid = valid_counts > 0
    if bool(has_valid.any()):
        out[has_valid] = acc[has_valid] / valid_counts[has_valid].to(p_eval.dtype).view(-1, *expand)
    return out.to(device=p.device, dtype=p.dtype)


def mean_curvature_from_projector(
    projector,
    p: torch.Tensor,
    *,
    mode: str = "exact",
    n_probes: int = 16,
    generator: Optional[torch.Generator] = None,
    codim: Optional[int] = None,
    eps: float = 1e-3,
    num_samples: int = 64,
    project_point: bool = False,
    use_project: bool = False,
    project_to_normal: bool = True,
    ensure_tangent_inputs: bool = False,
) -> torch.Tensor:
    """
    Compute mean curvature vector from a closest-point projector.

    Uses the trace convention
        H(p) = tr(II_p) = sum_k II_p(e_k, e_k),
    where II follows ``second_fundamental_form_from_projector``:
        II(u, v) = -D(dπ)_p[u]v = -D^2π(p)[u, v].

    ``mode="exact"`` computes the ambient-basis trace with tangent-projected basis
    vectors. ``mode="hutchinson"`` uses a stochastic trace estimator with tangent-
    projected Gaussian probes. ``mode="basis_free_mc"`` uses a basis-free symmetric
    Monte Carlo estimator based only on ``project_tangent`` finite differences.
    """
    if not isinstance(p, torch.Tensor):
        raise TypeError("p must be torch.Tensor.")
    if p.ndim < 2:
        raise ValueError(f"p must include a batch dimension and ambient coordinates, got shape {tuple(p.shape)}.")
    if p.numel() == 0:
        raise ValueError("p must be non-empty.")

    mode_key = str(mode).lower()
    if mode_key not in {"exact", "hutchinson", "basis_free_mc"}:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of: exact, hutchinson, basis_free_mc.")
    if mode_key == "hutchinson" and int(n_probes) <= 0:
        raise ValueError(f"n_probes must be > 0 for hutchinson mode, got {n_probes}.")
    if mode_key == "basis_free_mc":
        if codim is None:
            raise ValueError("codim must be provided for mode='basis_free_mc'.")
        return estimate_mean_curvature(
            projector=projector,
            p=projector.project(p) if project_point else p,
            codim=int(codim),
            eps=float(eps),
            num_samples=int(num_samples),
        )

    p_eval = projector.project(p) if project_point else p
    batch = p_eval.shape[0]
    ambient_shape = tuple(p_eval.shape[1:])
    ambient_dim = int(prod(ambient_shape))
    if ambient_dim <= 0:
        raise ValueError("Ambient dimension must be > 0.")

    if mode_key == "exact":
        basis = torch.eye(ambient_dim, device=p_eval.device, dtype=p_eval.dtype).reshape(
            ambient_dim, *ambient_shape
        )
        p_rep = p_eval.repeat_interleave(ambient_dim, dim=0).contiguous()
        e_rep = basis.repeat(batch, *([1] * len(ambient_shape))).contiguous()
        w_rep = projector.project_tangent(p_rep, e_rep)
        ii_rep = second_fundamental_form_from_projector(
            projector=projector,
            p=p_rep,
            u=w_rep,
            v=w_rep,
            project_to_normal=project_to_normal,
            ensure_tangent_inputs=ensure_tangent_inputs,
            project_point=False,
            use_project=use_project,
        )
        h = ii_rep.reshape(batch, ambient_dim, *ambient_shape).sum(dim=1)
        return h.to(device=p.device, dtype=p.dtype)

    n = int(n_probes)
    p_rep = p_eval.repeat_interleave(n, dim=0).contiguous()
    xi = torch.randn(
        batch, n, *ambient_shape,
        device=p_eval.device,
        dtype=p_eval.dtype,
        generator=generator,
    ).reshape(batch * n, *ambient_shape)
    w_rep = projector.project_tangent(p_rep, xi)
    ii_rep = second_fundamental_form_from_projector(
        projector=projector,
        p=p_rep,
        u=w_rep,
        v=w_rep,
        project_to_normal=project_to_normal,
        ensure_tangent_inputs=ensure_tangent_inputs,
        project_point=False,
        use_project=use_project,
    )
    h = ii_rep.reshape(batch, n, *ambient_shape).mean(dim=1)
    return h.to(device=p.device, dtype=p.dtype)
