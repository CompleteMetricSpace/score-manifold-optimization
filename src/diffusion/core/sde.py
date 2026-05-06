#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDE solvers and utilities.

This module contains:
- SDE class: Stochastic differential equation representation
- sde_euler_maruyama_scalar: Euler-Maruyama solver
- build_reverse_flow: Construct reverse-time SDE from score
"""

import torch
import numpy as np
from typing import Callable, List, Tuple


def sde_euler_maruyama_scalar(
    f: Callable,
    g: Callable,
    n_grid_pts: int = 200,
    t_start: float = 0.0,
    t_end: float = 1.0,
    init_list: torch.Tensor = torch.Tensor([[0.0]]),
    corrector: Callable = lambda t, X: X,
    X: torch.Tensor = None,
    end_only: bool = False,
    seed: int = None,
    device: str = 'cpu',
    print_progress: bool = False,
    check_nan: bool = False,
    clamp_idx: List = [],
    clamp_val: List = []
) -> torch.Tensor:
    """
    Euler-Maruyama method for scalar-noise SDEs.

    Solves: dX = f(t, X)dt + g(t)dW

    Args:
        f: Drift coefficient (t, X) -> dX/dt
        g: Diffusion coefficient (t, X) -> scalar
        n_grid_pts: Number of time steps
        t_start: Start time
        t_end: End time
        init_list: Initial conditions (n_traj, *dims)
        corrector: Optional correction function
        X: Pre-allocated tensor (optional)
        end_only: If True, only return final state
        seed: Random seed
        device: Device for computation
        print_progress: Print progress
        check_nan: Check for NaN values
        clamp_idx: Indices to clamp
        clamp_val: Values to clamp to

    Returns:
        X: Final state (n_traj, *dims) if end_only
           or full trajectory (n_traj, *dims, n_grid_pts)
        times: Time grid (only if not end_only)
    """
    n_traj, *n_dim = init_list.shape
    assert clamp_idx is None or clamp_val is not None

    if X is None:
        if end_only:
            X = torch.zeros([n_traj, *n_dim]).to(device)
        else:
            X = torch.zeros([n_traj, *n_dim, n_grid_pts]).to(device)
    else:
        expected_shape = (n_traj, *n_dim) if end_only else (n_traj, *n_dim, n_grid_pts)
        assert X.shape == expected_shape

    times = torch.linspace(t_start, t_end, n_grid_pts).to(device)
    dim_size = [1 for _ in n_dim]
    h = (t_end - t_start) / (n_grid_pts - 1)
    sqrth = np.sqrt(h)

    # Initialize
    if end_only:
        X = init_list.to(device)
        for idx, val in zip(clamp_idx, clamp_val):
            X[tuple(idx)] = val
    else:
        X[..., 0] = init_list.to(device)
        for idx, val in zip(clamp_idx, clamp_val):
            X[tuple(idx) + (0,)] = val

    # Time stepping
    for j, t in enumerate(times[:-1]):
        if print_progress:
            print(f"{j/n_grid_pts:.2%} Done (t = {t:.4f})")

        tm = torch.Tensor([t]).expand(n_traj).to(device)  # 1d array

        if seed is not None:
            torch.manual_seed(seed)

        if end_only:
            X = (X + h * f(tm, X) +
                 sqrth * g(tm, X).view(n_traj, *dim_size) * torch.randn(n_traj, *n_dim).to(device))
            for idx, val in zip(clamp_idx, clamp_val):
                X[tuple(idx)] = val
        else:
            X[..., j+1] = (X[..., j] + h * f(tm, X[..., j]) +
                           sqrth * g(tm, X[..., j]).view(n_traj, *dim_size) *
                           torch.randn(n_traj, *n_dim).to(device))
            for idx, val in zip(clamp_idx, clamp_val):
                X[tuple(idx) + (j+1,)] = val

        if check_nan and X.isnan().any():
            raise Exception("NaN values in solution")

    if end_only:
        return X
    else:
        return X, times


def build_reverse_flow(
    f_forward: Callable,
    g_forward: Callable,
    score_forward: Callable,
    T: float,
    ndim: List[int],
    flow_type: str = "SDE"
) -> Tuple[Callable, Callable]:
    """
    Build reverse-time SDE/ODE from forward SDE and score.

    Forward SDE: dx = f(t,x)dt + g(t)dW
    Reverse SDE: dx = [-f(T-t,x) + g(T-t)²∇log p_t(x)]dt + g(T-t)dW̃
    Reverse ODE: dx = [-f(T-t,x) + ½g(T-t)²∇log p_t(x)]dt

    Args:
        f_forward: Forward drift
        g_forward: Forward diffusion
        score_forward: Score function ∇log p_t(x)
        T: Terminal time
        ndim: List of dimensions for view
        flow_type: "SDE" or "ODE"

    Returns:
        f_rev: Reverse drift
        g_rev: Reverse diffusion
    """
    if flow_type == "SDE":
        f_rev = lambda t, x: (-f_forward(T-t, x) +
                             g_forward(T-t).view(t.shape[0], *ndim)**2 * score_forward(T-t, x))
        g_rev = lambda t, x: g_forward(T-t)  # Added x parameter (ignored) for compatibility
    elif flow_type == "ODE":
        f_rev = lambda t, x: (-f_forward(T-t, x) +
                             0.5 * g_forward(T-t).view(t.shape[0], *ndim)**2 * score_forward(T-t, x))
        g_rev = lambda t, x: torch.zeros_like(t)  # Added x parameter (ignored) for compatibility
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}. Must be 'SDE' or 'ODE'")

    return f_rev, g_rev


class SDE:
    """
    Stochastic Differential Equation with scalar diffusion coefficient.

    Represents: dX = f(t, X)dt + g(t)dW
    where g is scalar (state-independent diagonal noise).

    Attributes:
        f_drift: Drift coefficient
        g_diff: Diffusion coefficient
        t_start: Start time
        t_end: End time
        dim: Shape of state space (tuple)
    """

    def __init__(
        self,
        f_drift: Callable,
        g_diff: Callable,
        t_start: float,
        t_end: float,
        dim: Tuple[int, ...],
        const_coeffs: bool = False
    ):
        """
        Initialize SDE.

        Args:
            f_drift: Drift function (t, x) -> dx/dt or x -> dx/dt if const_coeffs
            g_diff: Diffusion function (t) -> scalar or scalar if const_coeffs
            t_start: Start time
            t_end: End time
            dim: State space dimensions (tuple)
            const_coeffs: If True, coefficients are time-independent
        """
        if const_coeffs:
            self.f_drift = lambda t, x: f_drift(x)
            self.g_diff = lambda t: g_diff
        else:
            self.f_drift = f_drift
            self.g_diff = g_diff

        self.t_start = t_start
        self.t_end = t_end
        self.dim = dim

    def sample(
        self,
        x_init: torch.Tensor,
        step_size: float = 1e-2,
        end_only: bool = False,
        corrector: Callable = None,
        device: str = 'cpu',
        print_progress: bool = False,
        clamp_idx: List = [],
        clamp_val: List = []
    ) -> torch.Tensor:
        """
        Sample trajectory using Euler-Maruyama.

        Args:
            x_init: Initial state (n_traj, *dims)
            step_size: Time step size
            end_only: Return only final state
            corrector: Optional correction function
            device: Computation device
            print_progress: Print progress
            clamp_idx: Indices to clamp
            clamp_val: Values to clamp to

        Returns:
            Trajectory or final state
        """
        n_grid_pts = int((self.t_end - self.t_start) / step_size) + 1

        if corrector is None:
            corrector = lambda t, X: X

        with torch.no_grad():
            return sde_euler_maruyama_scalar(
                self.f_drift,
                self.g_diff,
                n_grid_pts=n_grid_pts,
                t_start=self.t_start,
                t_end=self.t_end,
                init_list=x_init,
                corrector=corrector,
                end_only=end_only,
                device=device,
                print_progress=print_progress,
                clamp_idx=clamp_idx,
                clamp_val=clamp_val
            )

    @staticmethod
    def concat_sde(sde1: 'SDE', sde2: 'SDE') -> 'SDE':
        """
        Concatenate two SDEs sequentially.

        Args:
            sde1: First SDE
            sde2: Second SDE

        Returns:
            Concatenated SDE
        """
        assert sde1.dim == sde2.dim, "SDEs must have equal state dimensions"

        f_concat = lambda t, x: torch.where(
            (t <= sde1.t_end).view(t.shape[0], *[1 for _ in sde1.dim]).expand(t.shape[0], *sde1.dim),
            sde1.f_drift(t.clip(sde1.t_start, sde1.t_end), x),
            sde2.f_drift((t - sde1.t_end + sde2.t_start).clip(sde2.t_start, sde2.t_end), x)
        )

        g_concat = lambda t: torch.where(
            t < sde1.t_end,
            sde1.g_diff(t.clip(sde1.t_start, sde1.t_end)),
            sde2.g_diff((t - sde1.t_end + sde2.t_start).clip(sde2.t_start, sde2.t_end))
        )

        return SDE(f_concat, g_concat, sde1.t_start,
                   sde2.t_end - sde2.t_start + sde1.t_end, sde1.dim)

    def concat(self, sde: 'SDE') -> 'SDE':
        """Concatenate this SDE with another."""
        return SDE.concat_sde(self, sde)

    def truncate(self, t_min: float, t_max: float) -> 'SDE':
        """
        Truncate SDE to time interval [t_min, t_max].

        Args:
            t_min: New start time
            t_max: New end time

        Returns:
            Truncated SDE
        """
        return SDE(self.f_drift, self.g_diff, t_min, t_max, self.dim)
