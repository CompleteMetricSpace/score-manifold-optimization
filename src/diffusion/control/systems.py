#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal control-system definitions retained for paper workflows."""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class IOControlSystem(ABC):
    """Base interface for control systems with input/output trajectories."""

    def split_input_output(self, x: torch.Tensor):
        input_dim = int(self.get_input_dim())
        return x[..., :input_dim], x[..., input_dim:]

    @abstractmethod
    def get_input_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_output_dim(self) -> int:
        raise NotImplementedError

    def get_state_init_window_length(self):
        return None


class ContinuousDynamicalControlSystem(IOControlSystem):
    """Base class for continuous-time systems x_dot = f(x, u)."""

    @abstractmethod
    def _dyn(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_state_dim(self) -> int:
        raise NotImplementedError

    def get_output(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_output_dim(self) -> int:
        return int(self.get_state_dim())

    def is_state_output(self) -> bool:
        return int(self.get_state_dim()) == int(self.get_output_dim())

    def discretize(self, dT: float, Ts: float | None = None, method: str = "rk4"):
        if Ts is None:
            Ts = dT
        return _DiscretizedContinuousSystem(self, dT=dT, Ts=Ts, method=method)


class DynamicalControlSystem(IOControlSystem):
    """Base class for discrete-time systems x_{k+1} = F(x_k, u_k)."""

    @abstractmethod
    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def simulate(self, x0: torch.Tensor, u: torch.Tensor, print_progress: bool = False) -> torch.Tensor:
        n_batch, n_steps, _ = u.shape
        state_dim = x0.shape[-1]
        x = torch.empty(n_batch, n_steps, state_dim, device=u.device, dtype=u.dtype)
        x[:, 0] = x0
        for j in range(n_steps - 1):
            x[:, j + 1] = self.dynamics(x[:, j], u[:, j])
            if print_progress:
                print(f"{j / max(1, n_steps - 1):.3f} Done")
        return x

    def simulate_out(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.get_output(self.simulate(x0, u))

    def simulate_out_pair(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        y = self.simulate_out(x0, u)
        return torch.cat([u, y], dim=-1)

    def get_output(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @abstractmethod
    def get_state_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        return int(self.get_state_dim())

    def is_state_output(self) -> bool:
        return int(self.get_state_dim()) == int(self.get_output_dim())

    def to_constraint(self, horizon: int):
        from diffusion.control.trajectory import DynamicsConstraint

        return DynamicsConstraint.from_system(self, horizon)

    def get_space(self, horizon: int):
        from diffusion.spaces.euclidean import TrajectorySpace

        return TrajectorySpace(
            horizon=horizon,
            input_dim=int(self.get_input_dim()),
            output_dim=int(self.get_output_dim()),
        )


class _DiscretizedContinuousSystem(DynamicalControlSystem):
    """Thin discrete-time wrapper around a continuous-time system."""

    def __init__(
        self,
        continuous_system: ContinuousDynamicalControlSystem,
        dT: float,
        Ts: float,
        method: str = "rk4",
    ):
        self.continuous_system = continuous_system
        self.dT = float(dT)
        self.Ts = float(Ts)
        self.method = str(method).lower()

        if self.dT <= 0.0:
            raise ValueError(f"dT must be positive, got {self.dT}.")
        if self.Ts <= 0.0:
            raise ValueError(f"Ts must be positive, got {self.Ts}.")

        ratio = self.Ts / self.dT
        n_substeps = int(round(ratio))
        tolerance = 1e-9 * max(1.0, abs(ratio))
        if n_substeps < 1 or abs(ratio - n_substeps) > tolerance:
            raise ValueError(
                "Ts must be an integer multiple of dT (Ts = n*dT, n>=1). "
                f"Got Ts={self.Ts}, dT={self.dT}."
            )
        self.n_substeps = n_substeps

        if self.method not in ("rk4", "euler"):
            raise ValueError(
                f"Unsupported discretization method '{method}'. Supported methods: 'rk4', 'euler'."
            )

    def _dyn(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.continuous_system._dyn(x, u)

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dt = self.dT
        dyn = self.continuous_system._dyn
        x_next = x

        if self.method == "euler":
            for _ in range(self.n_substeps):
                x_next = x_next + dt * dyn(x_next, u)
            return x_next

        for _ in range(self.n_substeps):
            k1 = dyn(x_next, u)
            k2 = dyn(x_next + 0.5 * dt * k1, u)
            k3 = dyn(x_next + 0.5 * dt * k2, u)
            k4 = dyn(x_next + dt * k3, u)
            x_next = x_next + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def get_state_dim(self) -> int:
        return int(self.continuous_system.get_state_dim())

    def get_input_dim(self) -> int:
        return int(self.continuous_system.get_input_dim())

    def get_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.continuous_system.get_output(x)

    def get_output_dim(self) -> int:
        return int(self.continuous_system.get_output_dim())

    def get_state_init_window_length(self):
        getter = getattr(self.continuous_system, "get_state_init_window_length", None)
        return getter() if callable(getter) else None


class ContinuousDoublePendulum(ContinuousDynamicalControlSystem):
    def __init__(self, dT=0.1, g=1, m1=1, m2=0.5, l1=1, l2=0.5, d1=0.1, d2=0.1):
        self.dT = float(dT)
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2

    def _dyn(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        th1, w1, th2, w2 = x.unbind(dim=-1)
        m1_t = self.m1
        m2_t = self.m2
        l1_t = self.l1
        l2_t = self.l2
        d1_t = self.d1
        d2_t = self.d2
        g_t = self.g

        dth = th2 - th1
        s = torch.sin(dth)
        c = torch.cos(dth)

        m11 = (m1_t + m2_t) * l1_t * l1_t
        m12 = m2_t * l1_t * l2_t * c
        m22 = m2_t * l2_t * l2_t
        det = m11 * m22 - m12 * m12 + 1e-9

        c1 = -m2_t * l1_t * l2_t * s * (w2**2)
        c2 = m2_t * l1_t * l2_t * s * (w1**2)
        g1 = (m1_t + m2_t) * g_t * l1_t * torch.sin(th1)
        g2 = m2_t * g_t * l2_t * torch.sin(th2)

        d1_term = d1_t * w1 + d2_t * (w1 - w2)
        d2_term = d2_t * (w2 - w1)

        tau1 = u.squeeze(-1)
        tau2 = torch.zeros_like(tau1)

        rhs1 = tau1 - (c1 + g1 + d1_term)
        rhs2 = tau2 - (c2 + g2 + d2_term)

        a1 = (rhs1 * m22 - rhs2 * m12) / det
        a2 = (m11 * rhs2 - m12 * rhs1) / det
        return torch.stack([w1, a1, w2, a2], dim=-1)

    def get_state_dim(self) -> int:
        return 4

    def get_state_init_window_length(self):
        return None

    def get_input_dim(self) -> int:
        return 1


class DoublePendulum(_DiscretizedContinuousSystem):
    def __init__(self, dT=0.1, g=1, m1=1, m2=0.5, l1=1, l2=0.5, d1=0.1, d2=0.1):
        self.dT = float(dT)
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.d1 = d1
        self.d2 = d2
        continuous = ContinuousDoublePendulum(
            dT=self.dT,
            g=self.g,
            m1=self.m1,
            m2=self.m2,
            l1=self.l1,
            l2=self.l2,
            d1=self.d1,
            d2=self.d2,
        )
        super().__init__(continuous_system=continuous, dT=self.dT, Ts=self.dT, method="rk4")


class DoublePendulumOUT(DoublePendulum):
    """Observed-output double pendulum with output (theta1, theta2)."""

    def get_output_dim(self) -> int:
        return 2

    def get_output(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., [0, 2]]


class ContinuousUnicycle(ContinuousDynamicalControlSystem):
    def __init__(self, dT=0.05):
        self.dT = dT

    def _dyn(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                x[:, [4]] * torch.cos(x[:, [2]]),
                x[:, [4]] * torch.sin(x[:, [2]]),
                x[:, [3]],
                u[:, [1]],
                u[:, [0]],
            ],
            dim=1,
        )

    def get_state_dim(self) -> int:
        return 5

    def get_input_dim(self) -> int:
        return 2


class Unicycle(_DiscretizedContinuousSystem):
    def __init__(self, dT=0.05):
        self.dT = dT
        continuous = ContinuousUnicycle(dT=self.dT)
        super().__init__(continuous_system=continuous, dT=self.dT, Ts=self.dT, method="rk4")


__all__ = [
    "IOControlSystem",
    "ContinuousDynamicalControlSystem",
    "DynamicalControlSystem",
    "_DiscretizedContinuousSystem",
    "ContinuousDoublePendulum",
    "DoublePendulum",
    "DoublePendulumOUT",
    "ContinuousUnicycle",
    "Unicycle",
]
