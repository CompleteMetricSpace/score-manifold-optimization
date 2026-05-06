#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trajectory constraints retained for paper control workflows."""

from __future__ import annotations

from typing import Optional

import torch

from diffusion.spaces import Constraint, TrajectorySpace
from diffusion.utils.utils import compute_jacobian


class TrajectoryConstraint(Constraint):
    """
    Base class for constraints on trajectories in TrajectorySpace.

    Trajectories have shape (batch, horizon, input_dim + output_dim), where the
    first input_dim channels are controls and the remaining channels are outputs.
    """

    def __init__(self, space: TrajectorySpace):
        if not isinstance(space, TrajectorySpace):
            raise TypeError(f"TrajectoryConstraint requires TrajectorySpace, got {type(space)}.")
        self._space = space
        self.horizon = int(space.horizon)
        self.input_dim = int(space.input_dim)
        self.output_dim = int(space.output_dim)

    def get_space(self) -> TrajectorySpace:
        return self._space

    def split_input_output(self, x: torch.Tensor):
        return self.get_space().split_input_output(x)

    def get_dim_ambient(self):
        return self.horizon * (self.input_dim + self.output_dim)

    def get_dims(self):
        return (self.horizon, self.input_dim + self.output_dim)

    def closest(self, x):
        return self.project(x)


class DynamicsConstraint(TrajectoryConstraint):
    """
    Constraint enforcing outputs consistent with a dynamical control system.
    """

    def __init__(self, space: TrajectorySpace, system):
        super().__init__(space)
        self.system = system
        self.state_dim = int(system.get_state_dim())
        self.input_dim = int(system.get_input_dim())
        self.output_dim = int(system.get_output_dim())

    @classmethod
    def from_system(cls, system, horizon: int):
        from diffusion.spaces.euclidean import TrajectorySpace

        space = TrajectorySpace(
            horizon=horizon,
            input_dim=system.get_input_dim(),
            output_dim=system.get_output_dim(),
        )
        return cls(space=space, system=system)

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        u, y = self.split_input_output(x)
        x0 = self._infer_initial_state(y)
        y_sim = self._simulate_output(x0, u)
        return (y - y_sim).abs().sum(dim=[1, 2])

    def is_state_output(self):
        return self.state_dim == self.output_dim

    def project_output(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        u, y = self.split_input_output(x)
        if x0 is None:
            x0 = self._infer_initial_state(y)
        if x0 is None:
            raise ValueError(
                "Cannot infer initial state from outputs when output_dim != state_dim; "
                f"got output_dim={self.output_dim}, state_dim={self.state_dim}. "
                "Pass x0 explicitly."
            )
        y_sim = self._simulate_output(x0, u)
        return torch.cat([u, y_sim], dim=-1)

    def _infer_initial_state(self, y: torch.Tensor):
        y0 = y[:, 0]
        return y0 if self.is_state_output() else None

    def project(self, x: torch.Tensor, cfg: Optional[object] = None) -> torch.Tensor:
        del cfg
        if not self.is_state_output():
            raise ValueError(
                "DynamicsConstraint.project requires systems with state outputs "
                f"(output_dim={self.output_dim}, state_dim={self.state_dim})."
            )
        u, y = self.split_input_output(x)
        x0 = self._infer_initial_state(y)
        y_sim = self._simulate_output(x0, u)
        return torch.cat([u, y_sim], dim=-1)

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if not self.is_state_output():
            raise NotImplementedError(
                "Tangent projection for dynamics with non-state outputs is not implemented."
            )

        violation = self.violation(x)
        if not (violation < 1e-6).all():
            print(
                "Warning: some points are off the trajectory manifold. "
                f"Maximum violation: {float(violation.max())}"
            )
        if x.shape != v.shape:
            raise ValueError(f"x and v must have identical shapes, got {x.shape} and {v.shape}.")

        expected_shape = torch.Size([x.shape[0], self.horizon, self.input_dim + self.state_dim])
        if x.shape != expected_shape:
            raise ValueError(f"x must have shape {expected_shape}, got {x.shape}.")

        u, y = self.split_input_output(x)
        x0 = self._infer_initial_state(y)
        batch_size = x.shape[0]

        def pack(u_local: torch.Tensor, x0_local: torch.Tensor) -> torch.Tensor:
            return torch.cat([u_local.reshape(batch_size, self.horizon * self.input_dim), x0_local], dim=-1)

        def unpack(z_local: torch.Tensor):
            u_local = z_local[:, :-self.state_dim].reshape(batch_size, self.horizon, self.input_dim)
            x0_local = z_local[:, -self.state_dim:]
            return u_local, x0_local

        def pack_out(u_local: torch.Tensor, x0_local: torch.Tensor, x1_local: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    u_local.reshape(batch_size, self.horizon * self.input_dim),
                    x0_local,
                    x1_local.reshape(batch_size, (self.horizon - 1) * self.state_dim),
                ],
                dim=-1,
            )

        def unpack_out(z_local: torch.Tensor):
            u_local = z_local[:, : self.horizon * self.input_dim].reshape(
                batch_size, self.horizon, self.input_dim
            )
            x0_local = z_local[
                :, self.horizon * self.input_dim : self.horizon * self.input_dim + self.state_dim
            ]
            x1_local = z_local[:, self.horizon * self.input_dim + self.state_dim :].reshape(
                batch_size, self.horizon - 1, self.state_dim
            )
            return u_local, x0_local, x1_local

        def fn(z_local: torch.Tensor) -> torch.Tensor:
            u_local, x0_local = unpack(z_local)
            x1_sim = self.system.simulate(x0_local, u_local)[:, 1:]
            return pack_out(u_local, x0_local, x1_sim)

        uv, yv = self.split_input_output(v)
        x0v = self._infer_initial_state(yv)
        x1v = yv[:, 1:]
        zv = pack_out(uv, x0v, x1v)

        jacobian = compute_jacobian(fn, pack(u, x0))
        w = torch.linalg.lstsq(jacobian, zv.unsqueeze(-1)).solution.squeeze(-1)
        pv = (jacobian @ w.unsqueeze(-1)).squeeze(-1)
        upv, x0pv, x1pv = unpack_out(pv)
        return torch.cat([upv, torch.cat([x0pv.unsqueeze(1), x1pv], dim=1)], dim=-1)

    def sample(self, n: int) -> torch.Tensor:
        x0 = torch.randn(n, self.state_dim)
        u = torch.randn(n, self.horizon, self.input_dim)
        y = self._simulate_output(x0, u)
        return torch.cat([u, y], dim=-1)

    def _simulate_output(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.system.simulate_out(x0, u)

