#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional MuJoCo adapter for trajectory generation and control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .systems import DynamicalControlSystem


def _import_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "MuJoCo support is optional. Install it with `pip install -e .[mujoco]` "
            "or `pip install mujoco>=3.0.0`."
        ) from exc
    return mujoco


class MuJoCoSystem(DynamicalControlSystem):
    """Discrete-time MuJoCo system with full-state output y = [qpos, qvel]."""

    def __init__(
        self,
        model_path: str,
        dT: float,
        Ts: float | None = None,
    ):
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.dT = float(dT)
        if self.dT <= 0.0:
            raise ValueError(f"dT must be positive, got {self.dT}.")

        self.Ts = self.dT if Ts is None else float(Ts)
        if self.Ts <= 0.0:
            raise ValueError(f"Ts must be positive, got {self.Ts}.")
        ratio = self.Ts / self.dT
        self.frame_skip = int(round(ratio))
        tolerance = 1e-9 * max(1.0, abs(ratio))
        if self.frame_skip < 1 or abs(ratio - self.frame_skip) > tolerance:
            raise ValueError(
                "Ts must be an integer multiple of dT (Ts = frame_skip * dT). "
                f"Got Ts={self.Ts}, dT={self.dT}."
            )

        self._mujoco = _import_mujoco()
        self.model = self._mujoco.MjModel.from_xml_path(self.model_path)
        self.data = self._mujoco.MjData(self.model)
        self.model.opt.timestep = self.dT

        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.nu = int(self.model.nu)

        if int(self.model.na) != 0:
            raise ValueError(
                "MuJoCoSystem currently supports models with no actuator activation state "
                f"(model.na must be 0, got {int(self.model.na)})."
            )

    def get_input_dim(self) -> int:
        return self.nu

    def get_state_dim(self) -> int:
        return self.nq + self.nv

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_next = torch.empty_like(x)

        for i in range(x.shape[0]):
            x_i = np.asarray(x[i].detach().cpu().numpy(), dtype=np.float64)
            u_i = np.asarray(u[i].detach().cpu().numpy(), dtype=np.float64)

            qpos = x_i[: self.nq]
            qvel = x_i[self.nq :]

            self._mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            if self.nu > 0:
                self.data.ctrl[:] = u_i

            for _ in range(self.frame_skip):
                self._mujoco.mj_step(self.model, self.data)

            x_next_i = np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])
            x_next[i] = torch.from_numpy(x_next_i).to(device=x.device, dtype=x.dtype)

        return x_next


__all__ = ["MuJoCoSystem"]
