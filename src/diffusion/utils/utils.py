#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Optional

import torch


def _objective_scalar(
    f: Callable,
    x: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    value = f(x)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=x.device, dtype=x.dtype)
    if value.numel() == 1:
        return value.reshape(())

    reduction_key = str(reduction).lower()
    if reduction_key == "mean":
        return value.mean()
    if reduction_key == "sum":
        return value.sum()
    raise ValueError(f"Unsupported reduction '{reduction}'. Expected one of: mean, sum")


def compute_grad(
    x: torch.Tensor,
    f: Callable,
    grad_f: Optional[Callable] = None,
    reduction: str = "mean",
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute gradient of ``f`` at ``x`` with optional explicit gradient callback.

    Args:
        x: Input tensor.
        f: Objective callable.
        grad_f: Optional explicit gradient callable. If provided, it is used directly.
        reduction: Scalarization used when ``f(x)`` is non-scalar ("mean" or "sum").
        create_graph: Passed to ``torch.autograd.grad`` when ``grad_f`` is not provided.

    Returns:
        Gradient tensor matching ``x`` shape.
    """
    if grad_f is not None:
        grad = grad_f(x)
        if not isinstance(grad, torch.Tensor):
            grad = torch.as_tensor(grad, device=x.device, dtype=x.dtype)
        return grad

    x_req = x.detach().clone().requires_grad_(True)
    f_scalar = _objective_scalar(f, x_req, reduction=reduction)
    return torch.autograd.grad(f_scalar, x_req, create_graph=create_graph)[0]


def compute_jacobian(f, x: torch.Tensor) -> torch.Tensor:
    """
    Compute batched Jacobians for a batched map f: (B, d) -> (B, m).

    Args:
        f: Callable taking x with shape (B, d) and returning y with shape (B, m).
        x: Input points with shape (B, d).

    Returns:
        Jacobians with shape (B, m, d), where J[b, i, j] = d f_i(x_b) / d x_j.
    """
    if x.ndim != 2:
        raise ValueError(f"x must have shape (B, d), got {tuple(x.shape)}")

    x_req = x.detach().clone().requires_grad_(True)
    y = f(x_req)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y, device=x_req.device, dtype=x_req.dtype)

    if y.ndim != 2:
        raise ValueError(f"f(x) must have shape (B, m), got {tuple(y.shape)}")
    if y.shape[0] != x_req.shape[0]:
        raise ValueError(
            f"Batch size mismatch between x and f(x): {x_req.shape[0]} vs {y.shape[0]}"
        )

    m = y.shape[1]
    jac_cols = []
    for j in range(m):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, j] = 1.0
        grad_j = torch.autograd.grad(
            outputs=y,
            inputs=x_req,
            grad_outputs=grad_outputs,
            retain_graph=(j + 1 < m),
            create_graph=False,
        )[0]
        jac_cols.append(grad_j)

    return torch.stack(jac_cols, dim=1)
