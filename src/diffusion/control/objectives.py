#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
from typing import Callable, Sequence, List, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class WeightedSliceTerm:
    slice_spec: Any
    reference: torch.Tensor
    weight: float = 1.0
    loss_type: str = "L2"


def _to_index(slice_spec):
    return (...,) + tuple(slice_spec) if isinstance(slice_spec, tuple) else (..., slice_spec)


def _validate_terms(terms: Sequence[WeightedSliceTerm]) -> None:
    if len(terms) == 0:
        raise ValueError("At least one weighted slice term is required")
    weight_sum = 0.0
    for i, term in enumerate(terms):
        if not isinstance(term, WeightedSliceTerm):
            raise TypeError(f"terms[{i}] must be WeightedSliceTerm, got {type(term)}")
        if not math.isfinite(float(term.weight)):
            raise ValueError(f"weights[{i}] must be finite, got {term.weight}")
        weight_sum += float(term.weight)
    

def build_weighted_slice_objective(
    terms: Sequence[WeightedSliceTerm],
) -> Callable[[torch.Tensor], torch.Tensor]:
    
    _validate_terms(terms)
    weight_sum = sum(float(t.weight) for t in terms)

    def objective_fn(x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        vals = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        for term in terms:
            x_sel = x[_to_index(term.slice_spec)]
            ref = term.reference.to(device=x.device, dtype=x.dtype)
            assert x_sel.shape[1:] == ref.shape[1:], f"Input of invalid shape: {x_sel.shape[1:]}, but reference has {ref.shape[1:]}"
            err = (x_sel - ref).reshape(batch_size, -1)
            if term.loss_type == "L2":
                err_norm_loss = err.pow(2).sum(dim=1)
            elif term.loss_type == "L2sqrt":
                err_norm_loss = err.pow(2).sum(dim=1).sqrt()
            elif term.loss_type == "L1":
                err_norm_loss = err.abs().sum(dim=1)
            elif term.loss_type == "Linf":
                err_norm_loss = err.abs().max(dim=1)[0]
            else:
                raise ValueError(f"Loss type unknown: {term.loss_type}")
            vals = vals + term.weight * err_norm_loss
        return vals / weight_sum

    return objective_fn


def build_reference_tracking_objective(
    reference_trajectory: torch.Tensor,
    slice_spec: List[Any] | None = None,
    loss_type="L2"
) -> Callable[[torch.Tensor], torch.Tensor]:
    ref = torch.as_tensor(reference_trajectory)
    if slice_spec is None:
        slice_spec = tuple(slice(None) for _ in range(ref.ndim))
    
    return build_weighted_slice_objective(
        terms=[WeightedSliceTerm(slice_spec=slice_spec, reference=ref, weight=1.0, loss_type=loss_type)]
        )