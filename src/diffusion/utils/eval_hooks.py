#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Dict

import torch


def normalize_eval_fns(eval_fns, default_name: str) -> Dict[str, Callable]:
    if eval_fns is None:
        return {}
    if callable(eval_fns):
        return {default_name: eval_fns}
    if isinstance(eval_fns, dict):
        out: Dict[str, Callable] = {}
        for key, fn in eval_fns.items():
            if not callable(fn):
                raise TypeError(f"Evaluator '{key}' must be callable")
            out[str(key)] = fn
        return out
    raise TypeError("Evaluator spec must be callable, dict[str, callable], or None")


def to_float_scalar(val) -> float:
    if isinstance(val, torch.Tensor):
        val_detached = val.detach()
        if val_detached.numel() == 1:
            return float(val_detached.item())
        return float(val_detached.mean().item())
    return float(val)


def evaluate_iterate(eval_fns: Dict[str, Callable], iterate: torch.Tensor, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, fn in eval_fns.items():
        out[f"{prefix}_{name}"] = to_float_scalar(fn(iterate))
    return out
