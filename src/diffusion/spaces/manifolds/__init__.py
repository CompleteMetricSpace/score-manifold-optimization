#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical manifold package surface.

This package replaces the former single-file ``diffusion.spaces.manifolds``
module while preserving top-level imports for compatibility.
"""

from .base import AffineSubspace, Manifold, SliceSubspace
from .classical import DeformedTorus, Sphere, Stiefel

__all__ = [
    "Manifold",
    "AffineSubspace",
    "SliceSubspace",
    "Sphere",
    "Stiefel",
    "DeformedTorus",
]
