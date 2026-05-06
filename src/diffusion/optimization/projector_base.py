#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import torch


class Projector(ABC):
    """
    Unified projector interface used by optimization algorithms.
    """

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project ambient point(s) x onto the feasible/data manifold set.
        """
        raise NotImplementedError

    @abstractmethod
    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project ambient vector(s) v onto the tangent space defined by x.
        """
        raise NotImplementedError

class Proximal(ABC):
    """
    Unified proximal-operator interface used by splitting algorithms.
    """

    @abstractmethod
    def prox(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Evaluate proximal operator at x with step size gamma.
        """
        raise NotImplementedError


class ProjectorProximal(Proximal):
    """
    Proximal adapter around a Projector.

    This adapter intentionally ignores gamma and delegates to projector.project(x),
    which is useful when the prox map is a projection operator.
    """

    def __init__(self, projector: Projector):
        self.projector = projector

    def prox(self, x: torch.Tensor, gamma: float) -> torch.Tensor:
        _ = gamma
        return self.projector.project(x)


def build_proximal_from_projector(projector: Projector) -> Proximal:
    """
    Wrap a Projector as a Proximal operator.
    """
    return ProjectorProximal(projector=projector)


class IdProjector(Projector):
    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def project_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v
