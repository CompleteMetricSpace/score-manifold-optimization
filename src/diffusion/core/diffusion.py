#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion process definitions.

This module defines diffusion processes that specify the forward SDE
and provide methods for sampling and reverse-time generation.

Classes:
- DiffusionProcess: Abstract base class
- VPDiffusion: Variance Preserving diffusion
- VEDiffusion: Variance Exploding diffusion
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from diffusion.spaces import Space
from .sde import SDE, build_reverse_flow, sde_euler_maruyama_scalar


class DiffusionProcess(ABC):
    """
    Abstract diffusion process defining forward SDE and reverse sampling.

    A diffusion process specifies:
    - Forward SDE: dx = f(t,x)dt + g(t)dW
    - Reverse SDE: dx = [f(t,x) - g(t)²∇log p_t(x)]dt + g(t)dW̃
    - Terminal distribution p_T
    - Conditional distribution p_t(·|x_0)

    The process is independent of the score model (which provides ∇log p_t).

    Attributes:
        T: Terminal time
        space: Mathematical space where diffusion occurs
        data_dims: Shape of data in the space
    """

    def __init__(self, T: float, space: Space):
        """
        Initialize diffusion process.

        Args:
            T: Terminal time (diffusion runs from 0 to T)
            space: Space where diffusion occurs
        """
        assert T > 0, "Terminal time T must be positive"
        self.T = T
        self.space = space
        self.data_dims = space.get_dims()

    @abstractmethod
    def get_drift(self) -> Callable:
        """
        Return drift coefficient f(t, x).

        Returns:
            Function (t, x) -> drift where:
                t: (batch,) time values
                x: (batch, *data_dims) state values
                drift: (batch, *data_dims) drift values
        """
        pass

    @abstractmethod
    def get_diffusion(self) -> Callable:
        """
        Return diffusion coefficient g(t).

        Returns:
            Function (t) -> scalar where:
                t: (batch,) time values
                scalar: (batch,) diffusion coefficients
        """
        pass

    @abstractmethod
    def sample_prior(self, n: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from terminal distribution p_T.

        Args:
            n: Number of samples
            device: Device for computation

        Returns:
            samples: (n, *data_dims) samples from p_T
        """
        pass

    @abstractmethod
    def conditional_mean_var(
        self,
        t: torch.Tensor,
        x0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return mean and variance of conditional distribution p_t(·|x_0).

        For most diffusions, p_t(x|x_0) is Gaussian with closed-form
        mean and variance.

        Args:
            t: (batch,) time values
            x0: (batch, *data_dims) initial states

        Returns:
            mean: (batch, *data_dims) conditional mean
            var: (batch,) conditional variance (scalar per batch element)
        """
        pass

    def get_sde(self) -> SDE:
        """
        Get SDE object for forward process.

        Returns:
            SDE instance representing forward diffusion
        """
        return SDE(
            self.get_drift(),
            self.get_diffusion(),
            0,
            self.T,
            self.data_dims
        )

    def get_reverse_sde(self, score_model, flow_type: str = "SDE") -> SDE:
        """
        Construct reverse-time SDE for sampling.

        Args:
            score_model: Trained score model with .forward(t, x) method
            flow_type: "SDE" for stochastic, "ODE" for deterministic (probability flow)

        Returns:
            SDE instance representing reverse diffusion
        """
        f, g = self.get_drift(), self.get_diffusion()

        # Score model evaluates at diffusion time (no time reversal needed here
        # because build_reverse_flow handles the time reversal internally)
        score = lambda t, x: score_model(t, x)

        # Build reverse flow
        ndim = [1 for _ in self.data_dims]
        f_rev, g_rev = build_reverse_flow(f, g, score, self.T, ndim, flow_type)

        return SDE(f_rev, g_rev, 0, self.T, self.data_dims)

    def sample_forward(
        self,
        data_sample: torch.Tensor,
        T_max: float = None,
        end_only: bool = True,
        step_size: float = 2e-3,
        max_grid_pts: int = 10000,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Sample forward diffusion trajectory.

        Args:
            data_sample: (batch, *data_dims) initial data
            T_max: Maximum time (defaults to self.T)
            end_only: Return only final state
            step_size: Time step size
            max_grid_pts: Maximum number of grid points
            device: Computation device

        Returns:
            Final state or full trajectory
        """
        T_max = self.T if T_max is None else T_max
        assert 0 < T_max <= self.T

        n_grid_pts = min(int(T_max / step_size), max_grid_pts)

        g = self.get_diffusion()
        return sde_euler_maruyama_scalar(
            self.get_drift(),
            lambda t, x: g(t),
            t_end=T_max,
            n_grid_pts=n_grid_pts,
            init_list=data_sample,
            end_only=end_only,
            device=device
        )

    def sample_reverse(
        self,
        N_sample: int,
        score_model,
        noise_sample: torch.Tensor = None,
        T_max: float = None,
        flow_type: str = "SDE",
        end_only: bool = True,
        step_size: float = 2e-3,
        max_grid_pts: int = 10000,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Sample reverse diffusion trajectory (generation).

        Args:
            N_sample: Number of samples
            score_model: Trained score model
            noise_sample: Optional initial noise (defaults to sample_prior)
            T_max: Maximum time (defaults to self.T)
            flow_type: "SDE" or "ODE"
            end_only: Return only final state
            step_size: Time step size
            max_grid_pts: Maximum number of grid points
            device: Computation device

        Returns:
            Generated samples or full trajectory
        """
        T_max = self.T if T_max is None else T_max
        assert 0 < T_max <= self.T

        # Get reverse SDE
        f_rev, g_rev = self.get_reverse_sde(score_model, flow_type).f_drift, \
                       self.get_reverse_sde(score_model, flow_type).g_diff

        # Sample initial noise
        if noise_sample is None:
            noise_sample = self.sample_prior(N_sample, device=device)
        else:
            assert noise_sample.shape == torch.Size([N_sample, *self.data_dims])

        n_grid_pts = min(int(T_max / step_size), max_grid_pts)

        with torch.no_grad():
            result = sde_euler_maruyama_scalar(
                f_rev,
                lambda t, x: g_rev(t, x),
                t_end=T_max,
                n_grid_pts=n_grid_pts,
                init_list=noise_sample,
                end_only=end_only,
                device=device
            )

        return result

    def cond_sample(self, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Sample from conditional distribution p_t(·|x_0).

        Args:
            t: (batch,) time values
            x0: (batch, *data_dims) initial states

        Returns:
            samples: (batch, *data_dims) from p_t(·|x_0)
        """
        ndim = [1 for _ in x0.shape[1:]]
        mean, var = self.conditional_mean_var(t, x0)

        normal = torch.randn_like(x0)
        var_view = var.view(x0.shape[0], *ndim)

        return mean + torch.sqrt(var_view) * normal

    def __repr__(self):
        return f"{self.__class__.__name__}(T={self.T}, space={self.space})"


class VPDiffusion(DiffusionProcess):
    """
    Variance Preserving (VP) diffusion.

    Forward SDE:
        dx = -½β(t)x dt + √β(t) dW

    where β(t) is the noise schedule.

    Conditional distribution p_t(x|x_0) is Gaussian:
        mean = exp(-β̄(t)/2) x_0
        var = 1 - exp(-β̄(t))

    where β̄(t) = ∫₀ᵗ β(s)ds.

    Attributes:
        beta_min: Minimum noise level
        beta_max: Maximum noise level
    """

    def __init__(
        self,
        beta_min: float,
        beta_max: float,
        T: float,
        space: Space
    ):
        """
        Initialize VP diffusion.

        Args:
            beta_min: Minimum β value
            beta_max: Maximum β value
            T: Terminal time
            space: Space where diffusion occurs
        """
        assert beta_min <= beta_max, "Must have beta_min <= beta_max"
        super().__init__(T, space)

        self.beta_min = beta_min
        self.beta_max = beta_max

        # Define noise schedule functions
        # β̄(t) = t²(β_max - β_min)/(2T) + t*β_min
        self.beta_bar = lambda t: (
            t**2 * (self.beta_max - self.beta_min) / (2 * self.T) +
            t * self.beta_min
        )

        # β'(t) = t(β_max - β_min)/T + β_min
        self.beta_prime = lambda t: (
            t * (self.beta_max - self.beta_min) / self.T +
            self.beta_min
        )

        # α(t) = β'(t)/2
        self.alpha = lambda t: self.beta_prime(t) / 2

    def get_drift(self) -> Callable:
        """Return drift: f(t,x) = -α(t)x = -β'(t)x/2."""
        ndim = [1 for _ in self.data_dims]
        return lambda t, x: -self.alpha(t).view(x.shape[0], *ndim) * x

    def get_diffusion(self) -> Callable:
        """Return diffusion: g(t) = √β'(t)."""
        return lambda t: torch.sqrt(self.beta_prime(t))

    def sample_prior(self, n: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from N(0, I)."""
        return torch.randn(n, *self.data_dims, device=device)

    def conditional_mean_var(
        self,
        t: torch.Tensor,
        x0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return conditional distribution parameters.

        mean(t) = exp(-β̄(t)/2)
        var(t) = 1 - exp(-β̄(t))
        """
        # Compute mean coefficient: exp(-β̄(t)/2)
        beta_bar_t = self.beta_bar(t)
        mean_coef = torch.exp(-beta_bar_t / 2)

        # Compute variance: 1 - exp(-β̄(t))
        var = 1 - torch.exp(-beta_bar_t)

        # Broadcast mean coefficient
        ndim = [1 for _ in x0.shape[1:]]
        mean = mean_coef.view(-1, *ndim) * x0

        return mean, var

    def cond_mean_var_(self, t: torch.Tensor, mean: torch.Tensor, var: torch.Tensor):
        """
        In-place computation of conditional mean and variance coefficients.

        This is for backward compatibility with old code.

        Args:
            t: (batch,) time values
            mean: (batch,) tensor to fill with mean coefficients
            var: (batch,) tensor to fill with variance values
        """
        # Variance
        var.copy_(t)
        var.mul_(-(self.beta_max - self.beta_min) / (2 * self.T))
        var.add_(-self.beta_min).mul_(t).exp_().mul_(-1).add_(1)

        # Mean coefficient
        mean.copy_(t)
        mean.mul_(-(self.beta_max - self.beta_min) / (2 * self.T))
        mean.add_(-self.beta_min)
        mean.mul_(t)
        mean.mul_(0.5)
        mean.exp_()

    def __repr__(self):
        return f"VP(β_min={self.beta_min}, β_max={self.beta_max}, T={self.T})"


class VEDiffusion(DiffusionProcess):
    """
    Variance Exploding (VE) diffusion.

    Forward SDE:
        dx = σ(t)√(2σ'(t)/σ(t)) dW

    where σ(t) = σ_min * (σ_max/σ_min)^(t/T) is the noise schedule.

    Conditional distribution p_t(x|x_0) is Gaussian:
        mean = x_0
        var = σ²(t) - σ²_min

    Attributes:
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        sigma_ratio: σ_max/σ_min
    """

    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        T: float,
        space: Space
    ):
        """
        Initialize VE diffusion.

        Args:
            sigma_min: Minimum σ value
            sigma_max: Maximum σ value
            T: Terminal time
            space: Space where diffusion occurs
        """
        assert sigma_min <= sigma_max, "Must have sigma_min <= sigma_max"
        super().__init__(T, space)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_ratio = sigma_max / sigma_min

        # Define noise schedule: σ(t) = σ_min * (σ_max/σ_min)^(t/T)
        self.sigma = lambda t: sigma_min * torch.pow(self.sigma_ratio, t / T)
        self.cond_noise_var = lambda t: sigma_min**2*(torch.pow(self.sigma_ratio, 2*t/T)-1)

    def get_drift(self) -> Callable:
        """Return drift: f(t,x) = 0 (no drift in VE)."""
        return lambda t, x: torch.zeros_like(x, device=x.device)

    def get_diffusion(self) -> Callable:
        """Return diffusion: g(t) = σ(t)√(2log(σ_max/σ_min)/T)."""
        return lambda t: (
            self.sigma_min *
            torch.pow(self.sigma_ratio, t / self.T) *
            np.sqrt(2 * np.log(self.sigma_ratio) / self.T)
        )

    def sample_prior(self, n: int, device: str = 'cpu') -> torch.Tensor:
        """
        Sample from N(0, (σ²_max - σ²_min)I).

        Note: This accounts for the initial noise level σ_min.
        """
        std = np.sqrt(self.sigma_max**2 - self.sigma_min**2)
        return std * torch.randn(n, *self.data_dims, device=device)

    def conditional_mean_var(
        self,
        t: torch.Tensor,
        x0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return conditional distribution parameters.

        mean(t) = 1 (identity)
        var(t) = σ²_min * ((σ_max/σ_min)^(2t/T) - 1)
        """
        # Mean is just x0 (no drift)
        mean = x0

        # Variance
        var = self.sigma_min**2 * (torch.pow(self.sigma_ratio, 2 * t / self.T) - 1)

        return mean, var

    def cond_mean_var_(self, t: torch.Tensor, mean: torch.Tensor, var: torch.Tensor):
        """
        In-place computation of conditional mean and variance coefficients.

        This is for backward compatibility with old code.

        Args:
            t: (batch,) time values
            mean: (batch,) tensor to fill with mean coefficients
            var: (batch,) tensor to fill with variance values
        """
        # Mean coefficient is 1
        mean.copy_(torch.ones_like(t))

        # Variance
        var.copy_(self.sigma_min**2 * (torch.pow(self.sigma_ratio, 2 * t / self.T) - 1))

    def nabla_log_cond_density(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        x0: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute score of conditional distribution ∇log p_t(x|x_0).

        For VE: ∇log p_t(x|x_0) = -(x - x_0) / var(t)

        Args:
            t: (batch,) time values
            x: (batch, *data_dims) current states
            x0: (batch, *data_dims) initial states

        Returns:
            score: (batch, *data_dims) score values
        """
        ndim = [1 for _ in x0.shape[1:]]
        _, var = self.conditional_mean_var(t, x0)
        return -(x - x0) / var.view(x0.shape[0], *ndim)

    def __repr__(self):
        return f"VE(σ_min={self.sigma_min}, σ_max={self.sigma_max}, T={self.T})"


def create_diffusion(diff_options: dict, space: Space) -> DiffusionProcess:
    """
    Factory function to create diffusion process.

    Args:
        diff_options: Dictionary with keys:
            - type: "VariancePreserving" or "VarianceExploding"
            - beta_min, beta_max, T (for VP)
            - sigma_min, sigma_max, T (for VE)
        space: Space where diffusion occurs

    Returns:
        DiffusionProcess instance

    Example:
        >>> space = VectorSpace(10)
        >>> diff_options = {
        ...     "type": "VariancePreserving",
        ...     "beta_min": 0.1,
        ...     "beta_max": 20.0,
        ...     "T": 1.0
        ... }
        >>> diffusion = create_diffusion(diff_options, space)
    """
    diff_type = diff_options["type"]

    if diff_type == "VariancePreserving":
        return VPDiffusion(
            beta_min=diff_options["beta_min"],
            beta_max=diff_options["beta_max"],
            T=diff_options["T"],
            space=space
        )
    elif diff_type == "VarianceExploding":
        return VEDiffusion(
            sigma_min=diff_options["sigma_min"],
            sigma_max=diff_options["sigma_max"],
            T=diff_options["T"],
            space=space
        )
    else:
        raise ValueError(f"Unknown diff_type = {diff_type}")
