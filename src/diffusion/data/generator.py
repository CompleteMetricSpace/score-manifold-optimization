#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generators for diffusion models.

Provides unified interface for generating datasets from:
- Manifolds (Stiefel, Sphere, etc.)
- Dynamical systems (Unicycle, Pendulum, etc.)
- Regions (UnitSquare, Ball, etc.)
"""

import math
import torch
import torch.nn.functional as F
import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
from datetime import datetime

from diffusion.spaces import Constraint, Region, Manifold
from diffusion.spaces.euclidean import EuclideanSpace, MatrixSpace, TrajectorySpace, VectorSpace


def _space_to_metadata(space) -> Dict[str, Any]:
    """Serialize supported Space objects into canonical metadata dict form."""
    if isinstance(space, TrajectorySpace):
        return {
            "class": "TrajectorySpace",
            "params": {
                "horizon": int(space.horizon),
                "input_dim": int(space.input_dim),
                "output_dim": int(space.output_dim),
            },
        }
    if isinstance(space, MatrixSpace):
        return {
            "class": "MatrixSpace",
            "params": {"m": int(space.m), "n": int(space.n)},
        }
    if isinstance(space, VectorSpace):
        return {
            "class": "VectorSpace",
            "params": {"dim": int(space.dim)},
        }
    if isinstance(space, EuclideanSpace):
        return {
            "class": "EuclideanSpace",
            "params": {"dims": [int(d) for d in space.get_dims()]},
        }
    raise TypeError(f"Unsupported space type for metadata serialization: {type(space)}")


def generate_id_signals(
    N_data: int,
    N: int,
    d: int,
    minimum: float = 0.0,
    maximum: float = 1.0,
    proportions=None,
    *,
    device=None,
    dtype=torch.float32,
    seed=None,
):
    """Generate identification signals with shape ``(N_data, N, d)``."""
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(int(seed))
        rand = lambda *size: torch.rand(*size, generator=g, device=device, dtype=dtype)
        randn = lambda *size: torch.randn(*size, generator=g, device=device, dtype=dtype)
        randint = lambda low, high, size: torch.randint(low, high, size, generator=g, device=device)
    else:
        rand = lambda *size: torch.rand(*size, device=device, dtype=dtype)
        randn = lambda *size: torch.randn(*size, device=device, dtype=dtype)
        randint = lambda low, high, size: torch.randint(low, high, size, device=device)

    if proportions is None:
        p_noise = p_sine = p_prbs = 1 / 3
    else:
        if isinstance(proportions, dict):
            p_noise = float(proportions.get("noise", 0.0))
            p_sine = float(proportions.get("sine", 0.0))
            p_prbs = float(proportions.get("prbs", 0.0))
            if (p_noise + p_sine + p_prbs) == 0:
                p_noise = p_sine = p_prbs = 1 / 3
        else:
            p_noise, p_sine, p_prbs = map(float, proportions)
    total = max(p_noise + p_sine + p_prbs, 1e-12)
    p_noise, p_sine, p_prbs = p_noise / total, p_sine / total, p_prbs / total

    kinds = torch.bucketize(
        rand(N_data),
        boundaries=torch.tensor([p_noise, p_noise + p_sine], device=device),
    )
    out = torch.empty((N_data, N, d), device=device, dtype=dtype)

    def _scale_to_range(x):
        mx = x.abs().amax(dim=(0, 1), keepdim=True)
        mx = torch.where(mx < 1e-6, torch.ones_like(mx), mx)
        x = x / mx.clamp_min(1e-12)
        return (x + 1) * 0.5 * (maximum - minimum) + minimum

    def make_noise():
        x = randn(N, d)
        kernel_choices = torch.tensor([1, 3, 5, 9, 15, 31, 45], device=device)
        k = kernel_choices[int(randint(0, len(kernel_choices), (1,)))]
        if k.item() > 1:
            w = torch.ones((d, 1, int(k)), device=device, dtype=dtype) / float(k)
            x_ = x.T.unsqueeze(0)
            x = F.conv1d(x_, w, padding=int(k.item() // 2), groups=d).squeeze(0).T
        return _scale_to_range(x)

    def make_sines():
        t = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)
        n_comp = int(randint(1, 5, (1,)))
        freqs = 10 ** (torch.linspace(-2.7, -0.4, n_comp, device=device, dtype=dtype))
        freqs = freqs[torch.randperm(n_comp, device=device)[:n_comp]]
        amps = 0.5 + 0.5 * rand(n_comp, d)
        phases = 2 * math.pi * rand(n_comp, d)
        w = 2 * math.pi * freqs.view(-1, 1, 1)
        x = (amps.unsqueeze(1) * torch.sin(w * t.unsqueeze(0) + phases.unsqueeze(1))).sum(0)
        x += 0.05 * randn(N, d)
        return _scale_to_range(x)

    def make_prbs():
        x = torch.empty(N, d, device=device, dtype=dtype)
        dwell_min = int(max(1, N // 200))
        dwell_max = int(max(dwell_min + 1, N // 10))
        n_levels = int(randint(2, 5, (1,)))
        levels = torch.linspace(-1.0, 1.0, n_levels, device=device, dtype=dtype)
        for ch in range(d):
            vals = levels[torch.randperm(n_levels, device=device)]
            current = vals[0].item()
            idx = 0
            while idx < N:
                dwell = int(randint(dwell_min, dwell_max + 1, (1,)))
                x[idx : min(idx + dwell, N), ch] = current
                nxt = int(randint(0, n_levels, (1,)))
                if levels[nxt].item() == current and n_levels > 1:
                    nxt = (nxt + 1) % n_levels
                current = levels[nxt].item()
                idx += dwell
        x += 0.01 * randn(N, d)
        return _scale_to_range(x)

    for i in range(N_data):
        kind = int(kinds[i].item())
        if kind == 0:
            out[i] = make_noise()
        elif kind == 1:
            out[i] = make_sines()
        else:
            out[i] = make_prbs()

    return out


class DataGenerator(ABC):
    """
    Abstract base class for data generators.

    All generators provide:
    - generate(n_samples): Generate n samples
    - get_metadata(): Return generation metadata
    """

    @abstractmethod
    def generate(self, n_samples: int) -> torch.Tensor:
        """
        Generate n_samples from this data source.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tensor of shape (n_samples, *data_shape)
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata describing this data generator.

        Returns:
            Dictionary with generator configuration
        """
        pass

    def get_constraint(self) -> Optional[Constraint]:
        """
        Return the constraint object if available.

        Returns:
            Constraint instance or None
        """
        return None


class ManifoldDataGenerator(DataGenerator):
    """
    Generator for manifold data.

    Samples from a manifold with optional additive noise.
    """

    def __init__(self, manifold: Manifold, noise_std: float = 0.0):
        """
        Args:
            manifold: Manifold constraint to sample from
            noise_std: Standard deviation of additive Gaussian noise
        """
        if not isinstance(manifold, Manifold):
            raise TypeError(f"Expected Manifold, got {type(manifold)}")

        self.manifold = manifold
        self.noise_std = noise_std

    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate samples from manifold with optional noise"""
        # Sample from manifold
        samples = self.manifold.sample(n_samples)

        # Add noise if specified
        if self.noise_std > 0:
            noise = torch.randn_like(samples) * self.noise_std
            samples = samples + noise

        return samples

    def get_metadata(self) -> Dict[str, Any]:
        """Return manifold generation metadata"""
        metadata = {
            "space": _space_to_metadata(self.manifold.get_space()),
            "constraint": {
                "class": self.manifold.__class__.__name__,
                "params": self._extract_constraint_params(self.manifold),
            },
            "noise_std": self.noise_std,
            "generation_config": {},
        }
        return metadata

    @staticmethod
    def _extract_constraint_params(obj) -> Dict[str, Any]:
        """Extract all constructor parameters from constraint object."""
        params = {}
        try:
            sig = inspect.signature(obj.__class__.__init__)
            for param_name in sig.parameters:
                if param_name == 'self':
                    continue
                if hasattr(obj, param_name):
                    value = getattr(obj, param_name)
                    if isinstance(value, torch.Tensor):
                        params[param_name] = value.tolist()
                    elif isinstance(value, (int, float, str, bool, type(None))):
                        params[param_name] = value
        except Exception:
            pass
        return params

    def get_constraint(self) -> Manifold:
        """Return the manifold constraint"""
        return self.manifold


class DynamicsDataGenerator(DataGenerator):
    """
    Generator for dynamical system trajectory data.

    Simulates trajectories from a dynamical control system.
    """

    def __init__(
        self,
        system,  # DynamicalControlSystem
        horizon: int,
        input_generator: Optional[Callable] = None,
        x0_sampler: Optional[Callable] = None,
    ):
        """
        Args:
            system: DynamicalControlSystem instance
            horizon: Trajectory length (number of timesteps)
            input_generator: Optional function (n_samples) -> (n, horizon, input_dim)
                            Defaults to standard normal
            x0_sampler: Optional function (n_samples) -> (n, state_dim)
                       Defaults to standard normal
        """
        self.system = system
        self.horizon = horizon

        # Default input generator: standard normal
        if input_generator is None:
            input_dim = system.get_input_dim()
            input_generator = lambda n: torch.randn(n, horizon, input_dim)

        # Default initial state sampler: standard normal
        if x0_sampler is None:
            state_dim = system.get_state_dim()
            x0_sampler = lambda n: torch.randn(n, state_dim)

        self.input_generator = input_generator
        self.x0_sampler = x0_sampler

    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate trajectory samples"""
        # Generate control inputs
        u = self.input_generator(n_samples)

        # Sample initial states
        x0 = self.x0_sampler(n_samples)

        # Simulate output trajectories
        y = self.system.simulate_out(x0, u)

        # Concatenate [u, y] along last dimension
        trajectories = torch.cat([u, y], dim=-1)

        return trajectories

    def get_metadata(self) -> Dict[str, Any]:
        """Return dynamics generation metadata"""
        input_dim = self.system.get_input_dim()
        output_dim = self.system.get_output_dim()
        metadata = {
            "space": {
                "class": "TrajectorySpace",
                "params": {
                    "horizon": int(self.horizon),
                    "input_dim": int(input_dim),
                    "output_dim": int(output_dim),
                },
            },
            "constraint": {
                "class": self.system.__class__.__name__,
                "params": self._extract_system_params(),
            },
            "state_dim": self.system.get_state_dim(),
        }

        # Input generation info
        metadata["generation_config"] = {
            "input_type": "custom" if self.input_generator else "gaussian",
            "x0_distribution": "custom" if self.x0_sampler else "gaussian",
        }

        return metadata

    def _extract_system_params(self) -> Dict[str, Any]:
        """
        Extract all constructor parameters from system instance.

        Uses inspect.signature() to introspect __init__ parameters,
        then reads current values from instance attributes.

        Returns:
            Dict mapping parameter names to their values
        """
        params = {}

        try:
            # Get constructor signature
            sig = inspect.signature(self.system.__class__.__init__)

            # Iterate over parameters (skip 'self')
            for param_name, param_obj in sig.parameters.items():
                if param_name == 'self':
                    continue

                # Try to get current value from instance attribute
                if hasattr(self.system, param_name):
                    value = getattr(self.system, param_name)

                    # Handle different types
                    if isinstance(value, torch.Tensor):
                        # Convert tensors to lists for JSON serialization
                        params[param_name] = value.tolist()
                    elif isinstance(value, (int, float, str, bool, type(None))):
                        # Primitives can be stored directly
                        params[param_name] = value
                    else:
                        # For complex objects, store type info for debugging
                        params[param_name] = f"<{type(value).__name__}>"

        except Exception as e:
            # Fallback: if introspection fails, try simple attribute inspection
            print(f"Warning: Failed to extract params for {self.system.__class__.__name__}: {e}")
            print("  Falling back to attribute inspection...")

            for attr in dir(self.system):
                if not attr.startswith('_') and hasattr(self.system, attr):
                    value = getattr(self.system, attr)
                    if isinstance(value, (int, float, str, bool)):
                        params[attr] = value
                    elif isinstance(value, torch.Tensor):
                        params[attr] = value.tolist()

        return params

    def get_constraint(self):
        """
        Return dynamics constraint from system.

        Note: The constraint only includes the system's dynamics,
        not the custom input/x0 generators used for data generation.
        """
        from diffusion.control.trajectory import DynamicsConstraint
        return DynamicsConstraint.from_system(self.system, self.horizon)


class RegionDataGenerator(DataGenerator):
    """
    Generator for region data.

    Samples uniformly from a geometric region.
    """

    def __init__(self, region: Region):
        """
        Args:
            region: Region constraint to sample from
        """
        if not isinstance(region, Region):
            raise TypeError(f"Expected Region, got {type(region)}")

        self.region = region

    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate samples from region"""
        return self.region.sample(n_samples)

    def get_metadata(self) -> Dict[str, Any]:
        """Return region generation metadata"""
        metadata = {
            "space": _space_to_metadata(self.region.get_space()),
            "constraint": {
                "class": self.region.__class__.__name__,
                "params": ManifoldDataGenerator._extract_constraint_params(self.region),
            },
            "generation_config": {},
        }
        return metadata

    def get_constraint(self) -> Region:
        """Return the region constraint"""
        return self.region


def generate_dataset(
    generator_type: str,
    constraint: Constraint,
    n_train: int,
    n_test: int = 0,
    n_val: int = 0,
    val_ratio: float = 0.0,
    save_path: Optional[str] = None,
    **generator_kwargs
):
    """
    Unified dataset generation interface.

    Args:
        generator_type: "manifold", "dynamics", or "region"
        constraint: Constraint object (Manifold, System, or Region)
        n_train: Number of training samples
        n_test: Number of test samples (0 to skip)
        n_val: Number of validation samples (0 to skip, or use val_ratio)
        val_ratio: Fraction of train to use as validation (alternative to n_val)
        save_path: Path to save dataset (None to skip saving)
        **generator_kwargs: Additional args for specific generator:
            - ManifoldDataGenerator: noise_std
            - DynamicsDataGenerator: horizon, input_generator, x0_sampler
            - RegionDataGenerator: (none)

    Returns:
        DatasetBundle with generated data and metadata

    Example:
        >>> from diffusion.constraint import Stiefel
        >>> dataset = generate_dataset(
        ...     generator_type="manifold",
        ...     constraint=Stiefel(n=10, p=10),
        ...     n_train=5000,
        ...     n_test=1000,
        ...     noise_std=0.01,
        ...     save_path="stiefel_data.pt"
        ... )
    """
    from .loader import DatasetBundle
    from .saver import save_dataset

    # Create appropriate generator
    if generator_type == "manifold":
        if not isinstance(constraint, Manifold):
            raise TypeError(f"Expected Manifold for manifold generator, got {type(constraint)}")
        generator = ManifoldDataGenerator(constraint, **generator_kwargs)

    elif generator_type == "dynamics":
        # For dynamics, constraint should be a DynamicalControlSystem
        generator = DynamicsDataGenerator(constraint, **generator_kwargs)

    elif generator_type == "region":
        if not isinstance(constraint, Region):
            raise TypeError(f"Expected Region for region generator, got {type(constraint)}")
        generator = RegionDataGenerator(constraint, **generator_kwargs)

    else:
        raise ValueError(
            f"Unknown generator_type: {generator_type}. "
            f"Must be 'manifold', 'dynamics', or 'region'"
        )

    # Generate training data
    print(f"Generating {n_train} training samples...")
    train_data = generator.generate(n_train)

    # Handle validation split
    val_data = None
    if val_ratio > 0:
        n_val_actual = int(n_train * val_ratio)
        print(f"Splitting {n_val_actual} samples for validation...")
        val_data = train_data[-n_val_actual:]
        train_data = train_data[:-n_val_actual]
    elif n_val > 0:
        print(f"Generating {n_val} validation samples...")
        val_data = generator.generate(n_val)

    # Generate test data
    test_data = None
    if n_test > 0:
        print(f"Generating {n_test} test samples...")
        test_data = generator.generate(n_test)

    # Build metadata
    metadata = generator.get_metadata()
    metadata.update({
        "n_train": len(train_data),
        "n_test": len(test_data) if test_data is not None else 0,
        "n_val": len(val_data) if val_data is not None else 0,
        "val_ratio": val_ratio,
        "created": datetime.now().isoformat(),
        "version": "1.0",
    })

    # Add generator kwargs to metadata (ensure generation_config exists)
    if "generation_config" not in metadata:
        metadata["generation_config"] = {}
    metadata["generation_config"].update(generator_kwargs)

    bundle = DatasetBundle(
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        constraint=generator.get_constraint(),
        metadata=metadata,
    )

    # Save if requested
    if save_path:
        print(f"Saving dataset to {save_path}...")
        save_dataset(bundle, save_path)

    print("Dataset generation complete!")
    return bundle
