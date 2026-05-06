#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset loading utilities.

Canonical dataset format:
{
    "train_data": tensor,
    "test_data": tensor (optional),
    "val_data": tensor (optional),
    "metadata": {
        "space": {"class": "...", "params": {...}},
        "constraint": {"class": "...", "params": {...}} | None,
        ...
    }
}
"""

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

class DatasetBundle:
    """
    Container for dataset with tensors, optional reconstructed constraint, and metadata.

    Attributes:
        train_data: Training data tensor
        test_data: Test data tensor (optional)
        val_data: Validation data tensor (optional)
        constraint: Reconstructed constraint object (optional)
        metadata: Dataset metadata dictionary
    """

    def __init__(
        self,
        train_data: torch.Tensor,
        test_data: Optional[torch.Tensor] = None,
        val_data: Optional[torch.Tensor] = None,
        constraint=None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.constraint = constraint
        self.metadata = metadata or {}

    def get_space(self):
        """Reconstruct Space object from canonical metadata["space"]."""
        return _reconstruct_space(self.metadata.get("space"))

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = None,
        **kwargs,
    ) -> DataLoader:
        """Create PyTorch DataLoader for specified split."""
        if split == "train":
            data = self.train_data
            if shuffle is None:
                shuffle = True
        elif split == "test":
            data = self.test_data
            if shuffle is None:
                shuffle = False
        elif split == "val":
            data = self.val_data
            if shuffle is None:
                shuffle = False
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'test', or 'val'")

        if data is None:
            raise ValueError(f"No data available for split: {split}")

        return DataLoader(data, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __repr__(self):
        space_spec = self.metadata.get("space")
        return (
            f"DatasetBundle(\n"
            f"  train: {self.train_data.shape},\n"
            f"  test: {self.test_data.shape if self.test_data is not None else None},\n"
            f"  val: {self.val_data.shape if self.val_data is not None else None},\n"
            f"  constraint: {self.constraint.__class__.__name__ if self.constraint else None},\n"
            f"  space: {space_spec}\n"
            f")"
        )


def load_dataset(
    path: Union[str, Path],
    reconstruct_constraint: bool = True,
    device: str = "cpu",
    data_dir: Optional[str] = None,
) -> DatasetBundle:
    """Load dataset in canonical metadata format."""
    full_path = _resolve_data_path(path, data_dir)
    print(f"Loading dataset from {full_path}...")

    try:
        data_dict = torch.load(full_path, map_location=device, weights_only=False)
    except Exception as e:
        raise IOError(f"Failed to load dataset from {full_path}: {e}") from e
    if not isinstance(data_dict, dict):
        raise ValueError(
            "Unsupported dataset format. Expected dict payload with keys "
            "'train_data' and 'metadata'."
        )

    if "train_data" not in data_dict:
        raise ValueError("Dataset payload missing required key 'train_data'.")

    metadata = data_dict.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Dataset payload missing canonical metadata dict under key 'metadata'.")

    if "space" not in metadata:
        raise ValueError(
            "Dataset metadata missing required canonical key 'space'. "
            "Expected {'space': {'class': ..., 'params': {...}}, ...}."
        )

    # Validate space spec eagerly so malformed files fail at load time.
    _ = _reconstruct_space(metadata["space"])

    train_data = data_dict["train_data"]
    test_data = data_dict.get("test_data", None)
    val_data = data_dict.get("val_data", None)

    constraint = None
    if reconstruct_constraint and metadata.get("constraint") is not None:
        constraint = _reconstruct_constraint(metadata)
        constraint = _ensure_constraint_get_space(constraint)

    bundle = DatasetBundle(
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        constraint=constraint,
        metadata=metadata,
    )

    print("  Successfully loaded dataset:")
    print(f"    Train: {train_data.shape}")
    if test_data is not None:
        print(f"    Test: {test_data.shape}")
    if val_data is not None:
        print(f"    Val: {val_data.shape}")

    return bundle


def _ensure_constraint_get_space(constraint):
    """Enforce get_space()-based constraint interface."""
    if not hasattr(constraint, "get_space") or not callable(getattr(constraint, "get_space")):
        raise TypeError(
            "Reconstructed constraint does not implement get_space(). "
            "Constraint space must be accessed via get_space() only."
        )
    _ = constraint.get_space()
    return constraint


def _resolve_data_path(path: Union[str, Path], data_dir: Optional[str] = None) -> Path:
    """Resolve dataset path."""
    path = Path(path)

    if path.is_absolute():
        return path

    if data_dir is not None:
        return Path(data_dir) / path

    return path


def _reconstruct_space(space_spec: Dict[str, Any]):
    """Reconstruct space from canonical space dict."""
    if not isinstance(space_spec, dict):
        raise ValueError("metadata['space'] must be a dict with keys 'class' and 'params'.")

    class_name = space_spec.get("class")
    params = space_spec.get("params", {})
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("metadata['space']['class'] must be a non-empty string.")
    if not isinstance(params, dict):
        raise ValueError("metadata['space']['params'] must be a dict.")

    if class_name == "VectorSpace":
        from diffusion.spaces.euclidean import VectorSpace

        if "dim" not in params:
            raise ValueError("VectorSpace metadata requires params['dim'].")
        return VectorSpace(dim=int(params["dim"]))

    if class_name == "MatrixSpace":
        from diffusion.spaces.euclidean import MatrixSpace

        if "m" not in params or "n" not in params:
            raise ValueError("MatrixSpace metadata requires params['m'] and params['n'].")
        return MatrixSpace(m=int(params["m"]), n=int(params["n"]))

    if class_name == "TrajectorySpace":
        from diffusion.spaces.euclidean import TrajectorySpace

        for key in ("horizon", "input_dim", "output_dim"):
            if key not in params:
                raise ValueError(f"TrajectorySpace metadata requires params['{key}'].")
        return TrajectorySpace(
            horizon=int(params["horizon"]),
            input_dim=int(params["input_dim"]),
            output_dim=int(params["output_dim"]),
        )

    if class_name == "EuclideanSpace":
        from diffusion.spaces.euclidean import EuclideanSpace

        dims = params.get("dims")
        if not isinstance(dims, (list, tuple)):
            raise ValueError("EuclideanSpace metadata requires params['dims'] as list/tuple.")
        return EuclideanSpace(tuple(int(d) for d in dims))

    raise ValueError(f"Unsupported space class in metadata: '{class_name}'")


def _resolve_class(class_name: str, module_names):
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            if inspect.isclass(cls):
                return cls
    return None


def _filter_params_for_ctor(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only constructor-supported parameters."""
    try:
        sig = inspect.signature(cls.__init__)
        valid = {k for k in sig.parameters if k != "self"}
        return {k: v for k, v in params.items() if k in valid}
    except Exception:
        return params


def _inflate_constructor_value(value):
    """Inflate canonical object specs recursively into Python objects."""
    if isinstance(value, dict) and "class" in value and "params" in value:
        class_name = value.get("class")
        params = value.get("params", {})
        if isinstance(class_name, str) and class_name and isinstance(params, dict):
            resolved_class = _resolve_class(
                class_name,
                [
                    "diffusion.control.systems",
                    "diffusion.control.mujoco",
                    "diffusion.spaces",
                    "diffusion.spaces.manifolds",
                    "diffusion.spaces.regions",
                    "diffusion.control.trajectory",
                ],
            )
            if resolved_class is not None:
                nested_params = {
                    k: _inflate_constructor_value(v) for k, v in params.items()
                }
                return resolved_class(**_filter_params_for_ctor(resolved_class, nested_params))

    if isinstance(value, dict):
        return {k: _inflate_constructor_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_inflate_constructor_value(v) for v in value]

    return value


def _tensorize_nested_lists(value):
    """Convert numeric nested lists to tensors (for system reconstruction)."""
    if isinstance(value, list):
        try:
            return torch.tensor(value)
        except Exception:
            return [_tensorize_nested_lists(v) for v in value]
    if isinstance(value, dict):
        return {k: _tensorize_nested_lists(v) for k, v in value.items()}
    return value


def _reconstruct_constraint(metadata: Dict[str, Any]):
    """
    Reconstruct constraint from canonical metadata['constraint'].

    Expected shape:
    {
      "constraint": {"class": "<ConstraintOrSystemClass>", "params": {...}},
      "space": {"class": "...", "params": {...}}
    }
    """
    constraint_spec = metadata.get("constraint")
    if constraint_spec is None:
        return None
    if not isinstance(constraint_spec, dict):
        raise ValueError("metadata['constraint'] must be a dict or null.")

    class_name = constraint_spec.get("class")
    params = constraint_spec.get("params", {})
    if not isinstance(class_name, str) or not class_name:
        raise ValueError("metadata['constraint']['class'] must be a non-empty string.")
    if not isinstance(params, dict):
        raise ValueError("metadata['constraint']['params'] must be a dict.")

    # Case 1: Direct manifold/region-style constraint classes from diffusion.spaces
    constraint_cls = _resolve_class(
        class_name,
        ["diffusion.spaces", "diffusion.spaces.manifolds", "diffusion.spaces.regions"],
    )
    if constraint_cls is not None:
        return constraint_cls(**_filter_params_for_ctor(constraint_cls, params))

    # Case 2: System class in diffusion.control.systems or optional MuJoCo adapter
    try:
        from diffusion.control.systems import DynamicalControlSystem
        from diffusion.control.trajectory import DynamicsConstraint
        from diffusion.spaces.euclidean import TrajectorySpace
    except Exception as e:
        raise ValueError(f"Failed importing control-system reconstruction dependencies: {e}") from e

    system_class = _resolve_class(
        class_name,
        ["diffusion.control.systems", "diffusion.control.mujoco"],
    )
    if system_class is None:
        raise ValueError(
            "Unsupported control system class in metadata: "
            f"'{class_name}'. Paper repo supports only DynamicalControlSystem-based "
            "systems retained in diffusion.control.systems or optional systems in "
            "diffusion.control.mujoco. Regenerate the dataset with supported systems."
        )
    if not inspect.isclass(system_class):
        raise ValueError(f"Resolved '{class_name}' is not a class.")

    params = _inflate_constructor_value(params)
    filtered_params = _filter_params_for_ctor(system_class, params)

    try:
        system = system_class(**filtered_params)
    except (ImportError, FileNotFoundError, OSError) as e:
        raise ValueError(
            f"Failed to instantiate system class '{class_name}'. "
            "If this is a MuJoCo system, install the optional dependency with "
            "`pip install -e .[mujoco]` and ensure model_path exists. "
            f"Original error: {e}"
        ) from e
    except Exception as e:
        tensorized_params = {
            k: _tensorize_nested_lists(v) for k, v in filtered_params.items()
        }
        try:
            system = system_class(**tensorized_params)
        except Exception as e_tensor:
            raise ValueError(
                f"Failed to instantiate system class '{class_name}' from metadata params. "
                f"Original error: {e}. Tensorized error: {e_tensor}"
            ) from e_tensor

    space = _reconstruct_space(metadata.get("space"))
    if not isinstance(space, TrajectorySpace):
        raise ValueError(
            "System-based constraint reconstruction requires TrajectorySpace in metadata['space']."
        )

    if issubclass(system_class, DynamicalControlSystem):
        return DynamicsConstraint.from_system(system, horizon=space.horizon)

    raise ValueError(
        "Unsupported control system base class for paper repo: "
        f"'{class_name}'. Only DynamicalControlSystem-based constraints are supported."
    )
