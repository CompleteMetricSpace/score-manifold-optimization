#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factory for creating score models with wrapper configurations.

This module provides a registry-based factory system for creating score models
with appropriate wrappers (diffusion scaling, residual connections, etc.).

Canonical model runtime path:
- Use create_model() for all active model instantiation.
- Register base models and wrapper aliases in diffusion.models.registry.
"""

import torch
import warnings
import inspect
from collections.abc import Mapping
from typing import Dict, List, Tuple, Callable, Any, Optional
from .base import ScoreModel
from .mlp import MLPScoreModel
from .matrix import TraceMLPScoreModel
from .temporal import UNet1DTimeScoreModel, TemporalUnetScoreModel
from .wrappers import (
    DiffusionScalingWrapper,
    ResidualWrapper,
    LogTimeWrapper,
    VParamWrapper
)
from diffusion.utils.embeddings import SinCosEmbedder, LinearEmbedder, LogEmbedder

_EMBEDDER_REGISTRY = {
    "SinCosEmbedder": lambda dim: SinCosEmbedder(dim),
    "LinearEmbedder": lambda dim: LinearEmbedder(),
    "LogEmbedder":    lambda dim: LogEmbedder(),
}


def _build_time_embedder(kwargs: dict, accepted_params: set) -> dict:
    """Resolve time_embedder string + time_emb_dim into a TimeEmbedder object."""
    if "time_embedder" not in accepted_params:
        kwargs.pop("time_embedder", None)
        kwargs.pop("time_emb_dim", None)
        return kwargs

    if "time_embedder" not in kwargs:
        raise ValueError(
            "time_embedder is required in model.params but was not provided. "
            f"Choose from: {list(_EMBEDDER_REGISTRY)}"
        )
    if "time_emb_dim" not in kwargs:
        raise ValueError("time_emb_dim is required in model.params but was not provided.")

    name = kwargs.pop("time_embedder")
    dim = kwargs.pop("time_emb_dim")

    if name not in _EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown time_embedder: '{name}'. Choose from: {list(_EMBEDDER_REGISTRY)}")

    kwargs["time_embedder"] = _EMBEDDER_REGISTRY[name](dim)
    return kwargs


# ============================================================================
# MODEL REGISTRY
# ============================================================================
# Populated by registry.py - DO NOT EDIT DIRECTLY

MODEL_REGISTRY: Dict[str, Callable] = {}

# ============================================================================
# WRAPPER CONFIGURATIONS
# ============================================================================
# Populated by registry.py - DO NOT EDIT DIRECTLY

WRAPPER_CONFIGS: Dict[str, Dict[str, Any]] = {}
_RESERVED_MODEL_KEYS = {"type", "params"}


def register_model(name: str, model_class_or_fn: Callable):
    """Register a model class or creation function."""
    MODEL_REGISTRY[name] = model_class_or_fn


def register_wrapper_config(name: str, base_model: str, wrappers: List[Tuple[str, Dict]],
                            base_kwargs: Optional[Dict] = None):
    """
    Register a wrapped model configuration.

    Args:
        name: Full model type name (e.g., "TraceMLPTimeModuleDiffusionTerm")
        base_model: Base model type name in MODEL_REGISTRY
        wrappers: List of (wrapper_type, kwargs) tuples to apply in order
        base_kwargs: Optional dict of kwargs overrides for the base model constructor
    """
    config = {
        "base": base_model,
        "wrappers": wrappers
    }
    if base_kwargs:
        config["base_kwargs"] = base_kwargs
    WRAPPER_CONFIGS[name] = config


def apply_wrapper(
    model: ScoreModel,
    wrapper_type: str,
    sigma_fn: Optional[Callable] = None,
    **kwargs
) -> ScoreModel:
    """
    Apply a wrapper to a score model.

    Args:
        model: Base score model to wrap
        wrapper_type: Type of wrapper ("diffusion_scaling", "residual", "log_time", "vparam")
        sigma_fn: Sigma function for diffusion-related wrappers (returns sigma^2(t))
        **kwargs: Additional kwargs for the wrapper
    """
    if wrapper_type == "diffusion_scaling":
        mode = kwargs.get("mode", "sigma")
        clip_eps = kwargs.get("clip_eps", 1e-7)
        assert sigma_fn is not None, "sigma_fn required for diffusion_scaling"
        return DiffusionScalingWrapper(model, sigma_fn, clip_eps=clip_eps, mode=mode)

    elif wrapper_type == "residual":
        positive = kwargs.get("positive", True)
        return ResidualWrapper(model, positive=positive)

    elif wrapper_type == "log_time":
        return LogTimeWrapper(model)

    elif wrapper_type == "vparam":
        clip_eps = kwargs.get("clip_eps", 1e-5)
        assert sigma_fn is not None, "sigma_fn required for vparam"
        return VParamWrapper(model, sigma_fn, clip_eps=clip_eps)

    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")


def _apply_xavier_initialization(model: ScoreModel) -> None:
    """Apply legacy-style Xavier initialization to model parameters."""
    for p in model.parameters():
        if p.ndim == 1:
            torch.nn.init.normal_(p)
        else:
            torch.nn.init.xavier_uniform_(p)


def _initialize_model(model: ScoreModel, init_scheme: str) -> None:
    """Initialize model parameters using the selected scheme."""
    if init_scheme == "xavier":
        _apply_xavier_initialization(model)
        return
    raise ValueError(f"Unknown init_scheme: '{init_scheme}'. Supported schemes: ['xavier']")


def create_model(
    model_type: str,
    space: 'Space',
    diffusion: Optional['Diffusion'] = None,
    config: Optional[Any] = None,
    initialize: bool = True,
    init_scheme: str = "xavier",
    **model_kwargs
) -> ScoreModel:
    """
    Unified model creation factory.

    Args:
        model_type: Model type name (e.g., 'mlp', 'TemporalUNet', 'TraceMLPTimeModuleResNet4')
        space: Space object
        diffusion: Diffusion process (for sigma_fn in wrappers)
        config: Optional config object (DictConfig or dict) for extracting params
        initialize: Whether to initialize freshly created model parameters
        init_scheme: Initialization scheme name (currently: "xavier")
        **model_kwargs: Additional runtime kwargs (override base_kwargs)

    Returns:
        Initialized score model with wrappers applied
    """
    # Optional runtime sigma_fn override for backward compatibility
    runtime_sigma_fn = model_kwargs.pop("sigma_fn", None)

    # Validate model type and resolve constructor specification
    if not model_type:
        raise ValueError("Missing required model type. Expected non-empty `model.type`.")

    base_model_type, model_constructor = _resolve_model_constructor(model_type)
    accepted_params, accepts_var_kwargs = _get_constructor_kwargs_spec(model_constructor)

    # Extract kwargs from config if provided (config kwargs are overridden by runtime kwargs)
    if config is not None:
        config_kwargs = _extract_model_kwargs(
            config=config,
            model_type=model_type,
            accepted_params=accepted_params | {"time_emb_dim"},
            accepts_var_kwargs=accepts_var_kwargs
        )
        model_kwargs = {**config_kwargs, **model_kwargs}

    # Resolve time_embedder string + time_emb_dim into a TimeEmbedder object
    model_kwargs = _build_time_embedder(model_kwargs, accepted_params)

    # Validate merged kwargs before instantiation for clearer errors than TypeError
    _validate_kwargs(
        kwargs=model_kwargs,
        model_type=model_type,
        accepted_params=accepted_params,
        accepts_var_kwargs=accepts_var_kwargs,
        source="merged model kwargs"
    )

    # Get sigma function from diffusion if available
    sigma_fn = None
    if runtime_sigma_fn is not None:
        sigma_fn = runtime_sigma_fn
    elif diffusion is not None:
        sigma_fn = lambda t: diffusion.cond_noise_var(t)

    model = None

    # Check if wrapped configuration exists
    if model_type in WRAPPER_CONFIGS:
        config_spec = WRAPPER_CONFIGS[model_type]
        base_type = config_spec["base"]

        if base_type not in MODEL_REGISTRY:
            raise ValueError(f"Base model '{base_type}' not found in MODEL_REGISTRY")

        # Merge base_kwargs from config with runtime kwargs
        merged_kwargs = dict(config_spec.get("base_kwargs", {}))
        merged_kwargs.update(model_kwargs)
        _validate_kwargs(
            kwargs=merged_kwargs,
            model_type=model_type,
            accepted_params=accepted_params,
            accepts_var_kwargs=accepts_var_kwargs,
            source="wrapper base kwargs + config/runtime kwargs"
        )

        # Create base model
        model_class = MODEL_REGISTRY[base_type]
        model = model_class(space=space, **merged_kwargs)

        # Apply wrappers in order
        for wrapper_type, wrapper_kwargs in config_spec["wrappers"]:
            model = apply_wrapper(model, wrapper_type, sigma_fn=sigma_fn, **wrapper_kwargs)

    elif model_type in MODEL_REGISTRY:
        # Direct model (no wrappers)
        model = MODEL_REGISTRY[model_type](space=space, **model_kwargs)
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys()) + list(WRAPPER_CONFIGS.keys())}"
        )

    if initialize:
        _initialize_model(model, init_scheme)

    return model


def _resolve_model_constructor(model_type: str) -> Tuple[str, Callable]:
    """Resolve the model constructor used for kwargs validation/instantiation."""
    if model_type in WRAPPER_CONFIGS:
        base_model_type = WRAPPER_CONFIGS[model_type]["base"]
        if base_model_type not in MODEL_REGISTRY:
            raise ValueError(f"Base model '{base_model_type}' not found in MODEL_REGISTRY")
        return base_model_type, MODEL_REGISTRY[base_model_type]

    if model_type in MODEL_REGISTRY:
        return model_type, MODEL_REGISTRY[model_type]

    raise ValueError(
        f"Unknown model type: '{model_type}'. "
        f"Available: {list(MODEL_REGISTRY.keys()) + list(WRAPPER_CONFIGS.keys())}"
    )


def _get_constructor_kwargs_spec(model_constructor: Callable) -> Tuple[set[str], bool]:
    """Return accepted named kwargs and whether constructor accepts **kwargs."""
    target = model_constructor.__init__ if inspect.isclass(model_constructor) else model_constructor
    signature = inspect.signature(target)

    accepted_params: set[str] = set()
    accepts_var_kwargs = False

    for name, param in signature.parameters.items():
        if name in {"self", "space"}:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            accepted_params.add(name)

    return accepted_params, accepts_var_kwargs


def _config_get(section: Any, key: str, default: Any = None) -> Any:
    """Get key from mapping-like or attribute-like config sections."""
    if isinstance(section, Mapping):
        return section.get(key, default)
    return getattr(section, key, default)


def _config_has(section: Any, key: str) -> bool:
    """Check key existence in mapping-like or attribute-like config sections."""
    if isinstance(section, Mapping):
        return key in section
    return hasattr(section, key)


def _config_items(section: Any) -> List[Tuple[str, Any]]:
    """Get key-value items from mapping-like or attribute-like config sections."""
    if isinstance(section, Mapping):
        return list(section.items())
    if hasattr(section, "__dict__"):
        return list(vars(section).items())
    return []


def _to_plain_python(value: Any) -> Any:
    """Recursively convert mapping/list config objects to plain Python containers."""
    if isinstance(value, Mapping):
        return {k: _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_python(v) for v in value)
    return value


def _validate_kwargs(
    kwargs: dict,
    model_type: str,
    accepted_params: set[str],
    accepts_var_kwargs: bool,
    source: str
) -> None:
    """Validate provided kwargs against constructor signature."""
    if accepts_var_kwargs:
        return

    unknown = sorted(set(kwargs.keys()) - accepted_params)
    if unknown:
        raise ValueError(
            f"Unexpected model kwargs in {source} for model type '{model_type}': {unknown}. "
            f"Accepted constructor kwargs: {sorted(accepted_params)}"
        )


def _extract_legacy_flat_model_kwargs(model_cfg: Any) -> dict:
    """Extract legacy flat model kwargs from model.* excluding reserved fields."""
    kwargs = {}
    for key, value in _config_items(model_cfg):
        if key in _RESERVED_MODEL_KEYS:
            continue
        kwargs[key] = _to_plain_python(value)
    return kwargs


def _extract_model_kwargs(
    config: Any,
    model_type: str,
    accepted_params: set[str],
    accepts_var_kwargs: bool
) -> dict:
    """
    Extract model kwargs from config.

    Priority:
      1) model.params (canonical)
      2) legacy flat model.* (excluding type/params)
    """
    if not _config_has(config, "model"):
        return {}

    model_cfg = _config_get(config, "model")
    if model_cfg is None:
        return {}

    if not _config_has(model_cfg, "type"):
        raise ValueError("Config is missing required `model.type`.")

    cfg_model_type = _config_get(model_cfg, "type")
    if cfg_model_type is None:
        raise ValueError("Config field `model.type` is set to None; expected a valid model type.")
    if str(cfg_model_type) != str(model_type):
        warnings.warn(
            f"create_model called with model_type='{model_type}' but config has model.type='{cfg_model_type}'. "
            "Using explicit function argument.",
            stacklevel=2
        )

    params = _config_get(model_cfg, "params")
    if params is not None:
        if not isinstance(params, Mapping):
            raise ValueError(f"`model.params` must be a mapping/dict, got {type(params)}")
        params_dict = _to_plain_python(params)

        legacy_flat = _extract_legacy_flat_model_kwargs(model_cfg)
        if legacy_flat:
            warnings.warn(
                f"`model.params` is present; ignoring legacy flat `model.*` keys: "
                f"{sorted(legacy_flat.keys())}",
                stacklevel=2
            )

        _validate_kwargs(
            kwargs=params_dict,
            model_type=model_type,
            accepted_params=accepted_params,
            accepts_var_kwargs=accepts_var_kwargs,
            source="config.model.params"
        )
        return params_dict

    legacy_kwargs = _extract_legacy_flat_model_kwargs(model_cfg)
    _validate_kwargs(
        kwargs=legacy_kwargs,
        model_type=model_type,
        accepted_params=accepted_params,
        accepts_var_kwargs=accepts_var_kwargs,
        source="legacy flat config model.*"
    )
    return legacy_kwargs


# Backward compatibility alias
create_score_model_new = create_model


def infer_space_from_data_dim(data_dim):
    """
    Infer Space object from legacy data_dim specification.

    Args:
        data_dim: int, tuple, or list specifying data dimensions

    Returns:
        Space object
    """
    from diffusion.core import VectorSpace, MatrixSpace

    if isinstance(data_dim, int):
        return VectorSpace(data_dim)
    elif isinstance(data_dim, (list, tuple)):
        if len(data_dim) == 1:
            return VectorSpace(data_dim[0])
        elif len(data_dim) == 2:
            return MatrixSpace(data_dim[0], data_dim[1])
        elif len(data_dim) == 3:
            raise NotImplementedError(f"Cannot infer space from data_dim={data_dim}")
        else:
            raise ValueError(f"Cannot infer space from data_dim={data_dim}")
    else:
        raise TypeError(f"data_dim must be int, tuple, or list, got {type(data_dim)}")
