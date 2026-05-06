#!/usr/bin/env python3
"""Checkpoint loading utilities for canonical split checkpoints."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml


@dataclass(frozen=True)
class PretrainedScoreContext:
    device: torch.device
    checkpoint_dir: Path
    dataset_path: Path
    dataset: Any
    train_data: torch.Tensor
    test_data: Optional[torch.Tensor]
    space: Any
    diffusion: Any
    model: torch.nn.Module
    constraint: Any
    checkpoint_data: Dict[str, Any]
    checkpoint_config: Dict[str, Any]
    checkpoint_metadata: Dict[str, Any]
    dataset_metadata: Dict[str, Any]


def load_model_checkpoint(
    model_dir: Path,
    checkpoint_name: str,
    device: str = "cpu",
) -> Tuple[Dict, Dict, Dict, Dict, Optional[Any]]:
    """
    Load a model checkpoint from canonical split files.

    Required files:
    1. model.pth
    2. checkpoint_data.pt
    3. config.yaml

    Args:
        model_dir: Directory containing model files
        checkpoint_name: Kept for API compatibility (canonical loading uses fixed filenames)
        device: Device to load tensors on

    Returns:
        (model_state_dict, checkpoint_data, config, metadata, space) where:
        - model_state_dict: Model weights (dict)
        - checkpoint_data: Training state (global_step, epoch, etc.)
        - config: Model configuration dict
        - metadata: Metadata dictionary loaded from metadata.json (if present)
        - space: Always None in paper repo (space reconstructed from dataset)
    """
    model_dir = Path(model_dir).expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_path = model_dir / "model.pth"
    checkpoint_data_path = model_dir / "checkpoint_data.pt"
    config_path = model_dir / "config.yaml"
    missing = []
    if not model_path.exists():
        missing.append(str(model_path))
    if not checkpoint_data_path.exists():
        missing.append(str(checkpoint_data_path))
    if not config_path.exists():
        missing.append(str(config_path))

    if missing:
        raise FileNotFoundError(
            "Canonical split checkpoint format is required (model.pth + checkpoint_data.pt + config.yaml). "
            f"Missing files: {', '.join(missing)}. "
            f"Requested checkpoint_name='{checkpoint_name}' is not used for fallback."
        )

    print("Loading canonical format (model.pth + checkpoint_data.pt + config.yaml)")
    model_state_dict = torch.load(model_path, map_location=device)
    checkpoint_data = torch.load(checkpoint_data_path, map_location=device, weights_only=False)
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config.yaml at {config_path}: expected a YAML mapping.")

    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            metadata = {}
    else:
        metadata = {}
    space = None

    if "current_epoch" in checkpoint_data or "global_step" in checkpoint_data:
        epoch = checkpoint_data.get("current_epoch", "N/A")
        step = checkpoint_data.get("global_step", "N/A")
        print(f"Progress: epoch={epoch}, global_step={step}")

    return model_state_dict, checkpoint_data, config, metadata, space


def resolve_dataset_path(
    *,
    dataset_path_override: str | Path | None,
    cli_config: Optional[Dict[str, Any]],
    checkpoint_config: Optional[Dict[str, Any]],
) -> Path:
    """Resolve dataset path with precedence: explicit override > CLI config > checkpoint config."""
    if dataset_path_override:
        return Path(dataset_path_override).expanduser()

    if isinstance(cli_config, dict):
        cli_dataset = cli_config.get("dataset_path")
        if cli_dataset:
            return Path(cli_dataset).expanduser()

    if isinstance(checkpoint_config, dict):
        ckpt_dataset_cfg = checkpoint_config.get("dataset")
        if isinstance(ckpt_dataset_cfg, dict):
            ckpt_dataset = ckpt_dataset_cfg.get("path")
            if ckpt_dataset:
                return Path(ckpt_dataset).expanduser()

        ckpt_dataset_root = checkpoint_config.get("dataset_path")
        if ckpt_dataset_root:
            return Path(ckpt_dataset_root).expanduser()

    raise ValueError(
        "Dataset path could not be resolved. Provide --dataset-path, or include "
        "`dataset_path` in CLI config, or `dataset.path`/`dataset_path` in checkpoint config."
    )


def _freeze_model(model: torch.nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def load_pretrained_score_context(
    *,
    checkpoint_dir: str | Path,
    device: str | torch.device = "cpu",
    checkpoint_name: str = "checkpoint_data.pt",
    dataset_path_override: str | Path | None = None,
    cli_config: Optional[Dict[str, Any]] = None,
    reconstruct_constraint: bool = True,
    require_trajectory_data: bool = False,
) -> PretrainedScoreContext:
    """
    Load pretrained score model and reconstruct dataset/space/diffusion context.

    This is the canonical runtime loader for optimization and control workflows.
    """
    from diffusion.data import load_dataset
    from diffusion.models import create_model
    from diffusion.training import create_diffusion

    resolved_device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()

    (
        model_state_dict,
        checkpoint_data,
        checkpoint_config,
        checkpoint_metadata,
        _,
    ) = load_model_checkpoint(
        model_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        device=str(resolved_device),
    )
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Checkpoint at {checkpoint_dir} does not contain a valid config dictionary.")

    dataset_path = resolve_dataset_path(
        dataset_path_override=dataset_path_override,
        cli_config=cli_config,
        checkpoint_config=checkpoint_config,
    )
    dataset = load_dataset(
        dataset_path,
        reconstruct_constraint=bool(reconstruct_constraint),
        device=str(resolved_device),
    )

    train_data = dataset.train_data.to(resolved_device)
    test_data = dataset.test_data.to(resolved_device) if dataset.test_data is not None else None
    space = dataset.get_space()

    model_cfg = checkpoint_config.get("model")
    model_type = model_cfg.get("type") if isinstance(model_cfg, dict) else None
    if not model_type:
        raise ValueError(
            f"Checkpoint config at {checkpoint_dir} is missing required `model.type`."
        )

    diffusion = create_diffusion(space, checkpoint_config)
    model = create_model(
        model_type=str(model_type),
        space=space,
        diffusion=diffusion,
        config=checkpoint_config,
        initialize=False,
    ).to(resolved_device)
    model.load_state_dict(model_state_dict)
    _freeze_model(model)

    if require_trajectory_data and train_data.ndim != 3:
        raise ValueError(
            "Trajectory data expected for control workflow, but train_data has shape "
            f"{tuple(train_data.shape)}."
        )

    return PretrainedScoreContext(
        device=resolved_device,
        checkpoint_dir=checkpoint_dir,
        dataset_path=dataset_path,
        dataset=dataset,
        train_data=train_data,
        test_data=test_data,
        space=space,
        diffusion=diffusion,
        model=model,
        constraint=dataset.constraint,
        checkpoint_data=checkpoint_data,
        checkpoint_config=checkpoint_config,
        checkpoint_metadata=checkpoint_metadata if isinstance(checkpoint_metadata, dict) else {},
        dataset_metadata=dataset.metadata if isinstance(dataset.metadata, dict) else {},
    )
