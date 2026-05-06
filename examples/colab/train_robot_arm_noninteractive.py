#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from diffusion.control.mujoco import MuJoCoSystem
from diffusion.control.trajectory import DynamicsConstraint
from diffusion.data import load_dataset
from diffusion.models import create_model
from diffusion.training import DiffusionTrainer, TrainingOptions, create_diffusion, create_optimizer


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Train the robot arm score model from existing dataset/config (non-interactive)."
    )
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/train_robot_arm_quick.yaml",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/robot_arm_T100_n20000.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/robot_arm_train_demo",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to MuJoCo XML model (e.g. examples/colab/robot_arm_2dof.xml).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    repo_root = Path.cwd().resolve()

    def resolve_path(path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()

    device = args.device
    config_path = resolve_path(args.config_path)
    dataset_path = resolve_path(args.dataset_path)
    output_dir = resolve_path(args.output_dir)
    model_path = resolve_path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = load_dataset(str(dataset_path), device="cpu", reconstruct_constraint=False)
    space = dataset.get_space()
    diffusion = create_diffusion(space, cfg)
    model = create_model(
        model_type=str(cfg["model"]["type"]),
        space=space,
        diffusion=diffusion,
        config=cfg,
        initialize=True,
    ).to(device)
    optimizer = create_optimizer(model, cfg)

    options_dict = dict(cfg.get("training_options", {}))
    options_dict["batch_size"] = int(cfg.get("training", {}).get("batch_size", 32))
    options = TrainingOptions(**options_dict)

    constraint_params = dataset.metadata["constraint"]["params"]
    dT = float(constraint_params["dT"])
    Ts = float(constraint_params["Ts"])
    system = MuJoCoSystem(model_path=str(model_path), dT=dT, Ts=Ts)
    constraint = DynamicsConstraint.from_system(system, horizon=int(space.horizon))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def log_fn(samples):
        violations = constraint.violation(samples)
        return {
            "constraint_violation_mean": violations.mean().item(),
            "constraint_violation_std": violations.std().item(),
            "constraint_violation_max": violations.max().item(),
        }

    try:
        model_path_for_metadata = str(model_path.relative_to(repo_root))
    except ValueError:
        model_path_for_metadata = str(model_path)
    metadata = dict(dataset.metadata)
    metadata_constraint = dict(metadata.get("constraint", {}))
    metadata_constraint_params = dict(metadata_constraint.get("params", {}))
    metadata_constraint_params["model_path"] = model_path_for_metadata
    metadata_constraint["params"] = metadata_constraint_params
    metadata["constraint"] = metadata_constraint

    trainer = DiffusionTrainer(
        diffusion=diffusion,
        score_model=model,
        train_data=dataset.train_data,
        optimizer=optimizer,
        options=options,
        log_fn=log_fn,
        device=device,
        output_dir=output_dir,
        metadata=metadata,
    )

    num_epochs = int(cfg.get("training", {}).get("num_epochs", 50000))
    trainer.train(num_epochs=num_epochs)
    trainer.save_checkpoint(str(output_dir / "checkpoint.pt"))

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "num_epochs": num_epochs,
        "final_loss": float(trainer.train_losses[-1]) if trainer.train_losses else None,
        "num_params": num_params,
        "dataset_path": str(dataset_path),
        "model_type": str(cfg["model"]["type"]),
    }
    with open(output_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Training complete. Artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
