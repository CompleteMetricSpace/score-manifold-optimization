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

from diffusion.data import load_dataset
from diffusion.models import create_model
from diffusion.training import DiffusionTrainer, TrainingOptions, create_diffusion, create_optimizer

# Compatibility stubs for unpickling old dataset metadata.
# They are not used for training; only needed so torch.load can resolve names.
def input_generator(n):
    return None

def x0_sampler(n):
    return None


def sanitize_for_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if callable(obj):
        name = getattr(obj, "__name__", obj.__class__.__name__)
        return f"<callable:{name}>"
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parents[2]
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Train the Unicycle score model from existing dataset/config (non-interactive)."
    )
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(repo_dir / "configs" / "train_unicycle_quick.yaml"),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(repo_dir / "data" / "unicycle_T100_n20000.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_dir / "outputs" / "unicycle_train_demo"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = args.device
    config_path = Path(args.config_path).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = load_dataset(str(dataset_path), device="cpu")
    safe_metadata = sanitize_for_json(dataset.metadata)
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

    constraint = dataset.constraint
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def log_fn(samples):
        violations = constraint.violation(samples)
        return {
            "constraint_violation_mean": violations.mean().item(),
            "constraint_violation_std": violations.std().item(),
            "constraint_violation_max": violations.max().item(),
        }

    trainer = DiffusionTrainer(
        diffusion=diffusion,
        score_model=model,
        train_data=dataset.train_data,
        optimizer=optimizer,
        options=options,
        log_fn=log_fn,
        device=device,
        output_dir=output_dir,
        metadata=safe_metadata,
    )

    num_epochs = int(cfg.get("training", {}).get("num_epochs", 80))
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
