#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml


ASSET_SPECS = (
    {
        "name": "stiefel",
        "dataset": Path("data/stiefel_n3_p3.pt"),
        "checkpoint_dir": Path("outputs/stiefel_train_demo"),
    },
    {
        "name": "unicycle",
        "dataset": Path("data/unicycle_T100_n20000.pt"),
        "checkpoint_dir": Path("outputs/unicycle_train_demo"),
    },
    {
        "name": "robot_arm",
        "dataset": Path("data/robot_arm_T100_n20000.pt"),
        "checkpoint_dir": Path("outputs/robot_arm_train_demo"),
    },
)

CHECKPOINT_FILES = (
    "model.pth",
    "checkpoint_data.pt",
    "config.yaml",
    "metadata.json",
    "train_summary.json",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "release_assets" / "v1.0.0"

    parser = argparse.ArgumentParser(
        description="Sanitize and stage GitHub Release assets for v1 publication."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help="Directory where staged release assets will be written.",
    )
    parser.add_argument(
        "--release-tag",
        type=str,
        default="v1.0.0",
        help="Version string used in the generated release notes.",
    )
    return parser.parse_args()


def sanitize_train_summary(path: Path, checkpoint_dir: Path, dataset_path: Path) -> None:
    data = json.loads(path.read_text())
    data["output_dir"] = str(checkpoint_dir)
    data["dataset_path"] = str(dataset_path)
    path.write_text(json.dumps(data, indent=2) + "\n")


def sanitize_config(path: Path, dataset_path: Path) -> None:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}.")
    data.setdefault("dataset", {})
    data["dataset"]["path"] = str(dataset_path)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def sanitize_in_place(repo_root: Path) -> None:
    for spec in ASSET_SPECS:
        checkpoint_dir = repo_root / spec["checkpoint_dir"]
        dataset_path = spec["dataset"]
        sanitize_train_summary(
            checkpoint_dir / "train_summary.json",
            checkpoint_dir=spec["checkpoint_dir"],
            dataset_path=dataset_path,
        )
        sanitize_config(
            checkpoint_dir / "config.yaml",
            dataset_path=dataset_path,
        )


def stage_dataset(repo_root: Path, stage_dir: Path, dataset_rel: Path) -> None:
    src = repo_root / dataset_rel
    dst = stage_dir / dataset_rel.name
    shutil.copy2(src, dst)


def stage_checkpoint_bundle(repo_root: Path, stage_dir: Path, checkpoint_rel: Path) -> None:
    checkpoint_dir = repo_root / checkpoint_rel
    archive_path = stage_dir / f"{checkpoint_rel.name}.zip"
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for filename in CHECKPOINT_FILES:
            src = checkpoint_dir / filename
            zf.write(src, arcname=f"{checkpoint_rel.name}/{filename}")


def write_release_notes(stage_dir: Path, release_tag: str) -> None:
    notes = f"""# {release_tag} assets

Datasets and pretrained checkpoints for Stiefel, Unicycle, and Robot Arm.

Assets:
- `stiefel_n3_p3.pt`
- `stiefel_train_demo.zip`
- `unicycle_T100_n20000.pt`
- `unicycle_train_demo.zip`
- `robot_arm_T100_n20000.pt`
- `robot_arm_train_demo.zip`

Load:
```python
from diffusion.utils.checkpoint_utils import load_pretrained_score_context

ctx = load_pretrained_score_context(
    checkpoint_dir="outputs/unicycle_train_demo",
    dataset_path_override="data/unicycle_T100_n20000.pt",
    device="cpu",
    require_trajectory_data=True,
)
```

Compatibility: intended for repository version `{release_tag}`.
"""
    (stage_dir / "RELEASE_NOTES.md").write_text(notes)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    stage_dir = Path(args.output_dir).expanduser()
    if not stage_dir.is_absolute():
        stage_dir = (repo_root / stage_dir).resolve()

    stage_dir.mkdir(parents=True, exist_ok=True)

    sanitize_in_place(repo_root)

    for spec in ASSET_SPECS:
        stage_dataset(repo_root, stage_dir, spec["dataset"])
        stage_checkpoint_bundle(repo_root, stage_dir, spec["checkpoint_dir"])

    write_release_notes(stage_dir, args.release_tag)

    print(f"Staged release assets in {stage_dir}")
    for path in sorted(stage_dir.iterdir()):
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
