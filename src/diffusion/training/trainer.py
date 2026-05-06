#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified trainer for diffusion score models.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from diffusion.core import DiffusionProcess, sde_euler_maruyama_scalar
from diffusion.models import ScoreModel
from .losses import denoising_score_matching_loss, stratified_uniform
from .options import TrainingOptions


class _TrainerCheckpointMixin:
    """Shared checkpoint IO helpers for trainer classes."""

    def _get_managed_model(self) -> nn.Module:
        raise NotImplementedError

    def save_checkpoint(self, path: str, **kwargs):
        """
        Save training checkpoint as three separate files:
        - model.pth: Model weights only
        - checkpoint_data.pt: Training state
        - metadata.json: Metadata snapshot (refreshed each periodic save)

        Args:
            path: Base path for checkpoint (e.g., 'output/checkpoint.pt')
                  Three files will be created:
                  - {path.parent}/model.pth
                  - {path.parent}/checkpoint_data.pt
                  - {path.parent}/metadata.json (if metadata available)
            **kwargs: Additional items to save in checkpoint_data.pt
        """
        import json

        path = Path(path)
        output_dir = path.parent

        managed_model = self._get_managed_model()
        model_to_save = self._unwrap_model(managed_model)
        timestamp = datetime.now().isoformat()
        dataset_metadata = self.metadata or {}

        # Save model weights separately
        model_path = output_dir / "model.pth"
        torch.save(model_to_save.state_dict(), model_path)

        # Save checkpoint data WITHOUT model weights
        checkpoint_data = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "options": self.options,
            "timestamp": timestamp,
            **kwargs,
        }

        if self.scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_data_path = output_dir / "checkpoint_data.pt"
        torch.save(checkpoint_data, checkpoint_data_path)

        metadata_path = output_dir / "metadata.json"

        def make_json_serializable(obj):
            """Convert tensors and tuples/lists/dicts recursively to JSON-safe values."""
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            return obj

        metadata_dict = {
            "model_class": model_to_save.__class__.__name__,
            "num_params": self.get_num_parameters(),
            "space": str(getattr(model_to_save, "space", "unknown")),
            "dataset_metadata": dataset_metadata,
            "timestamp": timestamp,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        with open(metadata_path, "w") as f:
            json.dump(make_json_serializable(metadata_dict), f, indent=2)

        if self.options.rank == 0:
            print(f"Metadata saved to {metadata_path}")
            print(f"Model weights saved to {model_path}")
            print(f"Checkpoint data saved to {checkpoint_data_path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """
        Load training checkpoint from separate files:
        - model.pth: Model weights
        - checkpoint_data.pt: Training state

        Supports backward compatibility with old single-file checkpoints.

        Args:
            path: Base path for checkpoint (e.g., 'output/checkpoint.pt' or 'output/checkpoint_data.pt')
                  Will look for:
                  - {path.parent}/model.pth
                  - {path.parent}/checkpoint_data.pt
            load_optimizer: Whether to load optimizer state
        """
        path = Path(path)
        output_dir = path.parent

        model_path = output_dir / "model.pth"
        checkpoint_data_path = output_dir / "checkpoint_data.pt"
        target_model = self._unwrap_model(self._get_managed_model())

        if model_path.exists() and checkpoint_data_path.exists():
            if self.options.rank == 0:
                print("Loading checkpoint (new format with separate files)...")

            model_state_dict = torch.load(model_path, map_location=self.device)
            target_model.load_state_dict(model_state_dict)
            checkpoint = torch.load(checkpoint_data_path, map_location=self.device, weights_only=False)
        else:
            if self.options.rank == 0:
                print("Loading checkpoint (old format with embedded weights)...")

            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            if "model_state_dict" in checkpoint:
                target_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                raise ValueError(f"Checkpoint at {path} does not contain 'model_state_dict'")

        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_loss = checkpoint["best_loss"]

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.options.rank == 0:
            print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def get_num_parameters(self) -> int:
        """Return number of trainable parameters."""
        model = self._unwrap_model(self._get_managed_model())
        if hasattr(model, "get_num_parameters"):
            return model.get_num_parameters()
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Return underlying model when wrapped by DDP-style containers."""
        return model.module if hasattr(model, "module") else model


class DiffusionTrainer(_TrainerCheckpointMixin):
    """
    Unified trainer for (DiffusionProcess, ScoreModel) pairs.

    This trainer works with any compatible combination of:
    - DiffusionProcess (VP, VE, etc.)
    - ScoreModel (MLP, Transformer, UNet, etc.)
    - Space (VectorSpace, MatrixSpace, TrajectorySpace, etc.)

    Features:
    - Denoising score matching loss
    - Optional log_fn for sample-metric logging on generated samples
    - Distributed training support (DDP)
    - Flexible logging and checkpointing
    - Gradient clipping
    - Stratified time sampling

    Example:
        >>> from diffusion.core import VPDiffusion, VectorSpace
        >>> from diffusion.models import MLPScoreModel
        >>> from diffusion.training import DiffusionTrainer, TrainingOptions
        >>>
        >>> space = VectorSpace(50)
        >>> diffusion = VPDiffusion(0.1, 20.0, 1.0, space)
        >>> model = MLPScoreModel(space, hidden_dim=256)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> options = TrainingOptions(t_start=1e-5, t_stop=1.0)
        >>>
        >>> trainer = DiffusionTrainer(diffusion, model, train_data, optimizer, options)
        >>> trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        diffusion: DiffusionProcess,
        score_model: ScoreModel,
        train_data: Dataset,
        optimizer: torch.optim.Optimizer,
        options: TrainingOptions,
        log_fn: Optional[Callable[[torch.Tensor], Any]] = None,
        device: str = "cpu",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[Callable] = None,
        output_dir: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize trainer.

        Args:
            diffusion: DiffusionProcess instance
            score_model: ScoreModel instance
            train_data: PyTorch Dataset
            optimizer: PyTorch optimizer
            options: TrainingOptions configuration
            log_fn: Optional callable applied to generated sample batches for logging
            device: Device to train on ('cpu', 'cuda', 'cuda:0', etc.)
            scheduler: Optional learning rate scheduler
            logger: Optional logging function (e.g., wandb.log, tensorboard)
            output_dir: Optional output directory for checkpoints
            metadata: Optional dataset metadata (for checkpoint saving)
        """
        self.diffusion = diffusion
        self.score_model = score_model
        self.train_data = train_data
        self.optimizer = optimizer
        self.options = options
        self.log_fn = log_fn
        self.device = device
        self.scheduler = scheduler
        self.logger = logger
        self.output_dir = output_dir
        self.metadata = metadata

        # Keep explicit references to wrapped (training) and unwrapped (attribute access) models.
        self.base_model = self._unwrap_model(self.score_model)

        # Validate space compatibility
        assert diffusion.space == self.base_model.space, (
            f"Space mismatch: diffusion={diffusion.space}, model={self.base_model.space}"
        )

        # Move model to device only if not already wrapped (DDP wrapping owns device placement).
        if self.base_model is self.score_model:
            self.score_model = self.score_model.to(device)
            self.base_model = self._unwrap_model(self.score_model)

        # Create data loader (DDP uses DistributedSampler for disjoint shards)
        self.train_sampler = None
        if self.options.ddp:
            self.train_sampler = DistributedSampler(
                train_data,
                num_replicas=self.options.world_size,
                rank=self.options.rank,
                shuffle=True,
            )

        self.train_loader = DataLoader(
            train_data,
            batch_size=getattr(options, "batch_size", 32),
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=getattr(options, "num_workers", 0),
            pin_memory=(device != "cpu"),
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")

        # Statistics
        self.train_losses = []

    def _get_managed_model(self) -> nn.Module:
        return self.score_model

    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        Train for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)
        """
        self.score_model.train()

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Ensure each DDP rank uses a consistent-but-distinct shuffle per epoch.
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            epoch_loss = self.train_epoch()

            # Log epoch statistics
            if self.options.rank == 0:
                print(f"Epoch {epoch}/{num_epochs} | Loss: {epoch_loss:.6f}")

                if self.logger is not None:
                    self.logger({"epoch": epoch, "train_loss": epoch_loss})

            # Sample-metric evaluation via optional log_fn (every N epochs)
            if (
                self.options.rank == 0
                and epoch % self.options.eval_interval == 0
                and self.log_fn is not None
                and getattr(self.options, "log_sample_metrics", True)
            ):
                log_output = self.evaluate_log_fn_on_samples()

                if isinstance(log_output, dict):
                    print(f"Epoch {epoch} | Sample metrics: {log_output}")
                    if self.logger is not None:
                        self.logger({"epoch": epoch, **log_output})
                else:
                    print(f"Epoch {epoch} | log_fn output: {log_output}")
                    if self.logger is not None:
                        self.logger({"epoch": epoch, "log_fn_output": log_output})

            # Save checkpoint (every N epochs)
            if (
                self.options.rank == 0
                and epoch % self.options.save_interval == 0
                and self.output_dir is not None
            ):
                checkpoint_path = self.output_dir / "checkpoint.pt"
                self.save_checkpoint(checkpoint_path)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average loss for the epoch
        """
        epoch_losses = []

        for batch in self.train_loader:
            # Move data to device
            if isinstance(batch, (list, tuple)):
                x0 = batch[0].to(self.device)
            else:
                x0 = batch.to(self.device)

            # Compute loss
            loss = self.train_step(x0)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.options.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.score_model.parameters(), self.options.clip_value)

            # Optimizer step
            self.optimizer.step()

            # Record loss
            epoch_losses.append(loss.item())
            self.train_losses.append(loss.item())
            self.global_step += 1

        return sum(epoch_losses) / len(epoch_losses)

    def train_step(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Single training step (compute loss).

        Args:
            x0: (batch, *space.dims) clean data samples

        Returns:
            loss: Scalar loss value
        """
        batch_size = x0.shape[0]

        # Sample times
        if self.options.stratified:
            t = stratified_uniform(
                self.options.t_start,
                self.options.t_stop,
                batch_size,
                device=self.device,
            )
        else:
            t = torch.rand(batch_size, device=self.device)
            t = t * (self.options.t_stop - self.options.t_start) + self.options.t_start

        # Compute denoising score matching loss
        return denoising_score_matching_loss(
            self.score_model,
            self.diffusion,
            x0,
            t,
            weight_by_var=self.options.weight_by_var,
            reduction="mean",
        )

    @torch.no_grad()
    def evaluate(self, eval_data: Optional[Dataset] = None, num_samples: int = None) -> Dict[str, float]:
        """
        Evaluate model on validation data.

        Args:
            eval_data: Optional evaluation dataset (uses train_data if None)
            num_samples: Number of samples to evaluate (uses options.num_eval_samples if None)

        Returns:
            Dictionary of evaluation metrics
        """
        self.score_model.eval()

        if eval_data is None:
            eval_data = self.train_data

        if num_samples is None:
            num_samples = self.options.num_eval_samples

        eval_loader = DataLoader(
            eval_data,
            batch_size=self.options.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

        losses = []
        for i, batch in enumerate(eval_loader):
            if i * self.options.eval_batch_size >= num_samples:
                break

            if isinstance(batch, (list, tuple)):
                x0 = batch[0].to(self.device)
            else:
                x0 = batch.to(self.device)

            batch_size = x0.shape[0]
            t = stratified_uniform(
                self.options.t_start,
                self.options.t_stop,
                batch_size,
                device=self.device,
            )

            loss = denoising_score_matching_loss(
                self.score_model,
                self.diffusion,
                x0,
                t,
                weight_by_var=self.options.weight_by_var,
                reduction="mean",
            )
            losses.append(loss.item())

        metrics = {"eval_loss": sum(losses) / len(losses)}

        self.score_model.train()
        return metrics

    @torch.no_grad()
    def evaluate_log_fn_on_samples(self, num_samples: int = None, num_steps: int = None) -> Any:
        """
        Evaluate log_fn on generated samples.

        Args:
            num_samples: Number of samples to generate (uses options.eval_log_fn_samples if None)
            num_steps: Number of sampling steps (uses options.eval_log_fn_steps if None)

        Returns:
            Whatever object log_fn returns
        """
        if self.log_fn is None:
            return None

        if num_samples is None:
            num_samples = getattr(self.options, "eval_log_fn_samples", 50)
        if num_steps is None:
            num_steps = self.options.eval_log_fn_steps

        samples = self.sample(num_samples=num_samples, num_steps=num_steps)
        return self.log_fn(samples)

    @torch.no_grad()
    def sample(self, num_samples: int, num_steps: int = 1000, flow_type: str = "SDE") -> torch.Tensor:
        """
        Generate samples using reverse-time SDE/ODE.

        Args:
            num_samples: Number of samples to generate
            num_steps: Number of discretization steps
            flow_type: 'SDE' or 'ODE' (probability flow)

        Returns:
            samples: (num_samples, *space.dims) generated samples
        """
        self.score_model.eval()

        reverse_sde = self.diffusion.get_reverse_sde(self.score_model, flow_type=flow_type)
        x_T = self.diffusion.sample_prior(num_samples, device=self.device)

        trajectory, _times = sde_euler_maruyama_scalar(
            f=reverse_sde.f_drift,
            g=reverse_sde.g_diff,
            n_grid_pts=num_steps,
            t_start=reverse_sde.t_start,
            t_end=reverse_sde.t_end,
            init_list=x_T,
            device=self.device,
            end_only=False,
        )

        samples = trajectory[..., -1]

        self.score_model.train()
        return samples
