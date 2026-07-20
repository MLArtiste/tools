import warnings
from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Optimizer
from torchmetrics.metric import Metric
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler


from ._checkpointer import Checkpointer


class BaseNNTrainer(ABC):
    """
    Abstract base class for training neural networks in PyTorch.

    Args:
        model (nn.Module): Neural network to train.
        optimizer (Optimizer): Optimization algorithm for parameter updates.
        loss_fn (nn.Module): Loss function to minimize during training.
        metrics (dict[str, Metric] or None): Optional dictionary of metrics to track. Defaults to None.
        scheduler (LRScheduler or None): Optional learning rate scheduler. Defaults to None.
        lrs_metric (str): Metric name to monitor for ReduceLROnPlateau. Defaults to 'val_loss'.
        device (torch.device, str or None): Optional device to use. Defaults to None.
        checkpoint_path (str, Path or None): Optional file path for checkpointing based on best metric value.
        Trainer will checkpoint only when provided. Defaults to None.
        checkpoint_steps (int or None): Optional number of training steps between checkpoints
        for very long training epochs. Defaults to None.
        checkpoint_metric (str): name of metric to monitor. Expects one of keys in trainer history.
        Defaults to 'val_loss'.
        minimize_metric (bool): Whether to minimize or maximize the metric.
        Defaults to True.
        patience (int or None): Optional early stopping patience. No early stopping if None.
        Defaults to None.
        use_amp (bool): Whether to use automatic mixed precision. Defauls to True.
        grad_clip_val (float or None): Optional maximum gradient norm for clipping. Defaults to None.
        grad_accum_steps (int): Number of batches to accumulate gradients over
        before performing an optimizer step. Defaults to 1 (no accumulation).

    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        metrics: dict[str, Metric] | None = None,
        scheduler: LRScheduler | None = None,
        lrs_metric: str = "val_loss",
        device: torch.device | str | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_steps: int | None = None,
        checkpoint_metric: str = "val_loss",
        minimize_metric: bool = True,
        patience: int | None = None,
        use_amp: bool = True,
        grad_clip_val: float | None = None,
        grad_accum_steps: int = 1,
    ):
        if checkpoint_steps is not None:
            if checkpoint_steps < 1 or not isinstance(checkpoint_steps, int):
                raise ValueError("checkpoint_steps must be a positive integer")
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path must be provided when checkpoint_steps is not None"
                )

        if grad_accum_steps < 1 or not isinstance(grad_accum_steps, int):
            raise ValueError("grad_accum_steps must be a positive integer")

        if grad_clip_val is not None and grad_clip_val <= 0:
            raise ValueError("grad_clip_val must be positive")

        if torch.accelerator.device_count() > 1:
            warnings.warn(
                "Multiple-device training is unsupported. Using a single device instead."
            )

        if device is None:
            self.device = torch.device(
                torch.accelerator.current_accelerator()
                if torch.accelerator.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)

        model.to(self.device)
        self.model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._scheduler = scheduler
        self._lrs_metric = lrs_metric.lower()
        self._checkpoint_steps = checkpoint_steps
        self._train_steps = 0
        self._checkpoint_path = checkpoint_path
        self._checkpointer = None
        self._checkpoint_metric = checkpoint_metric.lower()
        self._minimize_metric = minimize_metric
        self._patience = patience
        self._es_counter = 0
        self._use_amp = use_amp
        self._grad_clip_val = grad_clip_val
        self._grad_accum_steps = grad_accum_steps
        self._train_loader_state_dict = None
        self._history = {"train_loss": [], "val_loss": []}
        self._best_metric_val = float("inf") if self._minimize_metric else -float("inf")
        self._scaler = torch.amp.GradScaler(self.device.type, enabled=self._use_amp)
        self._metrics = {
            name.lower(): metric for name, metric in (metrics or {}).items()
        }

        if self._metrics:
            for metric_name in self._metrics:
                self._history[f"train_{metric_name}"] = []
                self._history[f"val_{metric_name}"] = []

            for metric in self._metrics.values():
                metric.to(self.device)

        if self._checkpoint_path is not None:
            self._checkpoint_path = Path(self._checkpoint_path)
            best_filename = (
                f"{self._checkpoint_path.stem}_best{self._checkpoint_path.suffix}"
            )
            self._best_checkpoint_path = self._checkpoint_path.with_name(best_filename)

        if self._checkpoint_metric not in self._history:
            raise ValueError(f"""invalid checkpoint metric '{self._checkpoint_metric}'. 
                Expected one of {list(self._history.keys())}""")

        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self._lrs_metric not in self._history:
                raise ValueError(
                    f"""invalid learning rate scheduler metric '{self._lrs_metric}'. 
                    Expected one of {list(self._history.keys())}"""
                )

    def _update_metrics(self, *args: Any) -> None:
        """
        Update metric values using model predictions and ground truths.

        Args:
            *args (Any): Values returned from forward_step.
        """
        for metric in self._metrics.values():
            metric.update(*args)

    def _compute_metrics(self) -> dict[str, float]:
        """
        Compute and return the current values of all metrics.

        Returns:
            dict[str, float]: Mapping of metric names to their computed values.
        """
        return {name: metric.compute().item() for name, metric in self._metrics.items()}

    def _reset_metrics(self) -> None:
        """
        Reset all metric states to begin new accumulation for the next epoch.
        """
        for metric in self._metrics.values():
            metric.reset()

    def _scheduler_step(self) -> None:
        """
        Step the learning rate scheduler.
        Use a monitored metric for ReduceLROnPlateau.
        """
        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric_value = self._history[self._lrs_metric][-1]
            self._scheduler.step(metric_value)
        else:
            self._scheduler.step()

    def _check_improvement(self) -> bool:
        """
        Check if the current monitored metric has improved.

        Returns:
            (bool): Whether metric improved.
        """
        hist = self._history[self._checkpoint_metric]

        if not hist:
            return False

        cur_val = hist[-1]

        is_better = (self._minimize_metric and cur_val < self._best_metric_val) or (
            not self._minimize_metric and cur_val > self._best_metric_val
        )
        if is_better:
            self._best_metric_val = cur_val
            return True
        return False

    def _checkpoint_extra(self) -> dict[str, Any]:
        """
        Return additional objects to merge into the checkpoint at save time.
        Useful for immutable objects like integers.

        Returns:
            dict[str, Any]: Additional objects to save.
        """
        return {"es_counter": self._es_counter, "train_steps": self._train_steps}

    def _checkpoint(
        self,
        improved: bool,
        verbose: bool = True,
    ):
        """
        checkpoint training progress.

        Args:
            improved (bool): Whether the monitored metric has improved.
            verbose (bool): Whether to show checkpointing detail. Defaults to True.
        """
        if improved:
            if verbose:
                print(
                    f"Saving best checkpoint at "
                    f"{self._checkpoint_metric} = {self._best_metric_val:.4f}"
                )
            self._checkpointer.sync_save(
                self._best_checkpoint_path, extra=self._checkpoint_extra()
            )

        if self.device.type == "cpu":
            self._checkpointer.sync_save(
                self._checkpoint_path, extra=self._checkpoint_extra()
            )

        else:
            self._checkpointer.async_save(
                self._checkpoint_path, extra=self._checkpoint_extra()
            )

    def _load_checkpoint(self) -> int:
        """
        Load checkpoint from checkpoint_path.

        Returns:
            int: The next epoch index to start from.
        """
        if not self._checkpoint_path or not Path(self._checkpoint_path).exists():
            raise FileNotFoundError(
                f"no checkpoint found at {self._checkpoint_path} to resume from"
            )
        checkpoint = torch.load(
            self._checkpoint_path, map_location=self.device, weights_only=True
        )
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint is missing 'model_state_dict'")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self._history = checkpoint.get("history", self._history)
        if self._history is not None:
            metric = self._history[self._checkpoint_metric]
            self._best_metric_val = (
                min(metric) if self._minimize_metric else max(metric)
            )
        self._es_counter = checkpoint.get("es_counter", 0)
        self._train_steps = checkpoint.get("train_steps", 0)

        if self._use_amp and "scaler_state_dict" in checkpoint:
            self._scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self._scheduler and "scheduler_state_dict" in checkpoint:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "train_loader_state_dict" in checkpoint:
            self._train_loader_state_dict = checkpoint["train_loader_state_dict"]

        return len(self._history.get("train_loss", []))

    def _early_stopping(self, improved, verbose=True):
        """
        Check if early stopping should be triggered.

        Args:
            improved (bool): Whether the monitored metric has improved.
            verbose (bool): Whether to show early stopping detail. Defaults to True.

        Returns:
            bool: Whether early stopping should be triggered.
        """
        if improved:
            self._es_counter = 0
        else:
            self._es_counter += 1
            if verbose:
                print(f"Patience: {self._es_counter}/{self._patience}.")

        if self._es_counter >= self._patience:
            return True

        return False

    @abstractmethod
    def forward_step(self, batch_data: Any) -> Any:
        """
        Forward pass for a single batch. Should move data to self.device,
        pass input through self.model and return object (typically (y_pred, y))
        for loss computation and metric updates.

        Args:
            batch_data (Any): A batch from the dataloader.

        Returns:
            Any: Input to loss function and metrics.
            Typically model predictions and corresponding targets (y_pred, y).
        """

    def get_history(self) -> dict[str, list[float]]:
        """
        Retrieve the training and validation history of losses and metrics.

        Returns:
            dict[str, list[float]]: Recorded loss and metric values per epoch.
        """
        return self._history

    def _train_loop(
        self, train_dataloader: DataLoader, epoch: int, epochs: int, verbose=True
    ) -> None:
        """
        Perform one training epoch.

        Args:
            train_dataloader (DataLoader): Dataloader for the training set.
            epoch (int): current training epoch.
            epochs (int): total number of training epochs.
            verbose (bool): keep progress bars (except final progress bar) after they complete.
            Defaults to True.
        """
        self.model.train()
        total_loss = 0
        num_batches = len(train_dataloader)
        leave = verbose or (epoch + 1 == epochs)
        initial_batch = 0

        if self._train_loader_state_dict is not None:
            snapshot = self._train_loader_state_dict.get("_snapshot", {})
            initial_batch = snapshot.get("_snapshot_step", 0)
            self._train_loader_state_dict = None

        pbar = tqdm(
            train_dataloader,
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=leave,
            initial=initial_batch,
        )

        for batch_idx, batch_data in enumerate(pbar, start=initial_batch):
            with torch.autocast(self.device.type, enabled=self._use_amp):
                batch_output = self.forward_step(batch_data)
                batch_loss = self._loss_fn(*batch_output) / self._grad_accum_steps

            self._scaler.scale(batch_loss).backward()

            if (
                batch_idx + 1
            ) % self._grad_accum_steps == 0 or batch_idx + 1 == num_batches:
                self._scaler.unscale_(self._optimizer)
                if self._grad_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self._grad_clip_val
                    )
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad(set_to_none=True)

                self._train_steps += 1
                if (
                    self._checkpoint_steps is not None
                    and self._train_steps % self._checkpoint_steps == 0
                ):
                    extra = self._checkpoint_extra()
                    if hasattr(train_dataloader, "state_dict"):
                        extra["train_loader_state_dict"] = train_dataloader.state_dict()

                    self._checkpointer.async_save(self._checkpoint_path, extra=extra)

            total_loss += batch_loss.item() * self._grad_accum_steps
            avg_loss = total_loss / (batch_idx + 1)

            with torch.no_grad():
                self._update_metrics(*batch_output)

            current_metrics = self._compute_metrics()
            formatted_metrics = {k: f"{v:.4f}" for k, v in current_metrics.items()}
            pbar.set_postfix(loss=f"{avg_loss:.4f}", **formatted_metrics)

        self._history["train_loss"].append(avg_loss)
        for name, value in current_metrics.items():
            self._history[f"train_{name}"].append(value)
        self._reset_metrics()

    def _validation_loop(
        self, val_dataloader: DataLoader, epoch: int, epochs: int, verbose=True
    ) -> None:
        """
        Perform one validation epoch.

        Args:
            val_dataloader (DataLoader): Dataloader for the validation set.
            epoch (int): current validation epoch.
            epochs (int): total number of validation epochs.
            verbose (bool): keep progress bars (except final progress bar) after they complete.
            Defaults to True.
        """

        self.model.eval()
        total_loss = 0
        num_batches = len(val_dataloader)
        leave = verbose or (epoch + 1 == epochs)
        pbar = tqdm(val_dataloader, total=num_batches, desc="Validating", leave=leave)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                with torch.autocast(self.device.type, enabled=self._use_amp):
                    batch_output = self.forward_step(batch_data)
                    total_loss += self._loss_fn(*batch_output).item()
                avg_loss = total_loss / (batch_idx + 1)
                self._update_metrics(*batch_output)
                current_metrics = self._compute_metrics()
                formatted_metrics = {k: f"{v:.4f}" for k, v in current_metrics.items()}
                pbar.set_postfix(val_loss=f"{avg_loss:.4f}", **formatted_metrics)

        self._history["val_loss"].append(avg_loss)
        for name, value in current_metrics.items():
            self._history[f"val_{name}"].append(value)

        self._reset_metrics()

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int = 10,
        verbose: bool = True,
        resume: bool = False,
    ) -> None:
        """
        Train the model for a specified number of epochs with optional validation,
        learning rate scheduling and checkpointing.

        Args:
            train_dataloader (DataLoader): Training dataset loader.
            val_dataloader (DataLoader or None): Optional validation dataset loader. Defaults to None.
            epochs (int): Number of epochs to train for. Defaults to 10.
            verbose (bool): Whether to show full training details. Defaults to True.
            resume (bool): Whether to resume training with information from checkpoint_path. Defaults to False.
        """
        if self._checkpoint_path is not None:
            self._checkpointer = Checkpointer(
                model=self.model,
                optimizer=self._optimizer,
                scaler=self._scaler,
                history=self._history,
                scheduler=self._scheduler,
            )

        start_epoch = 0
        if resume:
            start_epoch = self._load_checkpoint()
            if start_epoch >= epochs:
                print(
                    f"Training already completed up to epoch {start_epoch}. Increase 'epochs' to continue."
                )
                return
            if verbose:
                print(f"Resuming training from Epoch {start_epoch+1}/{epochs}.")

            if self._train_loader_state_dict is not None and hasattr(
                train_dataloader, "load_state_dict"
            ):
                train_dataloader.load_state_dict(self._train_loader_state_dict)

        try:
            for epoch in range(start_epoch, epochs):
                self._train_loop(train_dataloader, epoch, epochs, verbose)

                if val_dataloader is not None:
                    self._validation_loop(val_dataloader, epoch, epochs, verbose)

                if self._scheduler:
                    self._scheduler_step()

                improved = self._check_improvement()

                if self._checkpoint_path:
                    self._checkpoint(improved, verbose)

                if self._patience is not None:
                    if self._early_stopping(improved, verbose):
                        print(
                            f"Early stopping triggered at Epoch {epoch + 1} with "
                            f"{self._checkpoint_metric} at {self._best_metric_val}."
                        )
                        if self._checkpoint_path and self._checkpointer.skipped:
                            self._checkpointer.sync_save(
                                self._checkpoint_path, extra=self._checkpoint_extra()
                            )
                        break
            if self._checkpoint_path and self._checkpointer.skipped:
                self._checkpointer.sync_save(
                    self._checkpoint_path, extra=self._checkpoint_extra()
                )

        finally:
            if self._checkpointer is not None:
                self._checkpointer.shutdown()

    def plot(self, figsize: tuple[int, int] = (6, 4)) -> None:
        """
        Plot training and validation curves for loss and all tracked metrics.

        Args:
            figsize (tuple[int, int]): Figure size for the plots. Defaults to (6, 4).
        """

        metrics_to_plot = ["loss"] + sorted(self._metrics.keys())

        for metric in metrics_to_plot:
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"

            plt.figure(figsize=figsize)

            if train_key in self._history:
                plt.plot(
                    range(1, len(self._history[train_key]) + 1),
                    self._history[train_key],
                    label=f"Train {metric}",
                )
            if val_key in self._history:
                plt.plot(
                    range(1, len(self._history[val_key]) + 1),
                    self._history[val_key],
                    label=f"Val {metric}",
                )

            plt.title(f"{metric.capitalize()} Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel(metric.upper())
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
