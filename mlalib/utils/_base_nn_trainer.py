import gc
from typing import Any
from abc import ABC, abstractmethod

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import Optimizer
from torchmetrics.metric import Metric
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler


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
        checkpoint_path (str or None): Optional file path for checkpointing based on best metric value.
        checkpoint_metric (str): name of metric to monitor. Expects one of keys in trainer history.
        Defaults to 'val_loss'.
        minimize_metric (bool): Whether to minimize or maximize the metric.
        Defaults to True.
        Trainer will checkpoint only when provided. Defaults to None.
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
        checkpoint_path: str | None = None,
        checkpoint_metric: str = "val_loss",
        minimize_metric: bool = True,
        use_amp: bool = True,
        grad_clip_val: float | None = None,
        grad_accum_steps: int = 1,
    ):
        if grad_accum_steps < 1 or not isinstance(grad_accum_steps, int):
            raise ValueError("grad_accum_steps must be a positive integer")

        if grad_clip_val is not None and grad_clip_val <= 0:
            raise ValueError("grad_clip_val must be positive")

        if device is None:
            self.device = torch.device(
                torch.accelerator.current_accelerator().type
                if torch.accelerator.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        model = model.to(self.device)

        if device is None and torch.accelerator.device_count() > 1:
            model = torch.nn.DataParallel(model)

        self.model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._scheduler = scheduler
        self._lrs_metric = lrs_metric.lower()
        self._checkpoint_metric = checkpoint_metric.lower()
        self._minimize_metric = minimize_metric
        self._checkpoint_path = checkpoint_path
        self._use_amp = use_amp
        self._grad_clip_val = grad_clip_val
        self._grad_accum_steps = grad_accum_steps
        self._history = {"train_loss": [], "val_loss": []}
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

        if self._checkpoint_metric not in self._history:
            raise ValueError(
                f"""invalid checkpoint metric '{self._checkpoint_metric}'. 
                Expected one of {list(self._history.keys())}"""
            )

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

    def _cleanup_memory(self) -> None:
        """
        Clear device memory cache and trigger garbage collection to free resources.
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    def _checkpoint(
        self,
        verbose: bool = True,
    ):
        """
        checkpoint training progress.

        Args:
            verbose (bool): Whether to show checkpointing detail. Defaults to True.
        """
        metr_cur_val = self._history[self._checkpoint_metric][-1]
        hist = self._history[self._checkpoint_metric]
        metr_hist = hist[:-1] if len(hist) > 1 else []

        if self._minimize_metric:
            metr_best_val = min(metr_hist) if len(metr_hist) else float("inf")
        else:
            metr_best_val = max(metr_hist) if len(metr_hist) else -float("inf")

        if (self._minimize_metric and metr_cur_val < metr_best_val) or (
            not self._minimize_metric and metr_cur_val > metr_best_val
        ):
            if verbose:
                print(f"Checkpoint at {self._checkpoint_metric} = {metr_cur_val:.4f}")

            model = (
                self.model.module
                if isinstance(self.model, torch.nn.DataParallel)
                else self.model
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "history": self._history,
            }

            if self._scheduler is not None:
                checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()

            torch.save(checkpoint, self._checkpoint_path)

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
        pbar = tqdm(
            train_dataloader,
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=leave,
        )

        for batch_idx, batch_data in enumerate(pbar):
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
        self._cleanup_memory()

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
        self._cleanup_memory()

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        Train the model for a specified number of epochs with optional validation,
        learning rate scheduling and checkpointing.

        Args:
            train_dataloader (DataLoader): Training dataset loader.
            val_dataloader (DataLoader or None): Optional validation dataset loader. Defaults to None.
            epochs (int): Number of epochs to train for. Defaults to 10.
            verbose (bool): Whether to show full training details. Defaults to True.
        """

        for epoch in range(epochs):
            self._train_loop(train_dataloader, epoch, epochs, verbose)

            if val_dataloader:
                self._validation_loop(val_dataloader, epoch, epochs, verbose)

            if self._scheduler:
                self._scheduler_step()

            if self._checkpoint_path:
                self._checkpoint(verbose)

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
