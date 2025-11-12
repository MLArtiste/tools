import gc, time
from tqdm import tqdm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torchmetrics.metric import Metric
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler


class BaseNNTrainer(ABC):
    """
    Abstract base class for training neural networks with PyTorch.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: Optimizer, 
        loss_fn: nn.Module,
        metrics: dict[str, Metric] | None = None,
        scheduler: LRScheduler | None = None,
        lrs_metric: str | None = None,
        device: torch.device| str | None = None,  
        model_save_path: str | None = None,
        use_amp: bool = True,
        grad_clip_val: float | None = None
    ):
        """
        Initialize the Trainer with model, optimizer, loss function, 
        metrics, scheduler, and other training options.

        Args:
            model (nn.Module): Neural network model to train.
            optimizer (Optimizer): Optimization algorithm for parameter updates.
            loss_fn (nn.Module): Loss function to minimize during training.
            metrics (dict[str, Metric] | None): Optional dictionary of metrics to track.
            scheduler (LRScheduler | None): Optional learning rate scheduler.
            lrs_metric (str | None): Metric name to monitor for ReduceLROnPlateau.
            device (torch.device | str | None): Device to use ('cpu', 'cuda', etc.).
            model_save_path (str | None): File path to save the best model.
            use_amp (bool): Whether to use automatic mixed precision.
            grad_clip_val (float | None): Maximum gradient norm for clipping.
        """
        
        self.device = device if device else torch.accelerator.current_accelerator().type\
                      if torch.accelerator.is_available() else "cpu"
    
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        
        model = model.to(self.device)
        if torch.accelerator.device_count() > 1:
            model = torch.nn.DataParallel(model)

        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._scheduler = scheduler
        self._lrs_metric = lrs_metric
        self._use_amp = use_amp
        self._model_save_path = model_save_path
        self._grad_clip_val = grad_clip_val
        self._history = {"train_loss": [], "val_loss": []}
        self._scaler = torch.amp.GradScaler(self.device.type, enabled=self._use_amp)
        self._metrics = {name.lower(): metric for name, metric in (metrics or {}).items()}
        

        if self._metrics:
            for metric_name in self._metrics:
                self._history[f"train_{metric_name}"] = []
                self._history[f"val_{metric_name}"] = []
        self._move_metrics_to_device()
        
    def _move_metrics_to_device(self) -> None:
        """
        Move all metric objects to the same device as the model.
        """
        for metric in self._metrics.values():
            metric.to(self.device)
    
    @torch.no_grad()
    def _update_metrics(self, *args: object) -> None:
        """
        Update metric values using model predictions and ground truths.

        Args:
            *args (object): Values returned from _forward_step. 
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

    def _scheduler_step(self, metric_value: float | None = None) -> None:
        """
        Step the learning rate scheduler.
        Use a monitored metric for ReduceLROnPlateau.

        Args:
            metric_value (float | None): Metric value used for ReduceLROnPlateau.
        """
        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric_value is None:
                raise ValueError("A metric must be provided for ReduceLROnPlateau.")
            self._scheduler.step(metric_value)
        else:
            self._scheduler.step()

    def _cleanup_memory(self) -> None:
        """
        Clear device memory cache and trigger garbage collection to free resources.
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.device == 'mps':
            torch.mps.empty_cache()
        gc.collect()

    def _save_best_model(
        self,
        metric: str,
        metric_cur_val: float, 
        minimize_metric: bool
    ):
        """
        Save model checkpoint if current metric improves upon the best recorded value.
    
        Args:
            metric (str): Metric name to monitor.
            metric_cur_val (float): Current metric value.
            minimize_metric (bool): Whether to minimize (True) or maximize (False) the metric.
        """
        metric = f"val_{metric}"
        if minimize_metric:
            metric_best_val = min(self._history[metric]) if len(self._history[metric]) else float("inf")
        else:
            metric_best_val = max(self._history[metric]) if len(self._history[metric]) else -float("inf")
            
        if (minimize_metric and metric_cur_val < metric_best_val) or \
        (not minimize_metric and metric_cur_val > metric_best_val):
            print(f"Saving best model with {metric}: {metric_cur_val:.4f}")
            model_to_save = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            torch.save(model_to_save.state_dict(), self._model_save_path)
        
    @abstractmethod
    def _forward_step(self, batch_data: object) -> object:
        """
        Forward pass for a single batch. Should move data to self.device and 
        return object (typically (y_pred, y)) for loss computation.
    
        Args:
            batch_data (object): A batch from the dataloader.
    
        Returns:
            object: Input to loss function and metrics. Typically model predictions and corresponding targets (y_pred, y).
        """

    def get_history(self) -> dict[str, list[float]]:
        """
        Retrieve the training and validation history of losses and metrics.
    
        Returns:
            dict[str, list[float]]: Recorded loss and metric values per epoch.
        """
        return self._history
    
    def _train_loop(self, train_dataloader: DataLoader) -> None:
        """
        Perform one training epoch, updating model parameters and metrics.
    
        Args:
            train_dataloader (DataLoader): Dataloader for the training set.
        """
        self._model.train()
        total_loss = 0
        num_batches = len(train_dataloader)
        pbar = tqdm(train_dataloader, total=num_batches)

        for batch_idx, batch_data in enumerate(pbar):
            with torch.autocast(self.device.type, enabled=self._use_amp):
                batch_output = self._forward_step(batch_data)
                batch_loss = self._loss_fn(*batch_output)
            
            # Backward pass
            self._scaler.scale(batch_loss).backward()
            self._scaler.unscale_(self._optimizer)
            if self._grad_clip_val:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip_val)
            self._scaler.step(self._optimizer)
            self._scaler.update()
            self._optimizer.zero_grad()

            # Accumulate loss and metrics
            total_loss += batch_loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            self._update_metrics(*batch_output)
            
            # compute metrics
            current_metrics = self._compute_metrics()
            
            # Display intermediate results
            metric_repr = "| ".join([f"train_{name}: {value:.4f}" for name, value in current_metrics.items()])
            pbar.set_description(f"Loss: {avg_loss:.4f}| {metric_repr}")

        self._history["train_loss"].append(avg_loss)
        for name, value in current_metrics.items():
            self._history[f"train_{name}"].append(value)
        self._reset_metrics()
        self._cleanup_memory()
    
    def _validation_loop(
        self, 
        val_dataloader: DataLoader,
        save_on_metric: str,
        minimize_metric: bool
    ) -> None:
        """
        Perform one validation epoch, compute metrics, and optionally save the best model.
    
        Args:
            val_dataloader (DataLoader): Dataloader for the validation set.
            save_on_metric (str): Metric name used to determine best model.
            minimize_metric (bool): Whether to minimize (True) or maximize (False) the metric.
        """

        self._model.eval()
        total_loss = 0
        num_batches = len(val_dataloader)
        pbar = tqdm(val_dataloader, total=num_batches)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                pbar.set_description("Validating")
                batch_output = self._forward_step(batch_data)
                total_loss += self._loss_fn(*batch_output).item()
                self._update_metrics(*batch_output)

        avg_loss = total_loss / num_batches
        
        if self._model_save_path:
            save_on_metric = save_on_metric.lower()
            if save_on_metric == 'loss':
                metric_cur_val = avg_loss
            elif save_on_metric in self._metrics:
                metric_cur_val = self._metrics[save_on_metric].compute().item()
            else: raise ValueError(f"{save_on_metric} is not in history")
                
            self._save_best_model(save_on_metric, metric_cur_val, minimize_metric)
        
        # compute metrics
        current_metrics = self._compute_metrics()
        
        # update history 
        self._history["val_loss"].append(avg_loss)
        for name, value in current_metrics.items():
            self._history[f'val_{name}'].append(value)
            
        # Display intermediate results
        metric_repr = "| ".join([f"val_{name}: {value:.4f}" for name, value in current_metrics.items()])        
        print(f"Val Loss: {avg_loss:.4f} {metric_repr}")
        self._reset_metrics()
        self._cleanup_memory()
    
    def fit(
        self, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader | None = None, 
        epochs: int = 10,
        save_on_metric: str = 'loss',
        minimize_metric: bool =True
    ) -> None:
        """
        Train the model for a specified number of epochs with optional validation.
    
        Args:
            train_dataloader (DataLoader): Training dataset loader.
            val_dataloader (DataLoader | None): Optional validation dataset loader.
            epochs (int): Number of epochs to train for.
            save_on_metric (str): Metric name to save best model on.
            minimize_metric (bool): Whether to minimize or maximize the save metric.
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}\n{'-' * 40}")
            start = time.time()
            self._train_loop(train_dataloader)
            print(f"Took {((time.time() - start) / 60):.2f} minutes for epoch {epoch+1}")
            
            if val_dataloader:
                self._validation_loop(val_dataloader, save_on_metric, minimize_metric)
                
            if self._scheduler:
                if self._lrs_metric:
                    if self._lrs_metric not in self._history:
                        raise ValueError(f"{self._lrs_metric} is not in history.")
                    metric_value = self._history[self._lrs_metric][-1]
                    self._scheduler_step(metric_value)
                else:
                    self._scheduler_step()

    def plot(self, figsize: tuple[float, float] = (6, 4)) -> None:
        """
        Plot training and validation curves for loss and all tracked metrics.
    
        Args:
            figsize (tuple[float, float]): Figure size for the plots.
        """
        
        metrics_to_plot = ["loss"] + sorted(self._metrics.keys())
    
        for metric in metrics_to_plot:
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"
    
            plt.figure(figsize=figsize)
            
            if train_key in self._history:
                plt.plot(range(1, len(self._history[train_key]) + 1), 
                         self._history[train_key], 
                         label=f"Train {metric}")
            if val_key in self._history:
                plt.plot(range(1, len(self._history[val_key]) + 1),
                         self._history[val_key], 
                         label=f"Val {metric}")
    
            plt.title(f"{metric.capitalize()} Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel(metric.upper())
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
