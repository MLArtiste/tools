import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch

from ._utils import apply_to_tensor


class Checkpointer:
    """
    Class for checkpointing training progress.

    Args:
        model (nn.Module): Model to checkpoint.
        optimizer (Optimizer): Optimizer.
        scaler (GradScaler): Gradient scaler.
        history (dict[str, list[float]]): Recorded loss and metric values per epoch.
        es_counter (int): Early stopping counter.
        scheduler (LRScheduler or None): Optional learning rate scheduler. Defaults to None.
        improved (bool): Whether the validation metric has improved. Defaults to False.
        device (str or torch.device): Device to use when loading checkpoint. Defaults to 'cpu'.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.GradScaler,
        history: dict[str, list[float]],
        es_counter: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._scaler = scaler
        self._history = history
        self._es_counter = es_counter
        self._scheduler = scheduler
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self.skipped = False

    def _stage(self, obj):
        """
        Copy nested tensors to CPU and return them.

        Returns:
            Any: Object with tensors staged.
        """
        return apply_to_tensor(obj, lambda x: x.detach().cpu())

    def checkpoint(self):
        """
        Return a checkpoint dictionary.

        Return
            dict: Dictonary to checkpoint.
        """
        checkpoint = {
            "model_state_dict": self._stage(self._model.state_dict()),
            "optimizer_state_dict": self._stage(self._optimizer.state_dict()),
            "scaler_state_dict": self._stage(self._scaler.state_dict()),
            "history": copy.deepcopy(self._history),
            "es_counter": self._es_counter,
        }

        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._stage(
                self._scheduler.state_dict()
            )

        return checkpoint

    def is_busy(self):
        """
        Return True if checkpointing is in progress.

        Returns:
            bool: Whether checkpointing is in progress.
        """
        return self._future is not None and not self._future.done()

    def async_save(self, path: str | Path):
        """
        Save checkpoint asynchronously.

        Args:
            path (str or Path): Path to save checkpoint.
        """
        if self.is_busy():
            self.skipped = True
            return
        self.skipped = False
        self._future = self._executor.submit(torch.save, self.checkpoint(), path)

    def sync_save(self, path: str | Path):
        """
        Save checkpoint synchronously.

        Args:
            path (str or Path): Path to save checkpoint.
        """
        if self._future is not None:
            self._future.result()
        torch.save(self.checkpoint(), path)

    def shutdown(self):
        """
        Shutdown executor.
        """
        if self._future is not None:
            self._future.result()
        self._executor.shutdown()
