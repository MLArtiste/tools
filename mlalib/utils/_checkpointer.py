import copy
from typing import Any
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
        scheduler (LRScheduler or None): Optional learning rate scheduler. Defaults to None.
        device (str or torch.device): Device to use when loading checkpoint. Defaults to 'cpu'.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.GradScaler,
        history: dict[str, list[float]],
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._scaler = scaler
        self._history = history
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

    def checkpoint(self, extra: dict | None = None) -> dict[str, Any]:
        """
        Return a checkpoint dictionary.

        Args:
            extra (dict or None): Additional objects to merge into the
            checkpoint dictionary.
        Return
            dict: Dictonary to checkpoint.
        """
        checkpoint = {
            "model_state_dict": self._stage(self._model.state_dict()),
            "optimizer_state_dict": self._stage(self._optimizer.state_dict()),
            "scaler_state_dict": self._stage(self._scaler.state_dict()),
            "history": copy.deepcopy(self._history),
        }

        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._stage(
                self._scheduler.state_dict()
            )

        if extra:
            checkpoint.update(extra)

        return checkpoint

    def is_busy(self):
        """
        Return True if checkpointing is in progress.

        Returns:
            bool: Whether checkpointing is in progress.
        """
        return self._future is not None and not self._future.done()

    def async_save(self, path: str | Path, extra: dict | None = None):
        """
        Save checkpoint asynchronously.

        Args:
            path (str or Path): Path to save checkpoint.
            extra (dict or None): Additional objects to merge into the
            checkpoint at save time.
        """
        if self.is_busy():
            self.skipped = True
            return
        self.skipped = False
        self._future = self._executor.submit(torch.save, self.checkpoint(extra), path)

    def sync_save(self, path: str | Path, extra: dict | None = None):
        """
        Save checkpoint synchronously.

        Args:
            path (str or Path): Path to save checkpoint.
            extra (dict or None): Additional objects to merge into the
            checkpoint at save time.
        """
        if self._future is not None:
            self._future.result()
        torch.save(self.checkpoint(extra), path)

    def shutdown(self):
        """
        Shutdown executor.
        """
        if self._future is not None:
            self._future.result()
        self._executor.shutdown()
