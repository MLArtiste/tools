import math
from typing import Callable

import torch

import numpy as np
import matplotlib.pyplot as plt


def plot_functions(
    funcs: Callable | list[Callable],
    x: torch.Tensor | None = None,
    grid: bool = False,
    nrows: int = 2,
    figsize: tuple[float, float] | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
):
    """
    Plot one or more mathematical functions.

    Args:
        funcs(Callable or list[Callable]): A function or list of functions.
        x (torch.Tensor or None): Optional input values. Defaults to torch.linspace(-10, 10, 100).
        grid (bool): If True, plot each function in its own subplot grid.
        Otherwise, overlay all functions on one plot. Defaults to False.
        nrows (int): Number of subplot rows when grid=True. Defaults to 2.
        figsize (tuple[float, float] or None): Optional matplotlib figure size.
        labels (list[str] or None): Optional labels for each function.
        title (str or None): Optional figure title.
    """

    x = torch.linspace(-10, 10, 100) if x is None else x
    funcs = [funcs] if callable(funcs) else list(funcs)
    labels = [f.__name__ for f in funcs] if labels is None else labels

    if len(funcs) != len(labels):
        raise ValueError(f"expected {len(funcs)} labels, got {len(labels)}")

    def to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    x_np = to_numpy(x)
    ys = [to_numpy(f(x)) for f in funcs]

    # Single combined plot
    if not grid or len(funcs) == 1:
        plt.figure(figsize=figsize)

        for y, label in zip(ys, labels):
            plt.plot(x_np, y, label=label)

        plt.xlabel("x")
        plt.ylabel("y")

        if title:
            plt.title(title)

        if len(funcs) > 1:
            plt.legend()

        plt.grid(True)
        plt.show()

        return

    # Grid plotting
    ncols = math.ceil(len(funcs) / nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, y, label in zip(axes, ys, labels):
        ax.plot(x_np, y)
        ax.set_title(label)
        ax.grid(True)

    # Hide unused axes
    for ax in axes[len(funcs) :]:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def show_images(
    imgs: torch.Tensor | list[torch.Tensor],
    nrows: int = 2,
    figsize: tuple[float, float] | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    cmap: str | None = None,
    label_fontsize: float | None = None,
):
    """
    Display one or more tensor images.

    Args:
        imgs (Tensor or list[Tensor]):
        Supported shapes:
        - (H, W)
        - (C, H, W)
        - (B, C, H, W)
        - list of images
        nrows (int): Number of subplot rows.
        figsize (tuple or None): Optional matplotlib figure size.
        labels (list[str] or None): Optional title for each subplot.
        title (str or None): Optional figure title.
        cmap (str or None): Optional matplotlib colormap.
        label_fontsize (float or None): Optional font size for labels. 
    """

    if isinstance(imgs, torch.Tensor):
        if imgs.ndim == 4:
            imgs = list(imgs)
        else:
            imgs = [imgs]

    n_images = len(imgs)

    if nrows < 1:
        raise ValueError("nrows must be at least 1")

    nrows = min(nrows, n_images)

    if labels is not None and len(labels) != n_images:
        raise ValueError(f"expected {n_images} labels, got {len(labels)}")

    ncols = math.ceil(n_images / nrows)
    figsize = figsize or (3 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = img.numpy()

        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        ax.imshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])

        if labels is not None:
            ax.set_title(labels[i], fontsize=label_fontsize)

    for ax in axes[n_images:]:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()
