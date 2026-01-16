import math
import zipfile
from pathlib import Path
from typing import Any, Callable
from collections import OrderedDict

import torch
import requests
import pandas as pd
from tqdm import tqdm


def download_from_url(
    url: str,
    root: str | Path | None = None,
    filename: str | None = None,
    timeout: float | None = 100.0,
) -> Path:
    """
    Download a file from a URL and save it locally.

    Args:
        url (str): Direct URL of the file to download.
        root (str, Path or None): Optional directory in which to save the file or
        current working directory if None. Defaults to None.
        filename (str or None): Optional name for file.
        If None, the name is inferred from the URL. Defaults to None.
        timeout (float or None): Optional timeout settings. Defaults to 100.0

    Returns:
        Path: The path to the downloaded file.
    """
    url_filename = Path(url).name
    url_suffix = Path(url).suffix

    if filename:
        if url_suffix and not Path(filename).suffix:
            filename = filename + url_suffix.lower()
    else:
        filename = url_filename

    if root is not None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / filename
    else:
        path = Path(filename)

    if path.exists():
        return path

    else:
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = float(response.headers.get("content-length", 0))
            chunk_size = 1 * 1024 * 1024

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=path.name,
            ) as pbar:

                with open(path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        except requests.RequestException as req_err:
            print(f"Request Error Occured: {req_err}")
            if path.exists():
                path.unlink()
            raise

        except Exception as err:
            print(f"Unexpected error occurred: {err}")
            if path.exists():
                path.unlink()
            raise

        return path


def extract_zip(
    zip_path: str | Path,
    root: str | Path | None = None,
) -> Path:
    """
    Extract a ZIP file to a target directory.

    Args:
        zip_path (str or Path): Path to the ZIP file.
        root (str, Path or None): Optional extraction directory or directory named
        after ZIP file if None. Defaults to None.

    Returns:
        Path: Directory where the files are extracted.
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if root is None:
        extract_dir = zip_path.with_suffix("")
    else:
        extract_dir = Path(root)

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()

        with tqdm(total=len(members, desc=f"Extracting {zip_path.name}")) as pbar:
            for member in members:
                target_path = extract_dir / member

                if target_path.exists():
                    pbar.update(1)
                    continue

                target_path.parent.mkdir(parents=True, exist_ok=True)

                zip_ref.extract(member, extract_dir)
                pbar.update(1)

    return extract_dir


def download_and_extract_zip(
    url: str,
    download_root: str | Path | None,
    filename: str | None = None,
    remove_zip: bool = False,
):
    """
    Download a ZIP file from URL and extracts its contents.

    Args:
        url (str): URL of the ZIP file to download.
        download_root (str, Path or None): Directory for download and extraction.
        filename (str or None): Optional filename for the downloaded ZIP file.
        remove_zip (bool): Whether to remove zip file after extracting its contents.

    Returns:
        Path: Directory where the files were extracted.
    """
    zip_path = download_from_url(url, root=download_root, filename=filename)
    extract_dir = extract_zip(zip_path, root=download_root)

    if remove_zip:
        zip_path.unlink()

    return extract_dir


def _readable_bytes(bytes: int) -> str:
    """
    Convert a raw byte count into a human-readable string (KB, MB, GB, TB).

    Args:
        bytes (int): Number of bytes.

    Returns:
        str: Human-readable string representation of bytes.
    """
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    base = 1024

    if bytes == 0:
        return "0.00 B"

    rank = int(math.log(bytes, base))
    rank = min(rank, len(suffixes) - 1)

    readable = f"{(bytes / (base**rank)):.2f}"

    return f"{readable} {suffixes[rank]}"


def _apply_to_data(data: Any, fn: Callable) -> Any:
    """
    Recursively apply a function to all tensors in a nested structure.

    Args:
        data (Any): Input data (could be tensor, list, tuple, dict).
        fn (Callable): Function to apply to each tensor.

    Returns:
        Any: Data with the function applied to all tensors.
    """
    if isinstance(data, torch.Tensor):
        return fn(data)
    elif isinstance(data, (list, tuple)):
        return type(data)(_apply_to_data(x, fn) for x in data)
    elif isinstance(data, dict):
        return {k: _apply_to_data(v, fn) for k, v in data.items()}
    else:
        return data


def summary(
    model: torch.nn.Module,
    input: Any,
    depth: int = 3,
    device: torch.device | str | None = None,
) -> pd.DataFrame:
    """
    Return a pd.DataFrame summary of a PyTorch model.

    Args:
        model (nn.Module): Model to summarize.
        input (Any): Input data to pass through the model for summary.
        depth (int): Depth of layers to include in the summary. Defaults to 3.
        device (torch.device, str or None): Optional device to perform model summary on.
        If None, uses the device of the model's parameters. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the model summary.
    """
    with pd.option_context("display.min_rows", 14):

        if device is None:
            device = next(model.parameters()).device
        else:
            device = torch.device(device)
            model = model.to(device)

        batch_size = None

        def get_batch_size(t):
            nonlocal batch_size
            batch_size = t.shape[0]

        _apply_to_data(input, get_batch_size)

        input = _apply_to_data(input, lambda t: t[0:1].to(device))

        summary_data = OrderedDict()
        hooks = []
        activation_numel = 0

        input_mem = 0

        def count_input_mem(t):
            nonlocal input_mem
            input_mem += t.numel() * t.element_size()
            return t

        _apply_to_data(input, count_input_mem)

        def register_hook_recursive(module, module_depth=0):
            is_leaf = len(list(module.children())) == 0
            idx = len(summary_data)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)

            summary_data[idx] = {
                "Layer": f"{module.__class__.__name__}_{module_depth}",
                "Output Shape": None,
                "Params": params,
                "depth": module_depth,
            }

            def hook(module, inp, out):
                nonlocal activation_numel

                def process_output(t):
                    nonlocal activation_numel
                    if summary_data[idx]["Output Shape"] is None:
                        summary_data[idx]["Output Shape"] = [
                            batch_size,
                            *list(t.shape[1:]),
                        ]
                    if is_leaf:
                        activation_numel += t.numel()

                _apply_to_data(out, process_output)

            hooks.append(module.register_forward_hook(hook))

            for child in module.children():
                register_hook_recursive(child, module_depth + 1)

        def run_model(data):
            if isinstance(data, dict):
                return model(**data)
            elif isinstance(data, (list, tuple)):
                return model(*data)
            else:
                return model(data)

        # Intialize lazy modules
        with torch.no_grad():
            run_model(input)

        register_hook_recursive(model, module_depth=0)

        with torch.no_grad():
            run_model(input)

        for h in hooks:
            h.remove()

        df = pd.DataFrame.from_dict(summary_data, orient="index")
        df = df[df["depth"] < depth].drop(columns=["depth"]).reset_index(drop=True)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        elem_size = next(model.parameters()).element_size()
        input_mem *= batch_size
        model_mem = total_params * elem_size
        activation_mem = activation_numel * elem_size * batch_size
        gradient_mem = trainable_params * elem_size
        est_min_training_mem = input_mem + model_mem + activation_mem + gradient_mem

        footer = pd.DataFrame(
            [
                ["", "", ""],
                [
                    "Total Params:",
                    "",
                    f"{total_params:,} ({_readable_bytes(model_mem)})",
                ],
                [
                    "Trainable Params:",
                    "",
                    f"{trainable_params:,} ({_readable_bytes(gradient_mem)})",
                ],
                [
                    "Non-trainable Params:",
                    "",
                    f"{non_trainable_params:,} ({_readable_bytes(non_trainable_params*elem_size)})",
                ],
                ["", "", ""],
                ["Input mem:", "", _readable_bytes(input_mem)],
                ["Activation mem:", "", _readable_bytes(activation_mem)],
                ["Est min training mem:", "", _readable_bytes(est_min_training_mem)],
            ],
            columns=df.columns,
        )

        return pd.concat([df, footer], ignore_index=True)
