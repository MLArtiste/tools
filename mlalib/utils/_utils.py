import math
import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path
from collections import OrderedDict
from typing import Any, Callable, Iterator, Literal

import torch
import requests
import pandas as pd
from tqdm import tqdm

from ._gdown import download_from_gdrive


def download_from_url(
    url: str,
    root: str | Path | None = None,
    filename: str | Path | None = None,
    timeout: float | None = 100.0,
) -> Path:
    """
    Download a file from a URL.

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
        filename = Path(filename)
        if url_suffix and not Path(filename).suffix:
            filename = filename.with_suffix(url_suffix.lower())
    else:
        filename = Path(url_filename)

    if root is not None:
        root = Path(root)
        path = root / filename
    else:
        path = filename

    path.parent.mkdir(parents=True, exist_ok=True)

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


def extract_tar(
    tar_path: str | Path,
    root: str | Path | None = None,
    mode: Literal["tar", "gz", "bz2", "xz"] | None = None,
) -> Path:
    """
    Extract a TAR file to a target directory.

    Args:
        tar_path (str or Path): Path to the TAR file.
        root (str, Path or None): Optional extraction directory. Uses the parent
        directory of the TAR file if None. Defaults to None.
        mode (str or None): Optional compression mode (e.g 'tar', 'gz', 'bz2', 'xz').
        Automatically detects mode if None. Defaults to None.

    Returns:
        Path: Directory where the files are extracted.
    """
    if mode not in {"tar", "gz", "bz2", "xz", None}:
        raise ValueError(
            f"invalid mode: {mode}, expected one of 'tar', 'gz', 'bz2' or 'xz'"
        )

    mode = "r:*" if mode is None else f"r:{mode}"
    tar_path = Path(tar_path)

    if not tar_path.exists():
        raise FileNotFoundError(f"TAR file not found: {tar_path}")

    if root is None:
        extract_dir = tar_path.parent
    else:
        extract_dir = Path(root)

    if tar_path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        tar_folder = extract_dir / tar_path.name.split(".", 1)[0]
    else:
        tar_folder = extract_dir / tar_path.with_suffix("")

    if tar_folder.exists() and any(tar_folder.iterdir()):
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, mode=mode) as tar_ref:
        print(f"Extracting {tar_path} to {extract_dir}")
        tar_ref.extractall(path=extract_dir, filter="data")
        print(f"Extraction complete")

    return extract_dir


def extract_zip(
    zip_path: str | Path,
    root: str | Path | None = None,
) -> Path:
    """
    Extract a ZIP file to a target directory.

    Args:
        zip_path (str or Path): Path to the ZIP file.
        root (str, Path or None): Optional extraction directory. Uses the parent
        directory of the ZIP file if None. Defaults to None.

    Returns:
        Path: Directory where the files are extracted.
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if root is None:
        extract_dir = zip_path.parent
    else:
        extract_dir = Path(root)

    zip_folder = extract_dir / zip_path.with_suffix("")

    if zip_folder.exists() and any(zip_folder.iterdir()):
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print(f"Extracting {zip_path} to {extract_dir}")
        zip_ref.extractall(path=extract_dir)
        print(f"Extraction complete")

    return extract_dir


def extract_gzip(
    gzip_path: str | Path,
    root: str | Path | None = None,
) -> Path:
    """
    Decompress a GZIP file.

    Args:
        gzip_path (str or Path): Path to the GZIP file.
        root (str, Path or None): Optional output directory. Uses the parent
        directory of the GZIP file if None. Defaults to None.

    Returns:
        Path: Path to the decompressed file.
    """
    gzip_path = Path(gzip_path)

    if not gzip_path.exists():
        raise FileNotFoundError(f"GZIP file not found: {gzip_path}")

    if root is None:
        output_dir = gzip_path.parent
    else:
        output_dir = Path(root)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / gzip_path.with_suffix("").name

    if output_file.exists():
        return output_file

    print(f"Extracting {gzip_path} to {output_file}")

    with gzip.open(gzip_path, "rb") as src:
        with output_file.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    print(f"Extraction complete")

    return output_file


def download_and_extract_tar(
    url: str,
    root: str | Path | None = None,
    filename: str | None = None,
    mode: Literal["tar", "gz", "bz2", "xz"] | None = None,
    from_gdrive: bool = False,
    remove_tar: bool = False,
):
    """
    Download a TAR file from URL and extracts its contents.

    Args:
        url (str): URL of the TAR file to download.
        root (str, Path or None): Optional directory for download and extraction.
        Current working directory if None.
        filename (str or None): Optional filename for the downloaded TAR file.
        mode (str or None): Optional compression mode (e.g 'tar', 'gz', 'bz2', 'xz').
        from_gdrive (bool): Whether the URL is from Google Drive. Defaults to False.
        remove_tar (bool): Whether to remove tar file after extracting its contents.

    Returns:
        Path: Directory where the files were extracted.
    """
    if from_gdrive:
        tar_path = download_from_gdrive(url, root=root, filename=filename)
    else:
        tar_path = download_from_url(url, root=root, filename=filename)
    extract_dir = extract_tar(tar_path, root=root, mode=mode)

    if remove_tar:
        tar_path.unlink()

    return extract_dir


def download_and_extract_zip(
    url: str,
    root: str | Path | None = None,
    filename: str | None = None,
    from_gdrive: bool = False,
    remove_zip: bool = False,
):
    """
    Download a ZIP file from URL and extracts its contents.

    Args:
        url (str): URL of the ZIP file to download.
        root (str, Path or None): Optional directory for download and extraction.
        Uses current working directory if None.
        filename (str or None): Optional filename for the downloaded ZIP file.
        from_gdrive (bool): Whether the URL is from Google Drive. Defaults to False.
        remove_zip (bool): Whether to remove zip file after extracting its contents.

    Returns:
        Path: Directory where the files were extracted.
    """
    if from_gdrive:
        zip_path = download_from_gdrive(url, root=root, filename=filename)
    else:
        zip_path = download_from_url(url, root=root, filename=filename)
    extract_dir = extract_zip(zip_path, root=root)

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


def apply_to_tensor(obj: Any, fn: Callable) -> Any:
    """
    Recursively apply a function to all tensors in a nested structure.

    Args:
        obj (Any): Nested structure (could be tensor, list, tuple, dict).
        fn (Callable): Function to apply to each tensor.

    Returns:
        Any: Nested structure with the function applied to all tensors.
    """
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(apply_to_tensor(x, fn) for x in obj)
    elif isinstance(obj, dict):
        return {k: apply_to_tensor(v, fn) for k, v in obj.items()}
    else:
        return obj


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

        apply_to_tensor(input, get_batch_size)

        input = apply_to_tensor(input, lambda t: t[0:1].to(device))

        summary_data = OrderedDict()
        hooks = []
        activation_numel = 0

        input_mem = 0

        def count_input_mem(t):
            nonlocal input_mem
            input_mem += t.numel() * t.element_size()
            return t

        apply_to_tensor(input, count_input_mem)

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

                apply_to_tensor(out, process_output)

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


class HardSamples:
    """
    Base class for finding hard samples in a dataset.

    The default implementation identifies misclassified samples.
    Subclass this class and override `evaluate_batch()` to define
    a custom notion of difficulty.

    Args:
        model (nn.Module): A PyTorch model.
        dataloader (torch.utils.data.DataLoader): A PyTorch dataloader.
        min_samples (int or None): The minimum number of indices to return
        when get_indices() is called.
        device (torch.device, str or None): Optional device to use for evaluation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        min_samples: int | None = None,
        device: torch.device | str | None = None,
    ):
        self.model = model.to(device)
        self._dataloader = dataloader
        self.device = device
        self._min_samples = min_samples
        self._indices = []

    def evaluate_batch(self, batch_data: Any) -> torch.Tensor:
        """
        Evaluate the difficulty of samples in a batch.

        Notes:
            Override this method in subclasses to define a custom
            difficulty criterion. Please move data to self.device.

        Args:
            batch_data (Any): A batch of data.

        Returns:
            torch.Tensor: A tensor whose non-zero values indicate
            difficult samples.
        """
        X, y = apply_to_tensor(batch_data, lambda x: x.to(self.device))
        y_pred = self.model(X)
        return y != y_pred.argmax(dim=1)

    def filter_batch(self, batch_data: Any, mask: torch.Tensor) -> Any:
        """
        Apply mask to batch data to return a filtered batch.

        Notes:
            Override this method in subclasses to define a custom
            filtering method for specific datasets.

        Args:
            batch_data (Any): A batch of data.
            mask (torch.Tensor): A boolean mask.

        Returns:
            Any: A filtered batch.
        """
        return apply_to_tensor(batch_data, lambda x: x[mask])

    def get_indices(self, recompute: bool = False) -> list[int]:
        """
        Return the Dataset indices of the samples that are difficult.

        Args:
            recompute (bool): Whether or not to recompute the indices.
            Defaults to False.

        Returns:
            list[int]: A list of indices.
        """
        if self._indices and not recompute:
            return self._indices

        if not isinstance(self._dataloader.sampler, torch.utils.data.SequentialSampler):
            raise ValueError(
                "Dataloader must use SequentialSampler." "Set shuffle=False."
            )
        self.model.eval()
        batch_masks = []
        hard_count = 0

        with torch.no_grad():
            for batch_data in tqdm(self._dataloader, desc="Finding hard samples"):
                scores = self.evaluate_batch(batch_data)
                mask = scores.bool()
                batch_masks.append(mask)
                hard_count += mask.sum().item()
                if self._min_samples is not None and hard_count >= self._min_samples:
                    break

        full_mask = torch.cat(batch_masks, dim=0)
        self._indices = torch.nonzero(full_mask).flatten().tolist()

        return self._indices

    def __iter__(self) -> Iterator[Any]:
        self.model.eval()
        with torch.no_grad():
            for batch_data in self._dataloader:
                scores = self.evaluate_batch(batch_data)
                mask = scores.bool()
                if not mask.any():
                    continue
                yield self.filter_batch(batch_data, mask)
