from pathlib import Path
from typing import Any, Callable

import requests

from ._dataloaders import TabularDataLoader
from ....utils import download_from_url


def download_from_uci(
    id: int,
    root: str | Path | None = None,
    filename: str | None = None,
    timeout: float | None = 100.0,
) -> Path:
    """
    Download file of with the given ID from UCI Machine Learning Repository.

    Args:
        id (int): ID of UCI dataset to download.
        root (str, Path or None): Optional directory in which to save the file or
        current working directory if None. Defaults to None.
        filename (str or None): Optional name for file.
        If None, the name is inferred from the URL. Defaults to None.
        timeout (float or None): Optional timeout settings. Defaults to 100.0

    Returns:
        Path: The path to the downloaded file.
    """
    if not isinstance(id, int):
        raise ValueError("ID must be an integer")

    if root is not None:
        path = Path(root) / filename
    else:
        path = Path(filename)
    if path.exists():
        return path

    api_url = "https://archive.ics.uci.edu/api/dataset?id=" + str(id)

    try:
        response = requests.get(api_url, timeout=100)
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as req_err:
        print(f"Request Error Occured: {req_err}")
        raise

    except Exception as err:
        print(f"Unexpected error occurred: {err}")
        raise

    if data["status"] != 200:
        raise FileNotFoundError("data not found in repository")

    data_url = data["data"]["data_url"]

    if not data_url:
        raise FileNotFoundError("data url is not available, try manual download")

    return download_from_url(data_url, root=root, filename=filename, timeout=timeout)


class UCIDataLoader(TabularDataLoader):
    """
    Class for loading datasets from UCI machine learning repository.

    Args:
        id (int): UCI repository ID of the dataset.
        filename (str): Name of the dataset file.
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        file_fmt (str): Format of the file (e.g, 'csv', 'excel', 'json', 'parquet'). Defaults to 'csv'.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        id: int,
        filename: str,
        root: str | Path | None = None,
        file_fmt: str = "csv",
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        root = Path(root) if root is not None else Path.cwd()
        path = root / filename

        if download:
            path = download_from_uci(id, root=root, filename=filename)

        super().__init__(
            path=path,
            file_fmt=file_fmt,
            transform=transform,
            **kwargs,
        )
