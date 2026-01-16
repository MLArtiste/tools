from pathlib import Path
from typing import Any, Callable, Literal, Union

import pandas as pd

PandasObj = Union[pd.DataFrame, pd.Series]


def load_dataset(
    path: str | Path,
    file_fmt: str,
    transform: Callable[[PandasObj], PandasObj] | None = None,
    **kwargs,
) -> Any:
    """
    Load tabular dataset and apply transform.

    Args:
        path (str or path): Path to the dataset file.
        file_fmt (str): Format of the file (e.g, 'csv', 'excel', 'json', 'parquet', etc).
        transform (Callable or None): A callable that takes a DataFrame/Series and returns
        the transformed version. If None, the raw DataFrame is returned. Defaults to None.
        **kwargs: Additional keyword arguments passed directly to the pandas reader.
    Returns:
        Any: The (and potentially transformed) dataset."""

    path = Path(path)
    fmt = file_fmt.lower().lstrip(".")
    readers = {
        "csv": pd.read_csv,
        "excel": pd.read_excel,
        "json": pd.read_json,
        "parquet": pd.read_parquet,
        "xml": pd.read_xml,
        "html": pd.read_html,
        "sql": pd.read_sql,
    }
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: '{path}'")

    if fmt not in readers:
        raise ValueError(
            f"unsupported file format: '{fmt}', expected one of {list(readers.keys())}"
        )
    df = readers[fmt](path, **kwargs)

    if transform:
        return transform(df)

    return df


class LazyDataLoader:
    """
    lazy dataloader for large tabular datasets.

    Args:
        path (str or path): Path to the dataset file.
        batch_size (int): Number of rows to load at a time. Defaults to 100000.
        transform (Callable or None): Optional transform to apply to each batch.
        Defaults to None.
        **kwargs: Additional keyword arguments passed directly to pd.read_csv.
    """

    def __init__(
        self,
        path: str | Path,
        batch_size: int = 100_000,
        transform: Callable[[PandasObj], PandasObj] | None = None,
        **kwargs: Any,
    ):
        self._path = Path(path)
        self._batch_size = batch_size
        self._transform = transform
        self._kwargs = kwargs

        if not self._path.exists():
            raise FileNotFoundError(f"dataset not found: '{self._path}'")

    def __iter__(self):
        """
        Return an iterator over the dataset.
        """
        reader = pd.read_csv(
            self._path,
            chunksize=self._batch_size,
            **self._kwargs,
        )
        for batch in reader:
            if self._transform:
                batch = self._transform(batch)
            yield batch


class TabularDataLoader:
    """
    Unified interface for loading tabular datasets in eager or lazy mode.

    Notes:
        - In eager mode, use 'load_dataset()' method to materialize the dataset.
        - In lazy mode, iterate over the object to stream batches.

    Args:
        path (str or path): Path to dataset file.
        file_fmt (str): Format of the file (e.g, 'csv', 'excel', 'json', 'parquet').
        transform (Callable or None): A callable that takes a DataFrame/Series and returns
        the transformed version. If None, the raw DataFrame is returned. Defaults to None.
        mode (str): Mode to use for loading the dataset. Defaults to "eager".
        batch_size (int): Number of rows to load at a time. Defaults to 100000.
        **kwargs (Any): Additional keyword arguments passed directly to pandas readers.
    """

    def __init__(
        self,
        path: str | Path,
        file_fmt: str,
        transform: Callable[[PandasObj], PandasObj] | None = None,
        mode: Literal["eager", "lazy"] = "eager",
        batch_size: int = 100000,
        **kwargs: Any,
    ):
        self._path = Path(path)
        self._file_fmt = file_fmt.lower().lstrip(".")

        self._transform = transform
        self._mode = mode.lower()
        self._batch_size = batch_size
        self._kwargs = kwargs

        if not self._path.exists():
            raise FileNotFoundError(f"dataset not found: '{self._path}'")

        if self._mode not in {"eager", "lazy"}:
            raise ValueError(
                f"Unsupported mode: '{mode}', expected one of 'eager' or 'lazy'"
            )
        
        if self._mode == "lazy" and self._file_fmt != "csv":
            raise RuntimeError("Lazy mode is only supported for CSV files.")


    def load_dataset(self) -> Any:
        """
        Load the dataset eagerly.

        Returns:
            Any: The (potentially transformed) dataset.
        """
        if self._mode != "eager":
            raise RuntimeError("'.load_dataset()' is only supported in eager mode")

        return load_dataset(
            self._path,
            self._file_fmt,
            self._transform,
            **self._kwargs,
        )

    def __iter__(self):
        if self._mode != "lazy":
            raise RuntimeError(
                "iteration is only supported in lazy mode"
                "did you mean to call '.load_dataset()'?"
            )
        return iter(
            LazyDataLoader(
                self._path,
                self._batch_size,
                self._transform,
                **self._kwargs,
            )
        )
