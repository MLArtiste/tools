from pathlib import Path
from typing import Any, Callable

from ..utils.data import TabularDataLoader
from ...utils import download_from_url


class Titanic(TabularDataLoader):
    """
    Titanic dataset.

    Source:
        Stanford University, CS109: Probability for Computer Scientists.
        Original data compiled by:
        British Board of Trade (1912).

        https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    FILE_NAME = "titanic.csv"

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        root = Path(root) if root is not None else Path.cwd()
        path = root / self.FILE_NAME

        if not path.exists():
            if download:
                path = download_from_url(self.URL, root=root, filename=self.FILE_NAME)
            else:
                raise FileNotFoundError(
                    f"{self.File_NAME} not found in {root}. "
                    "Set download=True to download it."
                )

        super().__init__(
            path=path,
            file_fmt="csv",
            transform=transform,
            **kwargs,
        )
