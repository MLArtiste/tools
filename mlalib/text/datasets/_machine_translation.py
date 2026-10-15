from pathlib import Path

from ...utils import download_from_url


class WMT14EnFr:
    """
    WMT14 English-French dataset.

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
        use_12m_subset (bool): Whether to use the 12M subset. Defaults to True.
    """

    _URLS = {
        "train": "https://huggingface.co/datasets/MLArtiste/mlalib_data/resolve/main/text/wmt14_enfr.parquet",
        "train_12m": "https://huggingface.co/datasets/MLArtiste/mlalib_data/resolve/main/text/wmt14_enfr_12m.parquet",
        "test": "https://huggingface.co/datasets/MLArtiste/mlalib_data/resolve/main/text/wmt14_enfr_test.parquet",
    }

    _FILENAMES = {
        "train": "wmt14_enfr.parquet",
        "train_12m": "wmt14_enfr_12m.parquet",
        "test": "wmt14_enfr_test.parquet",
    }

    NUM_SAMPLES = {
        "train": 40836715,
        "train_12m": 12000000,
        "test": 3003,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
        use_12m_subset: bool = True,
    ):
        root = Path(root) if root is not None else Path.cwd()
        train_set = "train_12m" if use_12m_subset else "train"
        split = train_set if train else "test"
        self.path = root / "wmt14" / (self._FILENAMES[split])

        if self.path.is_file():
            return

        if download:
            download_from_url(self._URLS[split], filename=self.path)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )
