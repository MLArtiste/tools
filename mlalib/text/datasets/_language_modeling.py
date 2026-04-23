from pathlib import Path
from typing import Literal

from ...utils import download_and_extract_tar, download_from_url


class PennTreebank:
    """
    PennTreebank Dataset.

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        split (str): Split to load. e.g ("train", "val", "test"). Defaults to "train".
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = {
        "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
        "val": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
        "test": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
    }
    NUM_LINES = {
        "train": 42068,
        "val": 3370,
        "test": 3761,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        split: Literal["train", "val", "test"] = "train",
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        paths = {
            "train": root / "penn_tree_bank" / "train.txt",
            "val": root / "penn_tree_bank" / "val.txt",
            "test": root / "penn_tree_bank" / "test.txt",
        }
        self.path = paths[split]

        if self.path.is_file():
            return

        if download:
            for split in paths:
                download_from_url(self._URL[split], filename=paths[split])

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class TimeMachine:
    """
    TimeMachine dataset.

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"

    def __init__(
        self,
        root: str | Path | None = None,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "time_machine" / "timemachine.txt"

        if self.path.is_file():
            return

        if download:
            filename = f"time_machine/timemachine.txt"
            download_from_url(self._URL, root=root, filename=filename)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class WikiText2:
    """
    WikiText2 Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://s3.amazonaws.com/fast-ai-nlp/wikitext-2.tgz"

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "wikitext-2" / ("train.csv" if train else "test.csv")

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="wikitext-2.tgz",
                mode="gz",
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class WikiText103:
    """
    WikiText103 Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz"

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "wikitext-103" / ("train.csv" if train else "test.csv")

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="wikitext-103.tgz",
                mode="gz",
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )
