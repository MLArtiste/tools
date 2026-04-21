from pathlib import Path
from typing import Literal

from ...utils import (
    download_from_url,
    download_and_extract_tar,
    download_and_extract_zip,
)


class AG_News:
    """
    AG News dataset.

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    _TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    NUM_LINES = {
        "train": 120000,
        "test": 7600,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "ag_news_csv" / ("train.csv" if train else "test.csv")

        if self.path.is_file():
            return

        if download:
            train_filename = f"ag_news_csv/train.csv"
            test_filename = f"ag_news_csv/test.csv"
            download_from_url(self._TRAIN_URL, root=root, filename=train_filename)
            download_from_url(self._TEST_URL, root=root, filename=test_filename)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class AmazonReviewFull:
    """
    AmazonReviewFull Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA"
    NUM_LINES = {
        "train": 3000000,
        "test": 650000,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = (
            root / "amazon_review_full_csv" / ("train.csv" if train else "test.csv")
        )

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="amazon_review_full_csv.tar.gz",
                mode="gz",
                from_gdrive=True,
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class AmazonReviewPolarity:
    """
    AmazonReviewPolarity Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"
    NUM_LINES = {
        "train": 3600000,
        "test": 400000,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = (
            root / "amazon_review_polarity_csv" / ("train.csv" if train else "test.csv")
        )

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="amazon_review_polarity_csv.tar.gz",
                mode="gz",
                from_gdrive=True,
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class CoLA:
    """
    CoLA Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        split (str): Split to load. e.g ("train", "val", "test"). Defaults to "train".
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"
    NUM_LINES = {
        "train": 8551,
        "val": 527,
        "test": 516,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        split: Literal["train", "val", "test"] = "train",
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        paths = {
            "train": root / "cola_public" / "raw" / "in_domain_train.tsv",
            "val": root / "cola_public" / "raw" / "in_domain_dev.tsv",
            "test": root / "cola_public" / "raw" / "out_of_domain_dev.tsv",
        }
        self.path = paths[split]

        if self.path.is_file():
            return

        if download:
            download_and_extract_zip(self._URL, root=root, filename="cola_public.zip")

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class DBpedia:
    """
    DBpedia Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k"
    NUM_LINES = {
        "train": 560000,
        "test": 70000,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "dbpedia_csv" / ("train.csv" if train else "test.csv")

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="dbpedia_csv.tar.gz",
                mode="gz",
                from_gdrive=True,
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class IMDB:
    """
    IMDB Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    NUM_LINES = {
        "train": 25000,
        "test": 25000,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "aclImdb" / ("train" if train else "test")

        if self.path.is_dir() and any(self.path.iterdir()):
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="aclImdb_v1.tar.gz",
                mode="gz",
                from_gdrive=True,
            )

        if not self.path.is_dir():
            raise FileNotFoundError(
                f"Dataset directory {self.path} not found. "
                "Set download=True to download it."
            )


class MNLI:
    """
    MNLI Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        split (str): Split to load. e.g ("train", "test_matched", "test_mismatched"). Defaults to "train".
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    NUM_LINES = {
        "train": 392702,
        "test_matched": 9815,
        "test_mismatched": 9832,
    }
    LABEL_TO_INT = {"entailment": 0, "neutral": 1, "contradiction": 2}

    def __init__(
        self,
        root: str | Path | None = None,
        split: Literal["train", "test_matched", "test_mismatched"] = "train",
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        paths = {
            "train": root / "multinli_1.0" / "multinli_1.0_train.txt",
            "test_matched": root / "multinli_1.0" / "multinli_1.0_dev_matched.txt",
            "test_mismatched": root
            / "multinli_1.0"
            / "multinli_1.0_dev_mismatched.txt",
        }
        self.path = paths[split]

        if self.path.is_file():
            return

        if download:
            download_and_extract_zip(self._URL, root=root, filename="multinli_1.0.zip")

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class MRPC:
    """
    MRPC dataset.

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _TRAIN_URL = (
        "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
    )
    _TEST_URL = (
        "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"
    )
    NUM_LINES = {
        "train": 4076,
        "test": 1725,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "MRPC" / ("train.csv" if train else "test.csv")

        if self.path.is_file():
            return

        if download:
            train_filename = f"MRPC/train.csv"
            test_filename = f"MRPC/test.csv"
            download_from_url(self._TRAIN_URL, root=root, filename=train_filename)
            download_from_url(self._TEST_URL, root=root, filename=test_filename)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class QNLI:
    """
    QNLI Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        split (str): Split to load. e.g ("train", "val", "test"). Defaults to "train".
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip"
    NUM_LINES = {
        "train": 104743,
        "val": 5463,
        "test": 5463,
    }
    LABEL_TO_INT = {"entailment": 0, "not_entailment": 1}

    def __init__(
        self,
        root: str | Path | None = None,
        split: Literal["train", "val", "test"] = "train",
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        paths = {
            "train": root / "QNLI" / "train.tsv",
            "val": root / "QNLI" / "dev.tsv",
            "test": root / "QNLI" / "test.tsv",
        }
        self.path = paths[split]

        if self.path.is_file():
            return

        if download:
            download_and_extract_zip(self._URL, root=root)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class QQP:
    """
    QQP dataset.

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"

    NUM_LINES = 404290

    def __init__(
        self,
        root: str | Path | None = None,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "QQP" / "quora_duplicate_questions.tsv"

        if self.path.is_file():
            return

        if download:
            filename = f"QQP/quora_duplicate_questions.tsv"
            download_from_url(self._URL, root=root, filename=filename)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class RTE:
    """
    RTE Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        split (str): Split to load. e.g ("train", "val", "test"). Defaults to "train".
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://dl.fbaipublicfiles.com/glue/data/RTE.zip"
    NUM_LINES = {
        "train": 2490,
        "val": 277,
        "test": 3000,
    }
    LABEL_TO_INT = {"entailment": 0, "not_entailment": 1}

    def __init__(
        self,
        root: str | Path | None = None,
        split: Literal["train", "val", "test"] = "train",
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        paths = {
            "train": root / "RTE" / "train.tsv",
            "val": root / "RTE" / "dev.tsv",
            "test": root / "RTE" / "test.tsv",
        }
        self.path = paths[split]

        if self.path.is_file():
            return

        if download:
            download_and_extract_zip(self._URL, root=root)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class SogouNews:
    """
    SogouNews Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE"
    NUM_LINES = {
        "train": 450000,
        "test": 60000,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "sogou_news_csv" / ("train.csv" if train else "test.csv")

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="sogou_news_csv.tar.gz",
                mode="gz",
                from_gdrive=True,
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class SST2:
    """
    SST2 Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        split (str): Split to load. e.g ("train", "val", "test"). Defaults to "train".
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    NUM_LINES = {
        "train": 67349,
        "val": 872,
        "test": 1821,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        split: Literal["train", "val", "test"] = "train",
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        paths = {
            "train": root / "SST-2" / "train.tsv",
            "val": root / "SST-2" / "dev.tsv",
            "test": root / "SST-2" / "test.tsv",
        }
        self.path = paths[split]

        if self.path.is_file():
            return

        if download:
            download_and_extract_zip(self._URL, root=root)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class STSB:
    """
    STSB Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        split (str): Split to load. e.g ("train", "val", "test"). Defaults to "train".
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
    NUM_LINES = {
        "train": 5749,
        "val": 1500,
        "test": 1379,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        split: Literal["train", "val", "test"] = "train",
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        paths = {
            "train": root / "stsbenchmark" / "sts-train.csv",
            "val": root / "stsbenchmark" / "sts-dev.csv",
            "test": root / "stsbenchmark" / "sts-test.csv",
        }
        self.path = paths[split]

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(self._URL, root=root, mode="gz")

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class WNLI:
    """
    WNLI Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        split (str): Split to load. e.g ("train", "val", "test"). Defaults to "train".
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip"
    NUM_LINES = {
        "train": 635,
        "val": 71,
        "test": 146,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        split: Literal["train", "val", "test"] = "train",
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        paths = {
            "train": root / "WNLI" / "train.tsv",
            "val": root / "WNLI" / "dev.tsv",
            "test": root / "WNLI" / "test.tsv",
        }
        self.path = paths[split]

        if self.path.is_file():
            return

        if download:
            download_and_extract_zip(self._URL, root=root)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class YahooAnswers:
    """
    YahooAnswers Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9Qhbd2JNdDBsQUdocVU"
    NUM_LINES = {
        "train": 1400000,
        "test": 60000,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = root / "yahoo_answers_csv" / ("train.csv" if train else "test.csv")

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="yahoo_answers_csv.tar.gz",
                mode="gz",
                from_gdrive=True,
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class YelpReviewFull:
    """
    YelpReviewFull Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0"
    NUM_LINES = {
        "train": 650000,
        "test": 50000,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = (
            root / "yelp_review_full_csv" / ("train.csv" if train else "test.csv")
        )

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="yelp_review_full_csv.tar.gz",
                mode="gz",
                from_gdrive=True,
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )


class YelpReviewPolarity:
    """
    YelpReviewPolarity Dataset

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        train (bool): Whether to load the training set. Defaults to True.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    _URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg"
    NUM_LINES = {
        "train": 560000,
        "test": 38000,
    }

    def __init__(
        self,
        root: str | Path | None = None,
        train: bool = True,
        download: bool = False,
    ):
        root = Path(root) if root is not None else Path.cwd()
        self.path = (
            root / "yelp_review_polarity_csv" / ("train.csv" if train else "test.csv")
        )

        if self.path.is_file():
            return

        if download:
            download_and_extract_tar(
                self._URL,
                root=root,
                filename="yelp_review_polarity_csv.tar.gz",
                mode="gz",
                from_gdrive=True,
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset file {self.path} not found. "
                "Set download=True to download it."
            )
