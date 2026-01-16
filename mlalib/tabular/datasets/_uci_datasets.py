from pathlib import Path
from typing import Any, Callable

from ..utils.data import UCIDataLoader


class Adult(UCIDataLoader):
    """
    UCI Adult dataset.

    Source:
    Becker, B and Kohavi, R. (1996).
    UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/2/adult

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            id=2,
            filename="adult.csv",
            root=root,
            transform=transform,
            download=download,
            **kwargs,
        )


class AirQuality(UCIDataLoader):
    """
    UCI Air quality dataset.

    Source:
    Vito, S. (2008).
    UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/360/air+quality

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            id=360,
            filename="air_quality.csv",
            root=root,
            transform=transform,
            download=download,
            **kwargs,
        )


class BankMarketing(UCIDataLoader):
    """
    UCI Bank Marketing dataset.

    Source:
    Moro et al. (2014).
    UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/222/bank+marketing

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            id=222,
            filename="heart_disease.csv",
            root=root,
            transform=transform,
            download=download,
            **kwargs,
        )


class HeartDisease(UCIDataLoader):
    """
    UCI Heart Disease dataset.

    Source:
    Janosi et al. (1989).
    UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/45/heart+disease

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            id=45,
            filename="heart_disease.csv",
            root=root,
            transform=transform,
            download=download,
            **kwargs,
        )


class Iris(UCIDataLoader):
    """
    UCI Iris dataset.

    Source:
        Fisher, R. A. (1936).
        UCI Machine Learning Repository.
        https://archive.ics.uci.edu/dataset/53/iris

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            id=53,
            filename="iris.csv",
            root=root,
            transform=transform,
            download=download,
            **kwargs,
        )


class OnlineRetail(UCIDataLoader):
    """
    UCI Online Retail dataset.

    Source:
    Chen, D. (2015).
    UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/352/online+retail

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            id=352,
            filename="online_retail.csv",
            root=root,
            transform=transform,
            download=download,
            **kwargs,
        )


class REV(UCIDataLoader):
    """
    UCI Real Estate Valuation dataset.

    Source:
    Yeh, I. (2018).
    UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            id=477,
            filename="rev.csv",
            root=root,
            transform=transform,
            download=download,
            **kwargs,
        )


class WDBC(UCIDataLoader):
    """
    UCI Breast Cancer Wisconsin Diagnostic dataset.

    Source:
    Wolberg et al. (1993).
    UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            id=17,
            filename="wdbc.csv",
            root=root,
            transform=transform,
            download=download,
            **kwargs,
        )


class WineQuality(UCIDataLoader):
    """
    UCI Wine Quality dataset.

    Source:
    Cortez et al. (2009).
    UCI Machine Learning Repository.
    https://archive.ics.uci.edu/dataset/186/wine+quality

    Args:
        root (str, Path or None): Optional directory where the dataset file is stored or to be downloaded.
        Uses current working directory if None. Defaults to None.
        transform (Callable or None): Optional transformation to apply to the dataset. Defaults to None.
        download (bool): Whether to download the dataset from the internet. Defaults to False.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        download: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            id=186,
            filename="wine_quality.csv",
            root=root,
            transform=transform,
            download=download,
            **kwargs,
        )
