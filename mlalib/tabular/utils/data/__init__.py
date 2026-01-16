from ._dataloaders import load_dataset, LazyDataLoader, TabularDataLoader
from ._uci_utils import download_from_uci
from ._uci_utils import UCIDataLoader


__all__ = [
    "download_from_uci",
    "load_dataset",
    "LazyDataLoader",
    "TabularDataLoader",
    "UCIDataLoader",
]
