import heapq
import random
import tempfile
import itertools

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, Iterable


class ExtSort(ABC):
    """
    Sort a file using external sort.

    Args:
        chunk_size: number of records to read at a time.
        Defaults to 500000.
    """

    def __init__(self, chunk_size=500000):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.chunk_size = chunk_size

    @abstractmethod
    def reader(self, path: str | Path) -> Iterable:
        """
        Return iterable of records from a file.

        Args:
            path (str or Path): Path to the input file.

        Returns:
            Iterable: An iterable of records.
        """

    @abstractmethod
    def writer(self, path: str | Path, items: Iterable):
        """
        Write an iterable of records to a file.

        Args:
            path (str or Path): Path to the output file.
            items: Iterable of records to write.

        """

    def sort(
        self,
        input_path: str | Path,
        output_path: str | Path,
        key: Callable | None = None,
        reverse: bool = False,
    ):
        """
        Sort a file using external sorting.

        Args:
            input_path (str or Path): Path to the input file.
            output_path (str or Path): Path to the output file.
            key (Callable or None): Optional key function for sorting.
            reverse (bool): If True, sort in descending order. Defaults to False.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_paths = []
            input_items = self.reader(input_path)

            while True:
                chunk = list(itertools.islice(input_items, self.chunk_size))
                if not chunk:
                    break

                chunk_path = Path(tmpdir) / f"chunk_{len(chunk_paths)}"
                chunk_paths.append(chunk_path)
                self.writer(chunk_path, sorted(chunk, key=key, reverse=reverse))

            sorted_chunks = [self.reader(path) for path in chunk_paths]

            self.writer(
                output_path, heapq.merge(*sorted_chunks, key=key, reverse=reverse)
            )


class ExtBatchedShuffler(ABC):
    """
    Shuffle batches of records in a file using bounded memory.
    Preserves record order within each batch while shuffling batches
    within chunks and shuffling the resulting chunks.

    Args:
        batch_size (int): Number of records per batch.
        Defaults to 8192.
        chunk_size (int): Maximum number of records to process in memory
        at once. Must be divisible by and greater than batch_size.
        Defaults to 499712.
        drop_last (bool): Whether to drop the last incomplete batch.
        seed (int or None): Optional random seed for shuffling.
    """

    def __init__(
        self,
        batch_size: int = 8192,
        chunk_size: int = 499712,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if chunk_size <= 0 or chunk_size < batch_size:
            raise ValueError("chunk_size must be positive and greater than batch_size")

        if chunk_size % batch_size != 0:
            raise ValueError("chunk_size must be divisible by batch_size")

        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.drop_last = drop_last
        self.rng = random.Random(seed)

    @abstractmethod
    def reader(self, path: str | Path) -> Iterable:
        """
        Return iterable of records from a file.

        Args:
            path (str or Path): Path to the input file.

        Returns:
            Iterable: An iterable of records.
        """

    @abstractmethod
    def writer(self, path: str | Path, items: Iterable):
        """
        Write an iterable of records to a file.

        Args:
            path (str or Path): Path to the output file.
            items: Iterable of records to write.

        """

    def shuffle(self, input_path: str | Path, output_path: str | Path):
        """
        Shuffle batches in a file using bounded memory.

        Args:
            input_path (str or Path): Path to the input file.
            output_path (str or Path): Path to the output file.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_paths = []
            input_items = self.reader(input_path)

            while True:
                chunk = list(
                    itertools.batched(
                        itertools.islice(input_items, self.chunk_size), self.batch_size
                    )
                )

                if self.drop_last and chunk and len(chunk[-1]) < self.batch_size:
                    chunk.pop()

                if not chunk:
                    break

                self.rng.shuffle(chunk)
                chunk_path = Path(tmpdir) / f"chunk_{len(chunk_paths)}"
                chunk_paths.append(chunk_path)

                self.writer(chunk_path, itertools.chain.from_iterable(chunk))

            self.rng.shuffle(chunk_paths)
            shuffled_items = itertools.chain.from_iterable(
                self.reader(path) for path in chunk_paths
            )
            self.writer(output_path, shuffled_items)
