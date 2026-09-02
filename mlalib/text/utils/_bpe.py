"""
This module is highly inspired by Andrej Karpathy's minBPE.
"""

import pickle
from collections import Counter
from pathlib import Path
from typing import Literal

import regex
from tqdm import tqdm

from ._utils import ngrams_iterator


class BPE:
    """
    Byte Pair Encoding (BPE) implementation.

    Args:
        split_pattern (str or None): Optional regular expression pattern used to split text before BPE.
        Use "gpt4" for GPT-4 split pattern and None for no split.
        vocab_merges (tuple or None): Optional tuple of vocab and merges dictionaries.
        specials (list[str] or dict[str, int] or None): Optional list of special tokens or dictionary
        of special tokens and their indices.
    """

    _GPT4_SPLIT_PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
        r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )

    def __init__(
        self,
        split_pattern: Literal["gpt4"] | None | str = "gpt4",
        vocab_merges: tuple[dict[int, bytes], dict[tuple[int, int], int]] | None = None,
        specials: list[str] | dict[str, int] | None = None,
    ):

        if split_pattern == "gpt4":
            self._split_pattern = regex.compile(self._GPT4_SPLIT_PATTERN)
        elif split_pattern is None:
            self._split_pattern = split_pattern
        else:
            self._split_pattern = regex.compile(split_pattern)

        if vocab_merges is not None:
            self.vocab, self.merges = vocab_merges
        else:
            self.vocab = {}
            self.merges = {}

        if specials is None:
            self._specials_type = None
            self.specials_to_idx = {}
        elif isinstance(specials, dict):
            self._specials_type = dict
            self.specials_to_idx = specials.copy()
        else:
            self._specials_type = list
            self.specials_to_idx = {special: None for special in specials}

        if vocab_merges is not None:
            self._update_vocab_with_specials(
                self.vocab, self.specials_to_idx, self._specials_type
            )

    def train(
        self,
        *,
        text: str | None = None,
        path: str | Path | None = None,
        vocab_size: int = 10000,
    ):
        """
        Train BPE vocabulary and merges.

        Args:
            text: (str): Optional text to train BPE on.
            path: (str or Path): Optional path to text file to train BPE on.
            vocab_size: (int): Size of vocabulary. Defaults to 10000.
        """
        if vocab_size < 256:
            raise ValueError("'vocab_size' must be at least 256")

        if text is not None and path is not None:
            raise ValueError("either 'text' or 'path' must be provided, but not both")

        if path is not None:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        num_merges = vocab_size - 256
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        if self._split_pattern is None:
            token_ids = [list(text.encode("utf-8"))]
        else:
            text_chunks = regex.findall(self._split_pattern, text)
            token_ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        for i in tqdm(range(num_merges)):
            counter = Counter()
            for chunk_ids in token_ids:
                bigrams = ngrams_iterator(chunk_ids, n=2)
                counter.update(bigrams)

            if not counter:
                print(
                    f"Stopped at a vocab size of {len(self.vocab):,}. No more pairs to merge."
                )
                break
            pair = counter.most_common(1)[0][0]
            idx = len(self.vocab)
            token_ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in token_ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        self._update_vocab_with_specials(
            self.vocab, self.specials_to_idx, self._specials_type
        )

    def encode(self, text: str) -> list[int]:
        """
        Encode text to list of token IDs.

        Args:
            text (str): Text to encode.

        Returns:
            list[int]: List of token IDs.
        """
        if len(self.specials_to_idx) == 0:
            return self._encode_without_specials(text)

        else:
            sorted_specials = sorted(self.specials_to_idx.keys(), key=len, reverse=True)
            special_pattern = (
                "(" + "|".join(regex.escape(k) for k in sorted_specials) + ")"
            )
            special_chunks = regex.split(special_pattern, text)
            ids_with_specials = []
            for part in special_chunks:
                if part in self.specials_to_idx:
                    ids_with_specials.append(self.specials_to_idx[part])
                else:
                    ids_with_specials.extend(self._encode_without_specials(part))
            return ids_with_specials

    def _encode_without_specials(self, text: str) -> list[int]:
        """
        Encode text to list of token IDs without special token handling.

        Args:
            text (str): Text to encode.

        Returns:
            list[int]: List of token IDs.
        """
        if self._split_pattern is None:
            text_chunks = [text]
        else:
            text_chunks = regex.findall(self._split_pattern, text)

        token_ids = []
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                bigrams = ngrams_iterator(chunk_ids, n=2)
                pair = min(
                    bigrams, key=lambda pair: self.merges.get(pair, float("inf"))
                )
                if pair not in self.merges:
                    break
                token_idx = self.merges[pair]
                chunk_ids = self._merge(chunk_ids, pair, token_idx)
            token_ids.extend(chunk_ids)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode list of token IDs to text.

        Args:
            token_ids (list[int]): List of token IDs to decode.

        Returns:
            str: Decoded text.
        """
        token_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        text = token_bytes.decode("utf-8", errors="replace")
        return text

    def _merge(
        self, token_ids: list[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        """
        Merge a pair of adjacent token IDs in a list of token IDs
        by replacing them with a new token ID.

        Args:
            token_ids (list[int]): List of token IDs.
            pair (tuple[int, int]): Pair of tokens to merge.
            new_id (int): New token ID.

        Returns:
            list[int]: List of token IDs with the merged token.
        """
        new_token_ids = []
        i = 0
        while i < len(token_ids):
            if (
                i < len(token_ids) - 1
                and token_ids[i] == pair[0]
                and token_ids[i + 1] == pair[1]
            ):
                new_token_ids.append(new_id)
                i = i + 2
            else:
                new_token_ids.append(token_ids[i])
                i = i + 1
        return new_token_ids

    def _update_vocab_with_specials(
        self,
        vocab: dict[int, bytes],
        specials: dict[str, int | None],
        specials_type: dict | list | None = dict,
    ):
        """
        Update the vocabulary with special tokens.

        Args:
            vocab (dict[int, bytes]): The vocabulary dictionary.
            specials (dict[str, int]): The dictionary of special tokens.
            specials_type (dict, list or None): Data type of specials provided
            during initialization.
        """
        if specials_type is dict:
            for key, value in specials.items():
                key = key.encode("utf-8")
                if value in vocab:
                    raise ValueError(f"index {value} already exists")
                vocab[value] = key

        elif specials_type is list:
            for special in specials.keys():
                new_idx = len(self.vocab)
                self.vocab[new_idx] = special.encode("utf-8")
                specials[special] = new_idx
        else:
            return

    def save(self, path: str | Path):
        """
        Save the BPE states to a file.

        Args:
            path (str or Path): Path to save the file to.
        """
        state = {
            "vocab": self.vocab,
            "merges": self.merges,
            "split_pattern": (
                None if self._split_pattern is None else self._split_pattern.pattern
            ),
            "specials_to_idx": self.specials_to_idx,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path):
        """
        Load the BPE state from a file.

        Args:
            path (str or Path): Path to load the file from.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        obj = cls()

        obj.vocab = state["vocab"]
        obj.merges = state["merges"]
        pattern = state["split_pattern"]
        obj._split_pattern = None if pattern is None else regex.compile(pattern)
        obj.specials_to_idx = state["specials_to_idx"]

        return obj
