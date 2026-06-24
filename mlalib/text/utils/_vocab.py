import json
from pathlib import Path


import torch


class Vocab:
    """
    Vocabulary class for text processing.

    Args:
        token_to_idx (dict[str, int] or None): Dictionary mapping tokens to indices.
        idx_to_token (dict[int, str] or None): Dictionary mapping indices to tokens.
    """

    def __init__(
        self,
        token_to_idx: dict[str, int] | None = None,
        idx_to_token: dict[int, str] | None = None,
    ):
        if token_to_idx is not None and not isinstance(token_to_idx, dict):
            raise ValueError("token_to_idx must be a dictionary")

        if idx_to_token is not None and not isinstance(idx_to_token, dict):
            raise ValueError("idx_to_token must be a dictionary")

        provided = (token_to_idx is not None) + (idx_to_token is not None)

        if provided != 1:
            raise ValueError(
                "only one of token_to_idx or idx_to_token must be provided"
            )

        if token_to_idx is not None:
            if len(set(token_to_idx.values())) != len(token_to_idx):
                raise ValueError("indices must be unique")

            if not torch.jit.isinstance(token_to_idx, dict[str, int]):
                raise TypeError("token_to_idx must be a dictionary of str to int")

            self._token_to_idx = token_to_idx.copy()
            self._idx_to_token = {v: k for k, v in self._token_to_idx.items()}

        if idx_to_token is not None:
            if len(set(idx_to_token.values())) != len(idx_to_token):
                raise ValueError("tokens must be unique")

            if not torch.jit.isinstance(idx_to_token, dict[int, str]):
                raise TypeError("idx_to_token must be a dictionary of int to str")

            self._idx_to_token = idx_to_token.copy()
            self._token_to_idx = {v: k for k, v in self._idx_to_token.items()}

        self.pad = self._token_to_idx.get("<pad>")
        self.unk = self._token_to_idx.get("<unk>")
        self.bos = self._token_to_idx.get("<bos>")
        self.eos = self._token_to_idx.get("<eos>")

    @classmethod
    def from_counter(
        cls,
        counter: dict[str, int],
        min_freq: int = 1,
        specials: list[str] | None = None,
        specials_first: bool = True,
        use_core_specials: bool = True,
    ):
        """
        Construct a Vocab object from a Counter object.

        Args:
            counter (dict[str, int]): Dictionary of token frequencies.
            min_freq (int): Minimum frequency for tokens to be included. Defaults to 1.
            specials (list[str] or None): List of special tokens to include.
            specials_first (bool): Whether to place `specials` at the beginning of the vocabulary.
            after core specials. Defaults to True.
            use_core_specials (bool): Whether to add core special tokens (<pad>, <unk>, <bos>, <eos>)
            to the vocabulary. Defaults to True. Core specials are always added at the beginning of the
            vocabulary.

        Returns:
            Vocab: A Vocab object.

        """
        counter = counter.copy()
        specials = specials or []
        core_specials = (
            ["<pad>", "<unk>", "<bos>", "<eos>"] if use_core_specials else []
        )
        user_specials = [s for s in specials if s not in core_specials]

        for token in core_specials + user_specials:
            counter.pop(token, None)

        token_freqs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        tokens = [token for token, freq in token_freqs if freq >= min_freq]

        if specials_first:
            idx_to_token = tuple(core_specials + user_specials + tokens)
        else:
            idx_to_token = tuple(core_specials + tokens + user_specials)

        return cls(token_to_idx={token: idx for idx, token in enumerate(idx_to_token)})

    def __len__(self) -> int:
        """
        Return the length of the vocabulary.

        Returns:
            int: The number of tokens in the vocabulary.
        """
        return len(self._idx_to_token)

    def __contains__(self, token: str) -> bool:
        """
        Check if a token is in the vocabulary.

        Args:
            token (str): The token to check.

        Returns:
            bool: True if the token is in the vocabulary, False otherwise.
        """
        return token in self._token_to_idx

    def encode(self, token: str | list[str]) -> int | None | list[int | None]:
        """
        Get the index of a token or a list of tokens.

        Args:
            token (str or list[str]): The token or list of tokens to get the index of.

        Returns:
            int or list[int]: The index or list of indices of the token(s).
        """
        if isinstance(token, str):
            return self._token_to_idx.get(token, self.unk)
        else:
            return [self._token_to_idx.get(t, self.unk) for t in token]

    def decode(self, indices: int | list[int]) -> str | list[str]:
        """
        Convert indices to tokens.

        Args:
            indices (int or list[int]): The index or list of indices to convert to tokens.

        Returns:
            str or list[str]: The token or list of tokens.
        """
        if isinstance(indices, int):
            return self._idx_to_token[indices]
        else:
            return [self._idx_to_token[idx] for idx in indices]

    def get_token_to_idx(self) -> dict[str, int]:
        """
        Get the token to index mapping.

        Returns:
            dict[str, int]: The string to index mapping.
        """
        return self._token_to_idx.copy()

    def get_idx_to_token(self) -> dict[int, str]:
        """
        Get the index to token mapping.

        Returns:
            dict[int, str]: The index to string mapping.
        """
        return self._idx_to_token.copy()

    def save(self, path: str | Path) -> None:
        """
        Save vocabulary to a JSON file.

        Args:
            path: (str or Path): Path to save vocabulary.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self._token_to_idx,
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        """
        Load vocabulary from a JSON file.

        Args:
            path: (str or Path): Path to load vocabulary.

        Returns:
            Vocab: Loaded vocabulary.
        """
        with open(path, "r", encoding="utf-8") as f:
            token_to_idx = json.load(f)
        return cls(token_to_idx=token_to_idx)
