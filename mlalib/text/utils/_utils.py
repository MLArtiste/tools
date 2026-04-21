from collections import Counter
from typing import Iterator
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.
    """

    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """
        Split string into tokens.

        Args:
            text (str): Input string.

        Returns:
            list[str]: List of tokens.
        """

        pass


class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer.
    """

    def tokenize(self, text: str) -> list[str]:
        """
        Split string into characters.

        Args:
            text (str): Input string.

        Returns:
            list[str]: List of characters.
        """
        return list(text)


class Vocab:
    """
    Vocabulary class for text processing.

    Args:
        counter (dict[str, int]): Dictionary of token frequencies.
        min_freq (int): Minimum frequency for tokens to be included. Defaults to 1.
        specials (list[str] or None): List of special tokens to include.
        special_first (bool): Whether to place special tokens at the beginning. Defaults to True.
    """

    def __init__(
        self,
        counter: dict[str, int],
        min_freq: int = 1,
        specials: list[str] | None = None,
        special_first: bool = True,
    ):
        counter = counter.copy()
        specials = specials or []
        core_specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        user_specials = [s for s in specials if s not in core_specials]

        for token in core_specials + user_specials:
            counter.pop(token, None)

        token_freqs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        tokens = [token for token, freq in token_freqs if freq >= min_freq]

        if special_first:
            self._idx_to_tokens = tuple(core_specials + user_specials + tokens)
        else:
            self._idx_to_tokens = tuple(core_specials + tokens + user_specials)

        self._tokens_to_idx = {
            token: idx for idx, token in enumerate(self._idx_to_tokens)
        }

        self.pad = self._tokens_to_idx["<pad>"]
        self.unk = self._tokens_to_idx["<unk>"]
        self.bos = self._tokens_to_idx["<bos>"]
        self.eos = self._tokens_to_idx["<eos>"]

    def __len__(self) -> int:
        """
        Return the length of the vocabulary.

        Returns:
            int: The number of tokens in the vocabulary.
        """
        return len(self._idx_to_tokens)

    def __contains__(self, token: str) -> bool:
        """
        Check if a token is in the vocabulary.

        Args:
            token (str): The token to check.

        Returns:
            bool: True if the token is in the vocabulary, False otherwise.
        """
        return token in self._tokens_to_idx

    def encode(self, token: str | list[str]) -> int | list[int]:
        """
        Get the index of a token or a list of tokens.

        Args:
            token (str or list[str]): The token or list of tokens to get the index of.

        Returns:
            int or list[int]: The index or list of indices of the token(s).
        """
        if isinstance(token, str):
            return self._tokens_to_idx.get(token, self.unk)
        else:
            return [self._tokens_to_idx.get(t, self.unk) for t in token]

    def decode(self, indices: int | list[int]) -> str | list[str]:
        """
        Convert indices to tokens.

        Args:
            indices (int or list[int]): The index or list of indices to convert to tokens.

        Returns:
            str or list[str]: The token or list of tokens.
        """
        if isinstance(indices, int):
            return self._idx_to_tokens[indices]
        else:
            return [self._idx_to_tokens[idx] for idx in indices]

    def get_stoi(self) -> dict[str, int]:
        """
        Get the string to index mapping.

        Returns:
            dict[str, int]: The string to index mapping.
        """
        return self._tokens_to_idx.copy()

    def get_itos(self) -> list[str]:
        """
        Get the index to string mapping.

        Returns:
            list[str]: The index to string mapping.
        """
        return self._idx_to_tokens


class WordTokenizer(Tokenizer):
    """
    Word-level tokenizer.
    """

    def tokenize(self, text: str) -> list[str]:
        """
        Split string into words.

        Args:
            text (str): Input string.

        Returns:
            list[str]: List of wordsd.
        """
        return text.split()


def build_counter_from_iterator(iterator: Iterator[str]) -> Counter[str]:
    """
    Build a counter from an iterator of tokens.

    Args:
        iterator (Iterator[str]): An iterator of tokens.

    Returns:
        Counter[str]: A counter of tokens.
    """
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    return counter


def build_vocab_from_iterator(
    iterator: Iterator[str],
    min_freq: int = 1,
    specials: list[str] | None = None,
    special_first: bool = True,
) -> Vocab:
    """
    Build a vocab from an iterator of tokens.

    Args:
        iterator (Iterator[str]): An iterator of tokens.
        min_freq (int): Minimum frequency for tokens to be included. Defaults to 1.
        specials (list[str] or None): List of special tokens to include.
        special_first (bool): Whether to place special tokens at the beginning. Defaults to True.

    Returns:
        Vocab: A vocab of tokens.
    """
    counter = build_counter_from_iterator(iterator)
    return Vocab(
        counter,
        min_freq=min_freq,
        specials=specials,
        special_first=special_first,
    )


def ngrams_iterator(
    tokens: list[str], n: int, only_n: bool = False, delimiter: str = " "
) -> Iterator[str]:
    """
    Args:
        tokens (list[str]): List of tokens.
        n (int): N-gram size.
        only_n (bool): Whether to only return n-grams of size n or all n-grams up to n.
        Defaults to False.
        delimiter (str): Delimiter to use for joining tokens. Defaults to " ".

    Returns:
        Iterator[str]: An iterator of n-grams.
    """

    def _get_ngram(n):
        for n_gram in zip(*[tokens[i:] for i in range(n)]):
            yield delimiter.join(n_gram)

    if only_n:
        yield from _get_ngram(n)
    else:
        for i in range(1, n + 1):
            yield from _get_ngram(i)
