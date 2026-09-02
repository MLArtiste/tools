from collections import Counter
from typing import Any, Callable, Iterator, Sequence

from ._vocab import Vocab


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
    return Vocab.from_counter(
        counter,
        min_freq=min_freq,
        specials=specials,
        specials_first=special_first,
    )


def ngrams_iterator(
    tokens: Sequence, n: int, only_n: bool = True, transform: Callable = tuple
) -> Iterator[Any]:
    """
    Args:
        tokens (Sequence): A sequence of tokens.
        n (int): N-gram size.
        only_n (bool): Whether to only return n-grams of size n instead of all n-grams up to n.
        Defaults to True.
        transform (Callable): A function applied to every n-gram tuple. Defaults to tuple.

    Returns:
        Iterator[Any]: An iterator of n-grams.
    """

    def _get_ngram(n):
        for n_gram in zip(*[tokens[i:] for i in range(n)]):
            yield transform(n_gram)

    if only_n:
        yield from _get_ngram(n)
    else:
        for i in range(1, n + 1):
            yield from _get_ngram(i)
