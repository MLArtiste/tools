import re
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


class WordTokenizer(Tokenizer):
    """
    Word-level tokenizer.

    Args:
        keep_punctuation (bool): Whether or not to keep punctuations. Defaults to True.

    """

    def __init__(self, keep_punctuation: bool = True):
        if keep_punctuation:
            pattern = "[A-Za-z]+|\\d+|[{-~\\[-`:-@!-\\/]"
        else:
            pattern = "[A-Za-z]+|\\d+"

        self.token_regex = re.compile(pattern)

    def tokenize(self, text: str) -> list[str]:
        """
        Split string into words.

        Args:
            text (str): Input string.

        Returns:
            list[str]: List of words.
        """
        return self.token_regex.findall(text)