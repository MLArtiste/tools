import re
from typing import Any, List

import torch
from torch import Tensor
from torch.nn import Module

from ..utils import Tokenizer, Vocab
from .. import functional as F


class AddToken(Module):
    """
    Add a token to the input.

    Args:
        token (str or int): The token to add.
        begin (bool): If True, add the token at the beginning, otherwise at the end.
    """

    def __init__(self, token: str | int, begin: bool = False):
        super().__init__()
        self.token = token
        self.begin = begin

    def forward(self, input: Any) -> Any:
        """
        Args:
            input (Any): The input to add the token to.

        Returns:
            Any: The input with token added.
        """
        return F.add_token(input, self.token, self.begin)


class Compose:
    """
    Compose multiple transforms together.

    Args:
        transforms (list): A list of transforms to compose.
    """

    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = transforms

    def __call__(self, input: Any) -> Any:
        """
        Args:
            input (Any): The input to transform.

        Returns:
            Any: The transformed input.
        """
        for transform in self.transforms:
            input = transform(input)
        return input


class PadTransform(Module):
    """
    Pad tensor to a fixed length with a given padding value.

    Args:
        max_length (int): The maximum length to pad to.
        padding_value (int): The padding value to use.
    """

    def __init__(self, max_length: int, padding_value: int):
        super().__init__()
        self.max_length = max_length
        self.padding_value = padding_value

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): The input tensor to pad.

        Returns:
            Tensor: The padded tensor.
        """
        current_length = input.size(-1)
        if current_length < self.max_length:
            padding_length = self.max_length - current_length
            padding = torch.nn.functional.pad(
                input, (0, padding_length), value=self.padding_value
            )
            return padding
        return input


class RegexReplace(Module):
    """
    Replace all occurrences of a pattern in a string, list of a strings or list of lists of strings
    with a replacement string.

    Args:
        pattern (str): The pattern to replace.
        replacement (str): The replacement string.
    """

    def __init__(self, pattern: str, replacement: str):
        super().__init__()
        self.pattern = re.compile(pattern)
        self.replacement = replacement

    def forward(self, input: Any) -> Any:
        """
        Args:
            input (Any): The input to transform.

        Returns:
            Any: A string, list of strings or a list of lists of strings with the pattern replaced.
        """
        return F.regex_replace(input, self.pattern, self.replacement)


class StrToIntTransform(Module):
    """
    Convert a list or list of lists of strings to a list or list of lists of integers.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Any) -> Any:
        """
        Args:
            input (Any): List or list of lists of strings.

        Returns:
            Any: List or list of lists of integers.
        """
        return F.str_to_int(input)


class TokenizerTransform(Module):
    """
    Tokenizer transform to convert strings into tokens.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
    """

    def __init__(self, tokenizer: Tokenizer):
        super().__init__()
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError(f"expected Tokenizer object but got {type(tokenizer)}")
        self.tokenizer = tokenizer

    def forward(self, input: Any) -> Any:
        """
        Args:
            input (Any): Input to tokenize.

        Returns:
            Any: Tokenized input.
        """
        if torch.jit.isinstance(input, List[str]):
            return [self.tokenizer.tokenize(i) for i in input]
        elif torch.jit.isinstance(input, List[List[str]]):
            return [[self.tokenizer.tokenize(i) for i in strs] for strs in input]
        else:
            raise TypeError(
                f"expected str, List[str], or List[List[str]] but got {type(input)}"
            )


class ToTensor(Module):
    """
    Convert input to torch tensor.

    Args:
        padding_value (int or None): Padding value to use.
        dtype (torch.dtype): Data type of the tensor.
        batch_first (bool): If True returns B x T x *, otherwise T x B x *.
        Where B is batch size and T is length of the longest sequence in the batch.
    """

    def __init__(
        self,
        padding_value: int | None = None,
        dtype: torch.dtype = torch.long,
        batch_first: bool = True,
    ):
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype
        self.batch_first = batch_first

    def forward(self, input: Any) -> torch.Tensor:
        """
        Args:
            input (list[int] or list[list[int]]): Sequence or batch of token ids.

        Returns:
            Tensor: Corresponding tensor output.
        """
        return F.to_tensor(
            input,
            padding_value=self.padding_value,
            dtype=self.dtype,
            batch_first=self.batch_first,
        )


class Truncate(Module):
    """
    Truncate input to a maximum length.

    Args:
        max_length (int): The maximum length to truncate to.
    """

    def __init__(self, max_length: int):
        super().__init__()
        self.max_length = max_length

    def forward(self, input: Any) -> Any:
        """
        Args:
            input (Any): The input to truncate.

        Returns:
            Any: The truncated input.
        """
        return F.truncate(input, self.max_length)


class VocabTransform(Module):
    """
    Vocab transform to convert tokens to corresponding token ids.

    Args:
        vocab (Vocab): The vocabulary to use.
    """

    def __init__(self, vocab: Vocab):
        super().__init__()
        if not isinstance(vocab, Vocab):
            raise TypeError(f"expected Vocab object but got {type(vocab)}")
        self.vocab = vocab

    def forward(self, input: Any) -> Any:
        """
        Args:
            input (Any): The input to encode.

        Returns:
            Any: The encoded input.
        """
        if torch.jit.isinstance(input, List[str]):
            return self.vocab.encode(input)
        elif torch.jit.isinstance(input, List[List[str]]):
            return [self.vocab.encode(i) for i in input]
        else:
            raise TypeError(
                f"expected str, List[str], or List[List[str]] but got {type(input)}"
            )
