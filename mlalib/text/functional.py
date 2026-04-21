import re
from typing import Any, List

import torch


def add_token(input: Any, token: int | str, begin: bool = True) -> Any:
    """
    Add a token to a list of integers (or strings) or a list of lists of integers (or strings).

    Args:
        input (Any): A list of integers (or strings) or a list of lists of integers (or strings).
        token (int | str): The token to add.
        begin (bool): If True, add the token at the beginning, otherwise at the end. Defaults to True.

    Returns:
        Any: A list of integers (or strings) or a list of lists of integers (or strings) with the token added.
    """
    if torch.jit.isinstance(input, List[str]) and isinstance(token, str):
        if begin:
            return [token] + input
        else:
            return input + [token]
    elif torch.jit.isinstance(input, List[int]) and isinstance(token, int):
        if begin:
            return [token] + input
        else:
            return input + [token]
    elif torch.jit.isinstance(input, List[List[int]]) and isinstance(token, int):
        if begin:
            return [[token] + seq for seq in input]
        else:
            return [seq + [token] for seq in input]
    elif torch.jit.isinstance(input, List[List[str]]) and isinstance(token, str):
        if begin:
            return [[token] + seq for seq in input]
        else:
            return [seq + [token] for seq in input]
    else:
        raise TypeError(
            "Input must be a sequence of strings or integers and token must match the type of the input."
        )


def regex_replace(input: Any, pattern: str, replacement: str) -> Any:
    """
    Replace all occurrences of a pattern in a string, list of strings or list of lists of strings
    with a replacement string.

    Args:
        input (Any): A string, list of strings or a list of lists of strings.
        pattern (str): The pattern to replace.
        replacement (str): The replacement string.

    Returns:
        Any: A string, list of strings or a list of lists of strings with the pattern replaced.
    """
    regex = re.compile(pattern)

    if isinstance(input, str):
        return regex.sub(replacement, input)

    if torch.jit.isinstance(input, List[str]):
        return [regex.sub(replacement, i) for i in input]

    elif torch.jit.isinstance(input, List[List[str]]):
        return [[regex.sub(replacement, i) for i in seq] for seq in input]

    else:
        raise TypeError(
            "Input must be a sequence of strings or a list of lists of strings."
        )


def str_to_int(input: Any) -> Any:
    """
    Convert a list of strings of a list of lists of strings to integers.

    Args:
        input (Any): A list of strings or a list of lists of strings.

    Returns:
        Any: A list of integers or a list of lists of integers.
    """
    if torch.jit.isinstance(input, List[str]):
        return [ord(t) for t in input]
    elif torch.jit.isinstance(input, List[List[str]]):
        return [[ord(t) for t in seq] for seq in input]
    else:
        raise TypeError(
            "Input must be a list of strings or a list of lists of strings."
        )


def to_tensor(
    input: list[int] | list[list[int]],
    padding_value: int | None = None,
    dtype: torch.dtype = torch.long,
    batch_first: bool = True,
) -> torch.Tensor:
    """
    Convert a list of integers or a list of lists of integers to a tensor.

    Args:
        input (list[int] or list[list[int]]): A list of integers or a list of lists of integers.
        padding_value (int | None): The padding value to use.
        dtype (torch.dtype): The data type of the tensor. Defaults to torch.long.
        batch_first (bool): If True returns B x T x *, otherwise T x B x *.
        Where B is batch size and T is length of the longest sequence in the batch.

    Returns:
        torch.Tensor: A tensor of integers.
    """
    if torch.jit.isinstance(input, List[int]):
        return torch.tensor(input, dtype=dtype)
    elif torch.jit.isinstance(input, List[List[int]]):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=dtype) for seq in input],
            batch_first=batch_first,
            padding_value=padding_value,
        )
    else:
        raise TypeError(
            "Input must be a list of integers (or strings) or a list of lists of integers (or strings)."
        )


def truncate(input: Any, max_length: int) -> Any:
    """
    Truncate a list of integers (or strings) or a list of lists of integers (or strings) to a maximum length.

    Args:
        input (Any): A list of integers (or strings) or a list of lists of integers (or strings).
        max_length (int): The maximum length to truncate to.

    Returns:
        Any: A truncated list of integers (or strings) or a list of lists of integers (or strings).
    """
    if torch.jit.isinstance(input, (List[str], List[int])):
        return input[:max_length]
    elif torch.jit.isinstance(input, (List[List[int]], List[List[str]])):
        return [seq[:max_length] for seq in input]
    else:
        raise TypeError(
            "Input must be a list of integers (or strings) or a list of lists of integers (or strings)."
        )
