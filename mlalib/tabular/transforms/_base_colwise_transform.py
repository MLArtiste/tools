from typing import Any, Callable
from abc import ABC, abstractmethod

import pandas as pd


class ColwiseTransform(ABC):
    """
    Base class for column-wise transformas on a DataFrames/Series.

    Args:
        columns (str, list[str] or None): Optional column name or list of column names to apply
        the transform to or None for all. Defaults to None.
    """

    def __init__(self, columns: str | list[str] | None = None):

        self._columns = [columns] if isinstance(columns, str) else columns

    def __call__(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Apply the transformation to a DataFrame or Series.

        Args:
            data (pd.DataFrame or pd.Series): Input data to transform.

        Returns:
            pd.DataFrame or pd.Series: Transformed data.
        """
        return self._call_func_on_data(self.compute, data)

    def invert(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Apply the inverse of the transformation to the given data.

        Args:
            data (pd.DataFrame or pd.Series): Data to invert.

        Returns:
            pd.DataFrame or pd.Series: Inverted Data.
        """
        if self.compute_inverse is ColwiseTransform.compute_inverse:
            raise NotImplementedError("transformation does not have an inverse")

        return self._call_func_on_data(self.compute_inverse, data)

    @abstractmethod
    def compute(self, column: pd.Series) -> pd.Series:
        """
        Compute the transformation for a single column.

        Args:
            column (pd.Series): Column to transform.

        Returns:
            pd.Series: Transformed column.
        """

    def compute_inverse(self, column: pd.Series) -> pd.Series:
        """
        Compute the inverse transformation for a single column.

        Args:
            column (pd.Series): Column to invert.

        Returns:
            pd.Series: Inverted Column.
        """
        return None

    def _call_func_on_data(
        self, func: Callable, data: pd.DataFrame | pd.Series
    ) -> pd.DataFrame | pd.Series:
        """
        Apply a colwise-wise function to DataFrame or Series.

        Args:
            func (Callable): Function to apply.
            data (pd.DataFrame or pd.Series): Data being transformed.

        Returns:
            (pd.DataFrame or pd.Series): Transformed data.
        """
        if isinstance(data, pd.Series):
            if self._columns is not None:
                if len(self._columns) != 1:
                    raise KeyError("expected exactly one column name for Series")
                elif self._columns[0] != data.name:
                    raise KeyError(
                        f"expected Series named '{self._columns[0]}', "
                        f"but received Series named '{data.name}'"
                    )
            return func(data.copy())

        df = data.copy()
        cols = self._columns or df.columns

        for col in cols:
            df[col] = func(df[col])

        return df


class StatefulColwiseTransform(ColwiseTransform, ABC):
    """
    Abstract base class for defining column-wise transformations
    on DataFrames or Series that can save and reuse parameters.

    Args:
        columns (str, list[str] or None): Optional column name or list of column names to apply
        the transform to or None for all. Defaults to None.
        retain_params (bool): Whether to store parameters. Defaults to True.
    """

    def __init__(
        self, columns: str | list[str] | None = None, retain_params: bool = True
    ):
        super().__init__(columns)

        self.retain_params = retain_params
        self.params = {}

    @abstractmethod
    def compute_params(self, column: pd.Series) -> dict[str, Any]:
        """
        Compute parameters required to transform a single column.

        Args:
            column (pd.Series): Column for which parameters should be computed.

        Returns:
            dict[str, Any]: A dictionary of transformation parameters ({"param_name": value}).
        """

    def _get_params(self, column: pd.Series) -> dict[str, Any]:
        """
        Retrieve stored parameters for a given column.

        Args:
            column (pd.Series): Column whose parameters are requested.

        Returns:
            dict[str, Any]: Stored parameters for this column.
        """
        col_name = column.name

        if not self.params:
            raise ValueError(
                "no parameters have been stored. "
                "Did you mean to set retain_params=True?"
            )
        if col_name not in self.params:
            raise ValueError(f"no stored parameters found for '{col_name}'")

        return self.params[col_name]

    def _get_or_compute_params(self, column: pd.Series) -> dict[str, Any]:
        """
        Retrieve stored parameters for a column if avaialble, otherwise compute them.
        If 'retain_params=True', computed parameters are stored for future use.

        Args:
            column (pd.Series): Column whose parameters are requested.

        Returns:
            dict[str, Any]: Stored parameters for this column.
        """
        col_name = column.name

        if col_name in self.params:
            return self._get_params(column)

        new_params = self.compute_params(column)

        if self.retain_params:
            self.params[col_name] = new_params

        return new_params

    def fit(self, data: pd.DataFrame | pd.Series) -> None:
        """
        Compute and store parameters for all relevant columns.

        Args:
            data (pd.DataFrame or pd.Series): Data to compute parameters for.
        """
        if isinstance(data, pd.Series):
            if self._columns is not None:
                if len(self._columns) != 1:
                    raise KeyError("expected exactly one column name for Series")
                elif self._columns[0] != data.name:
                    raise KeyError(
                        f"expected Series named '{self._columns[0]}', "
                        f"but received Series named '{data.name}'"
                    )
            self._get_or_compute_params(data)
            return

        if isinstance(data, pd.DataFrame):
            for col in self._columns or data.columns:
                self._get_or_compute_params(data[col])
            return

        raise TypeError("fit() expects pandas DataFrame or Series")
