from typing import Any, Callable, Union

import torch
import numpy as np
import pandas as pd

from ._base_colwise_transform import ColwiseTransform, StatefulColwiseTransform


PandasObj = Union[pd.DataFrame, pd.Series]


class ColumnDropper:
    """
    Drop the specified columns from the DataFrame.

    Args:
        columns (list[str]): List of columns to drop.
    """

    def __init__(self, columns: list[str]):
        self._columns = columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame without specified columns.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with specified columns removed.
        """

        return df.drop(columns=self._columns)


class ColumnSelector:
    """
    Select (keep) only specified columns.

    Args:
        columns (str or list[str]): Name(s) of column(s) to keep.
    """

    def __init__(self, columns: str | list[str]):
        self._columns = columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame containing only selected columns.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: a New DataFrame containing only specified columns.
        """
        return df[self._columns].copy()


class Compose:
    """
    A transform pipeline that applies a sequence of callable transforms to
    a pandas DataFrame or Series.

    Args:
        transforms (list[Callable]): List of transforms to be applied sequentially.
    """

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Apply all transforms in order to the input data.

        Args:
            data (pd.DataFrame or pd.Series): Input data.

        Returns:
            pd.DataFrame or pd.Series: Transformed data.
        """
        for transform in self.transforms:
            data = transform(data)
        return data


class DropNaColumns:
    """
    Drop columns that contain NaN values.
    """

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with missing values from the input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with columns containing NaNs removed.
        """
        return df.dropna(axis=1)


class DropNaRows:
    """
    Drop rows that contain NaN in the specified columns.

    Args:
        columns (str, list[str] or None): Column(s) to check for NaN values. All if None. Defaults to None.
    """

    def __init__(self, columns: str | list[str] | None = None):
        self._columns = [columns] if isinstance(columns, str) else columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with NaN in the given columns.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with rows containing NaN in the specified columns removed.
        """
        cols = self._columns or df.columns
        return df.dropna(subset=cols)


class OneHotEncoder:
    """
    One-hot encode specified columns.

    Args:
        columns (str, list[str] or None): Column(s) to one-hot encode or None for all. Defaults to None.
        drop_first (bool): Whether to drop the first category in each encoded column to avoid collinearity.
        Defaults to False.
        dtype (type): Data type for the resulting one-hot encoded columns. Defaults to np.uint8.
    """

    def __init__(
        self,
        columns: str | list[str] | None = None,
        drop_first: bool = False,
        dtype: type = np.uint8,
    ):
        self._columns = [columns] if isinstance(columns, str) else columns
        self._drop_first = drop_first
        self._dtype = dtype

    def __call__(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to Series or to selected columns of the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with selected columns one-hot encoded.
        """
        if isinstance(data, pd.Series):
            if self._columns is not None:
                if len(self._columns) != 1:
                    raise KeyError("expected exactly one column name for Series")

                elif self._columns[0] != data.name:
                    raise KeyError(
                        f"expected Series name '{data.name}', "
                        f"but received '{self._columns[0]}'"
                    )

            return pd.get_dummies(
                data,
                drop_first=self._drop_first,
                dtype=self._dtype,
            )

        if isinstance(data, pd.DataFrame):
            cols = self._columns or data.columns
            return pd.get_dummies(
                data,
                columns=cols,
                drop_first=self._drop_first,
                dtype=self._dtype,
            )


class SelectDtypes:
    """
    Select columns from a DataFrame based on pandas dtypes.

    Args:
        include (str, list[str], or None): Dtypes to include.
        exclude (str, list[str], or None): Dtypes to exclude.
    """

    def __init__(
        self, include: str | list[str] | None, exclude: str | list[str] | None
    ):
        if include is None and exclude is None:
            raise ValueError("at least one of 'include' or 'exclude' must be specified")
        self._include = include
        self._exclude = exclude

    def __call__(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Select columns from the DataFrame based on dtype inclusion or exclusion

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            (pd.DataFrame or pd.Series): Output containing only the selected columns.
        """
        return df.select_dtypes(include=self._include, exclude=self._exclude)


class SplitXY:
    """
    Split a DataFrame into features and target.

    Args:
        target (str or list[str]): Name of target or list of names of targets.
    """

    def __init__(self, target: str | list[str]):
        self._target = [target] if isinstance(target, str) else target

    def __call__(self, df: pd.DataFrame) -> tuple[PandasObj, PandasObj]:
        """
        Split DataFrame into X and y.

        Args:
            df(pd.DataFrame): Input DataFrame.

        Returns:
            (X, y): Features and target.
        """
        y = df[self._target].copy()
        X = df.drop(columns=self._target).copy()

        return X, y


class TrainTestSplit:
    """
    Split data into train and test subsets.

    Supports pandas DataFrame, Series, or aligned tuples/lists of them.

    Args:
        train_size (float): Fraction of data to use for training.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, train_size: float = 0.8, random_state: int = 42):
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")

        self._train_size = train_size
        self._random_state = random_state

    def __call__(self, data: PandasObj | tuple[PandasObj, ...]) -> tuple[Any, Any]:
        """
        Split data into train and test subsets.

        Args:
            data: A pandas DataFrame, Series, or an aligned tuple/list of them.

        Returns:
            tuple: train-test split.
        """
        if isinstance(data, (pd.Series, pd.DataFrame)):
            train = data.sample(frac=self._train_size, random_state=self._random_state)
            test = data.drop(train.index)
            return train.reset_index(drop=True), test.reset_index(drop=True)

        if isinstance(data, (tuple, list)):
            first = data[0]

            train_idx = first.sample(
                frac=self._train_size, random_state=self._random_state
            ).index

            train = tuple(d.loc[train_idx].reset_index(drop=True).copy() for d in data)
            test = tuple(d.drop(train_idx).reset_index(drop=True).copy() for d in data)

            return train, test


class ToTensor:
    """
    Recursively convert pandas objects to torch tensors.

    Args:
        dtype (torch.dtype): Target tensor dtype. Defaults to torch.float32.
        device (torch.device or str): Target device. Defaults to 'cpu'.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self._dtype = dtype
        self._device = device

    def __call__(self, data: Any) -> Any:
        """
        Convert pandas objects in nested structure to tensors.

        Args:
            data (Any): pd.DataFrame, pd.Series or nested tuple/list of them.

        Returns:
            Any: Corresponding tensors.
        """
        return self._to_tensor(data)

    def _to_tensor(self, data: Any) -> Any:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return torch.as_tensor(
                data.to_numpy(),
                dtype=self._dtype,
                device=self._device,
            )

        if isinstance(data, (tuple, list)):
            return tuple(self._to_tensor(d) for d in data)

        return data


class Binarizer(ColwiseTransform):
    """
    Binarize numeric columns based on a threshold.

    Args:
        columns (str, list[str] or None): Column(s) to binarize or None for all. Defaults to None.
        threshold (float): Value above which entries become 1. Defaults to 0.0.
        dtype (type): Data type for the resulting binarized columns. Defaults to np.uint8.

    """

    def __init__(
        self,
        columns: list[str] | None = None,
        threshold: float = 0.0,
        dtype: type = np.uint8,
    ):
        super().__init__(columns)
        self._threshold = threshold
        self._dtype = dtype

    def compute(self, column: pd.Series) -> pd.Series:
        """
        Binarize single column based on threshold.

        Args:
            column (pd.Series): Input numeric column to binarize.

        Returns:
            pd.Series: Binarized column.
        """
        return (column > self._threshold).astype(self._dtype)


class LogTransformer(ColwiseTransform):
    """
    Apply log transformation to numeric columns.

    Args:
        columns (str, list[str] or None): Column(s) to log-transform or None for all. Defaults to None.
        offset (float): Value added to each entry before applying log. Defaults to 1.0.
    """

    def __init__(self, columns=None, offset=1.0):
        super().__init__(columns)
        self.offset = offset

    def compute(self, column: pd.Series) -> pd.Series:
        """
        Apply log transformation to a single column.

        Args:
            column (pd.Series): Input numeric column.

        Returns:
            pd.Series: Log-transformed column.
        """
        if (column + self.offset <= 0).any():
            raise ValueError("log(x + offset) invalid for some values")
        return np.log(column + self.offset)

    def compute_inverse(self, column: pd.Series) -> pd.Series:
        """
        Apply inverse log transformation to a single column.

        Args:
            column (pd.Series): Input column.

        Returns:
            pd.Series: Original-scale numeric column.
        """
        return np.exp(column) - self.offset


class FrequencyEncoder(StatefulColwiseTransform):
    """
    Encode categorical columns using frequency encoding.

    Args:
        columns (str, list[str] or None): Column(s) to frequency-encode or None for all. Defaults to None.
        normalize (bool): Whether to encode using relative frequencies instead
            of raw counts. Defaults to True.
        handle_unknown (str): Strategy for unseen categories during transform. Defaults to "error".
            Options:
                "error": raise an error
                "use_encoded_value": encode as unknown_value
                "ignore": leave unknown categories as NaN
        unknown_value (float): Encoding value for unseen categories when
            handle_unknown="use_encoded_value". Defaults to 0.0.
        retain_params (bool): Whether to store computed parameters. Defaults to True.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        normalize: bool = True,
        handle_unknown: str = "error",
        unknown_value: float = 0.0,
        retain_params: bool = True,
    ):
        super().__init__(
            columns,
            retain_params,
        )
        valid = {"error", "use_encoded_value", "ignore"}

        if handle_unknown not in valid:
            raise ValueError(f"handle_unkown must be one of {valid}")

        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self._normalize = normalize

    def compute_params(self, column: pd.Series) -> dict[str, dict]:
        """
        Compute the frequency mapping for a single column.

        Args:
            column (pd.Series): Column to analyze.

        Returns:
            dict[str, dict]: Dictionary containing the frequency map under "freq".
        """
        freq = (column.value_counts(normalize=self._normalize)).to_dict()

        return {"freq": freq}

    def compute(self, column: pd.Series) -> pd.Series:
        """
        Encode a single column using stored or computed frequency parameters.

        Args:
            column (pd.Series): Column to encode.

        Returns:
            pd.Series: Frequency-encoded column.
        """
        freq_map = self._get_or_compute_params(column)["freq"]

        encoded = column.map(freq_map)

        if self.handle_unknown == "error":
            mask_unknown = encoded.isna() & column.notna()
            if mask_unknown.any():
                unknowns = column[mask_unknown].unique()
                raise ValueError(
                    f"unknown category in column '{column.name}': {unknowns}"
                )
        elif self.handle_unknown == "use_encoded_value":
            encoded = encoded.fillna(self.unknown_value)

        return encoded


class LabelTransform(StatefulColwiseTransform):
    """
    Convert categorical columns into integer label encodings

    Args:
        columns (str, list[str]): Columns to encode.
        handle_unknown (str): Strategy for unseen categories during transform. Defaults to "error".
            Options:
                "error": raise a error
                "ignore": leave unseen categories as NaN
        retain_params (bool): Whether to store computed parameters. Defaults to True.
    """

    def __init__(
        self,
        columns: list[str],
        handle_unknown: str = "error",
        retain_params: bool = True,
    ):
        super().__init__(columns, retain_params)

        if handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unkown must be 'error' or 'ignore'")
        self.handle_unknown = handle_unknown

    def compute_params(self, column) -> dict[str, dict]:
        """
        Compute a mapping from category to integer labels.

        Args:
            column (pd.Series): Column to encode.

        Returns:
            dict[str, dict]: Dictionary containing the label map under "label_map".
        """

        categories = column.dropna().unique().tolist()

        labels = {cat: i for i, cat in enumerate(categories)}
        return {"label_map": labels}

    def compute(self, column: pd.Series) -> pd.Series:
        """
        Encode the column by using stored or computed label mappings.

        Args:
            column (pd.Series): Column to encode.

        Returns:
            pd.Series: Label-encoded column.
        """
        label_map = self._get_or_compute_params(column)["label_map"]
        encoded = column.map(label_map)

        if self.handle_unknown == "error":
            mask_unknown = encoded.isna() & column.notna()
            if mask_unknown.any():
                unknowns = column[mask_unknown].unique()
                raise ValueError(
                    f"unknown category in column '{column.name}': {unknowns}"
                )
        return encoded

    def compute_inverse(self, column: pd.Series) -> pd.Series:
        """
        Convert integer labels back to original categories for a single column.

        Args:
            column (pd.Series): Label-encoded column.

        Returns:
            pd.Series: Decoded column with original categories.
        """
        label_map = self._get_params(column)["label_map"]
        inv_label_map = {v: k for k, v in label_map.items()}

        return column.map(inv_label_map)


class MaxAbsScaler(StatefulColwiseTransform):
    """
    Scale columns by dividing by the column's maximum absolute value.
    This preserves sign and maps the data to [-1, 1].

    Args:
        columns (str, list[str] or None): Column(s) to scale or None for all. Defaults to None.
        retain_params (bool): Whether to store computed parameters. Defaults to True.
    """

    def compute_params(self, column: pd.Series) -> dict[str, float]:
        """
        Compute the maximum absolute value for a single column.

        Args:
            column (pd.Series): Column for which to compute maximum absolute value.

        Returns:
            dict[str, float]: Dictionary containing maximum absolute value for column under "max_abs".
        """
        max_abs = column.abs().max()

        if pd.isna(max_abs):
            raise ValueError(f"column '{column.name}' contains only NaNs")

        if max_abs == 0:
            max_abs = 1.0  # avoid division by zero

        return {"max_abs": float(max_abs)}

    def compute(self, column: pd.Series) -> pd.Series:
        """
        Scale column by maximum absolute value.

        Args:
            column (pd.Series): Column to scale.

        Returns:
            (pd.Series): Scaled column
        """
        params = self._get_or_compute_params(column)
        max_abs = params["max_abs"]

        return column / max_abs

    def compute_inverse(self, column: pd.Series) -> pd.Series:
        """
        Reverse MaxAbs scaling transformation for a single column.

        Args:
            column (pd.Series): Column to unscale.

        Returns:
            (pd.Series): Original-scale column.
        """
        max_abs = self._get_params(column)["max_abs"]
        return column * max_abs


class MinMaxScaler(StatefulColwiseTransform):
    """
    Scale columns to a specified range using Min-Max normalization.

    Args:
        columns (str, list[str] or None): Column(s) to min-max scale or None for all. Defaults to None.
        feature_range (tuple[float, float]): Desired output range (min, max).
        retain_params (bool): Whether to store computed parameters.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        feature_range: tuple[float, float] = (0.0, 1.0),
        retain_params: bool = True,
    ):
        super().__init__(columns, retain_params)
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "first element of feature_range must be less than the second"
            )
        self.feature_range = feature_range

    def compute_params(self, column: pd.Series) -> dict[str, float]:
        """
        Compute minimum and maximum values for the column.

        Args:
            column (pd.Series): Column for which to compute minimum and maximum.

        Returns:
            dict[str, float]: Dictionary containing "min" and "max" values.
        """
        col_min = column.min()
        col_max = column.max()

        if pd.isna(col_min) or pd.isna(col_max):
            raise ValueError(f"column '{column.name}' contains only NaNs")

        if col_max == col_min:
            col_max = col_min + 1.0

        return {"min": float(col_min), "max": float(col_max)}

    def compute(self, column: pd.Series) -> pd.Series:
        """
        Scale column to the specified feature range.

        Args:
            column (pd.Series): column to scale.

        Returns:
            (pd.Series): scaled column.
        """
        params = self._get_or_compute_params(column)
        col_min, col_max = params["min"], params["max"]
        min_range, max_range = self.feature_range

        scaled = (column - col_min) / (col_max - col_min)
        scaled = scaled * (max_range - min_range) + min_range
        return scaled

    def compute_inverse(self, column: pd.Series) -> pd.Series:
        """
        Reverse Min-Max scaling to original values.

        Args:
            data (pd.Series): Scaled column.

        Returns:
            pd.Series: Original-scale column.
        """

        params = self._get_params(column)
        col_min, col_max = params["min"], params["max"]
        min_range, max_range = self.feature_range

        return (column - min_range) / (max_range - min_range) * (
            col_max - col_min
        ) + col_min


class OrdinalEncoder(StatefulColwiseTransform):
    """
    Ordinally encode categorical columns by mapping categories to integers
    in the order they appear.

    Args:
        columns (str, list[str] or None): Column(s) to ordinally encode or None for all. Defaults to None.
        handle_unknown (str): Strategy for unseen categories during transform. Defaults to "error".
            Options:
                "error": raise an error
                "use_encoded_value": encode as unknown_value
                "ignore": leave unseen categories as NaN
        unknown_value (int): Value used for unseen categories when handle_unknown="use_encoded_value".
        Defaults to -1.
        retain_params (bool): Whether to store computed parameters. Defaults to True.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        handle_unknown: str = "error",
        unknown_value: int = -1,
        retain_params: bool = True,
    ):
        super().__init__(columns, retain_params)

        valid = {"error", "use_encoded_value", "ignore"}
        if handle_unknown not in valid:
            raise ValueError(f"handle_known must be one of {valid}")

        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def compute_params(self, column: pd.Series) -> dict[str, dict]:
        """
        Compute category-to-integer mapping for a single column.

        Args:
            column (pd.Series): Column for which integer mapping is computed.

        Returns:
            dict[str, dict]: A dictonary containing the ordinal mapping under "ord_map".
        """
        categories = pd.Index(column.dropna().unique())
        ord_map = {cat: i for i, cat in enumerate(categories)}
        return {"ord_map": ord_map}

    def compute(self, column: pd.Series) -> pd.Series:
        """
        Ordinally encode a column using stored or newly computed mapping.

        Args:
            column (pd.Series): Column to ordinally encode.

        Returns:
            pd.Series: Ordinally encoded column.
        """
        ord_map = self._get_or_compute_params(column)["ord_map"]
        encoded = column.map(ord_map)

        if self.handle_unknown == "error":
            mask_unknown = encoded.isna() & column.notna()
            if mask_unknown.any():
                unknowns = column[mask_unknown].unique()
                raise ValueError(
                    f"unknown category in column '{column.name}': {unknowns}"
                )
        elif self.handle_unknown == "use_encoded_value":
            encoded = encoded.fillna(self.unknown_value)

        return encoded


class SimpleImputer(StatefulColwiseTransform):
    """
    Impute missing values in each column using a chosen strategy.

    Supported strategies:
        "mean": replace NaN with column mean
        "median": replace NaN with column median
        "most_frequent": replace NaN with most frequent value
        "constant": replace NaN with a provided fill_value

    Args:
        columns (str, list[str]): Column(s) to impute.
        strategy (str): Imputation strategy to use. Defaults to "mean".
        fill_value (Any | None): Value used when strategy="constant". Defaults to None.
        retain_params (bool): Whether to store computed parameters. Defaults to True.
    """

    def __init__(
        self,
        columns: list[str],
        strategy: str = "mean",
        fill_value: Any | None = None,
        retain_params: bool = True,
    ):

        super().__init__(columns, retain_params)

        valid = {"mean", "median", "most_frequent", "constant"}
        if strategy not in valid:
            raise ValueError(f"invalid strategy: {strategy}. Must be one of {valid}")

        if strategy == "constant" and fill_value is None:
            raise ValueError("fill_value must be provided when strategy='constant'")
        self.strategy = strategy
        self.fill_value = fill_value

    def compute_params(self, column: pd.Series) -> dict[str, Any]:
        """
        Compute the imputation value for a single column.

        Args:
            column (pd.Series): Column for which the imputation value is computed.

        Returns:
            dict[str, Any]: A dictionary containing the imputation value under "value".
        """
        if column.isna().all():
            if self.strategy == "constant":
                return {"value": self.fill_value}
            raise ValueError(
                f"cannot compute {self.strategy} for entirely NaN column: {column}"
            )

        if self.strategy == "mean":
            value = column.mean()
        elif self.strategy == "median":
            value = column.median()
        elif self.strategy == "most_frequent":
            value = column.mode().iloc[0]
        elif self.strategy == "constant":
            value = self.fill_value
        else:
            raise RuntimeError(f"unhandled strategy: {self.strategy}")

        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass

        return {"value": value}

    def compute(self, column: pd.Series) -> pd.Series:
        """
        Replace missing values in the column using stored or computed imputation value.

        Args:
            column (pd.Series): Column to impute.

        Returns:
            pd.Series: Column with missing values filled.
        """
        value = self._get_or_compute_params(column)["value"]
        return column.fillna(value)


class StandardScaler(StatefulColwiseTransform):
    """
    Standardize numeric columns by subtracting the mean and scaling to unit variance.

    Args:
        columns (str, list[str] or None): Column(s) to scale or None for all. Defaults to None.
        retain_params (bool): Whether to store computed parameters per column. Defaults to True.
    """

    def compute_params(self, column: pd.Series) -> dict[str, float]:
        """
        Compute the mean and standard deviation for a single column.

        Args:
            column (pd.Series): Column for which mean and standard deviation are computed.

        Returns:
            dict[str, float]: A dictionary containing "mean" and "std" values.
        """
        mean = column.mean()
        std = column.std()

        # Avoid division by zero
        if std == 0 or pd.isna(std):
            std = 1.0

        return {"mean": float(mean), "std": float(std)}

    def compute(self, column) -> pd.Series:
        """
        Standardize a single column using stored or computed parameters.

        Args:
            column (pd.Series): Column to standardize.

        Returns:
            pd.Series: Standardized column.
        """
        params = self._get_or_compute_params(column)
        mean, std = params["mean"], params["std"]

        return (column - mean) / std

    def compute_inverse(self, column: pd.Series) -> pd.Series:
        """
        Reverse standardization for a single column.

        Args:
            column (pd.Series): Standardized column.

        Returns:
            pd.Series: Column restored to original scale.
        """
        params = self._get_params(column)
        mean, std = params["mean"], params["std"]

        return column * std + mean
