from typing import Any, Dict, List, Optional

import pandas as pd

from mltools.data.transform import BaseTransformer


class MovingAverage(BaseTransformer):
    """
    Utilizes the pandas rolling method to calculate the moving average of the
    specified columns, for the specified windows. Note that the data is expected
    to be sorted when using this transformer.

    Parameters
    ----------
    cols: List[str]
        The columns to calculate the moving average for
    windows: List[int]
        The window sizes to calculate the moving average for
    rolling_kwargs: Optional[Dict[str, Any]]
        Any additional arguments to pass to the rolling method
    """

    def __init__(
            self,
            cols: list[str],
            windows: list[int],
            rolling_kwargs: dict[str, Any] | None = None,
        ):
        self.cols = cols
        self.windows = windows
        self.rolling_kwargs = rolling_kwargs or {}

    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {self._column_fmt(col, window): df[col].rolling(window, **self.rolling_kwargs).mean()
            for col in self.cols
            for window in self.windows
        }
        return df.assign(**new_cols)

    def _column_fmt(self, col: str, window: int) -> str:
        return f"{col}_ma{window}"


class Expanding(BaseTransformer):
    """
    Utilizes the pandas expanding method to calculate the expanding average of the
    specified columns. Note that the data is expected to be sorted when using this
    transformer.

    Parameters
    ----------
    cols: List[str]
        The columns to calculate the expanding window on.
    min_periods: int
        The minimum number of periods to include in the expanding window.
    agg_fn: str
        The aggregation function to use for the expanding window.
    """

    def __init__(self, cols: list[str], min_periods: int = 1, agg_fn: str = "mean"):
        self.cols = cols
        self.min_periods = min_periods
        self.agg_fn = agg_fn

    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_cols = {self._column_fmt(col): df[col].expanding(min_periods=self.min_periods).agg(self.agg_fn)
            for col in self.cols
        }
        return df.assign(**new_cols)

    def _column_fmt(self, col: str) -> str:
        return f"{col}_exp{self.agg_fn}"
