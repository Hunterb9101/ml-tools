"""Base classes for dataframe transformations."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseTransformer(ABC):
    """Define the transformer interface."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Fit the transformer to a dataframe."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the transformer and return transformed data."""
        self.fit(df)
        return self.transform(df)
