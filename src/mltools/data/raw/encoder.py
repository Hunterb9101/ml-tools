"""Categorical encoders."""

from abc import ABC, abstractmethod

import pandas as pd


class UnivariateEncoder(ABC):
    """Encode columns independently using a target column."""

    def __init__(self, columns: list[str], target_col: str):
        self.is_fit = False
        self.columns = columns
        self.target_col = target_col

    def fit(self, df: pd.DataFrame):
        """Fit all configured columns."""
        for c in self.columns:
            self.fit_column(df[c], df[self.target_col])

    @abstractmethod
    def fit_column(self, col: pd.Series, target: pd.Series) -> None:
        """Fit one column."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform all configured columns."""
        dfc = df.copy()
        for c in self.columns:
            dfc[c] = self.transform_column(df[c])
        return dfc

    @abstractmethod
    def transform_column(self, col: pd.Series) -> pd.Series:
        """Transform one column."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the encoder and return transformed data."""
        self.fit(df)
        return self.transform(df)


class TargetEncoder(UnivariateEncoder):
    """Encode categories by target mean."""

    def __init__(self, columns: list[str], target_col: str):
        super().__init__(columns, target_col=target_col)
        self.mapping: dict[str, dict[str, float]] = {}

    def fit_column(self, col: pd.Series, target: pd.Series) -> None:
        """Fit a target mean mapping for one column."""
        df = pd.DataFrame({"col": col, "target": target})
        self.mapping[col.name] = df.groupby("col").mean().to_dict()["target"]

    def transform_column(self, col: pd.Series) -> pd.Series:
        """Map one column using the fitted target mean mapping."""
        return col.map(self.mapping[col.name])
