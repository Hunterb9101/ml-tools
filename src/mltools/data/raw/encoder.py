from abc import ABC, abstractmethod
from typing import List, Dict

import pandas as pd

class UnivariateEncoder(ABC):
    def __init__(self, columns: List[str], target_col: str):
        self.is_fit = False
        self.columns = columns
        self.target_col = target_col

    def fit(self, df: pd.DataFrame):
        for c in self.columns:
            self.fit_column(df[c], df[self.target_col])

    @abstractmethod
    def fit_column(self, col: pd.Series, target: pd.Series) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = df.copy()
        for c in self.columns:
            dfc[c] = self.transform_column(df[c])
        return dfc

    @abstractmethod
    def transform_column(self, col: pd.Series) -> pd.Series:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class TargetEncoder(UnivariateEncoder):
    def __init__(self, columns: List[str], target_col: str):
        super().__init__(columns, target_col=target_col)
        self.mapping: Dict[str, Dict[str, float]] = {}

    def fit_column(self, col: pd.Series, target: pd.Series) -> None:
        df = pd.DataFrame({"col": col, "target": target})
        self.mapping[col.name] = df.groupby("col").mean().to_dict()["target"]

    def transform_column(self, col: pd.Series) -> pd.Series:
        return col.map(self.mapping[col.name])
