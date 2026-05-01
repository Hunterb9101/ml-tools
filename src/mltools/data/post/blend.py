from typing import List

import pandas as pd

from mltools.data.transform import BaseTransformer


class Blender(BaseTransformer):
    def __init__(self, weights: list[float], cols: list[float], out_col: str = "blend", normalize: bool = True):
        self._validate_weights(weights)
        sum_weights = sum(weights)
        self.weights = [w / sum_weights for w in weights] if normalize else weights
        self.cols = cols
        self.out_col = out_col

    def _validate_weights(self, weights: list[float]) -> None:
        for w in weights:
            if w < 0:
                msg = "Weights must be positive"
                raise ValueError(msg)
        sum_weights = sum(weights)
        if sum_weights == 0:
            msg = "Sum of weights must be positive"
            raise ValueError(msg)

    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = df.copy()
        dfc[self.out_col] = 0

        for c in self.cols:
            if c not in df.columns:
                msg = f"Column {c} not in df"
                raise ValueError(msg)
            dfc[self.out_col] += dfc[c] * self.weights[self.cols.index(c)]
        return dfc
