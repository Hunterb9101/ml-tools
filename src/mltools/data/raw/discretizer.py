from typing import List, Dict

import pandas as pd
import numpy as np

class Discretizer:
    """
    The Discretizer class is used to discretize continuous features into buckets
    """

    def __init__(self, cols: List[str], max_unique_vals: int = 10):
        """
        Initialize the discretizer.

        Parameters
        ----------
        cols : List[str]
            The columns to discretize
        max_unique_vals : int
            The maximum number of unique values that any one column can have before it is binned down to
            `max_unique_vals` unique values
        """
        self.cols = cols
        self.bucket_cols = None
        self.bucket_map: Dict[str, np.ndarray] = {}
        self.max_unique_vals = max_unique_vals
        self.feature_name: str = "q_{x}"

    def fit(self, df: pd.DataFrame):
        """
        Create quantized buckets for grouping

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            A DataFrame with quantized buckets on interaction columns
        """
        out = pd.DataFrame(index=df.index)

        self.bucket_cols = [col for col in self.cols if df[col].nunique() > self.max_unique_vals]
        for b in self.bucket_cols:
            bins = min(self.max_unique_vals, df[b].nunique())
            _, self.bucket_map[b] = pd.qcut(df[b], bins, retbins=True, labels=False, duplicates='drop')
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.bucket_cols:
            df[self.feature_name.format(x=c)] = self.transform_column(df[c])
        return df

    def transform_column(self, ser: pd.Series) -> pd.Series:
        """
        Convert a single column into a quantized bucket using self.bucket_map
        """
        return pd.cut(ser, self.bucket_map[ser.name], labels=False, include_lowest=True)

    def fit_transform(self, df:pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
