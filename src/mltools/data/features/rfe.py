from typing import Any, Optional, Dict

import pandas as pd
from sklearn.feature_selection import RFE

from mltools.data.features import BasePruner


class RFEPruner(BasePruner):
    def __init__(self, model: Any, n_features_to_select: int, target_col: str):
        super().__init__()
        self.model = model
        self.n_features_to_select = n_features_to_select
        self.target_col = target_col
        self.ranking_: Optional[Dict[str, int]] = None
        self.selector: Optional[RFE] = None
        self._drop_cols = []
        self._is_fit = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the feature selector to the data. This should be done on the training data only.
        """
        assert self.target_col in df.columns, f"Target column {self.target_col} not in df columns"
        self.selector = RFE(self.model, n_features_to_select=1)
        self.selector.fit(df.drop(columns=self.target_col), df[self.target_col])
        self.ranking_ = dict(zip(df.columns, self.selector.ranking_))
        self._drop_cols = [
            col for rank, col in zip(self.selector.ranking_, df.columns) if rank > self.n_features_to_select
        ]
        self._is_fit = True
