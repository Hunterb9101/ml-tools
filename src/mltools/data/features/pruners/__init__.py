from abc import ABC, abstractmethod
from typing import List, Dict

import pandas as pd

from mltools.data.transform import BaseTransformer


class BasePruner(BaseTransformer, ABC):
    """
    An abstract class for feature selection. The object contains information on
    the columns to drop.
    """
    def __init__(self):
        self._drop_cols = []
        self._is_fit = False

    @property
    def drop_cols(self):
        if not self._is_fit:
            raise RuntimeError("Must fit before accessing drop_cols")
        return self._drop_cols

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.drop_cols)

    def __repr__(self) -> str:
        return self.__class__.__name__


class FeatureSelectionPipeline(BasePruner):
    """
    A class for managing multiple feature selection pruners.
    """
    def __init__(self, pruners: List[BasePruner]):
        super().__init__()
        self.pruners = pruners
        self.waterfall_: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> None:
        dfc = df.copy()
        for pruner in self.pruners:
            dfc = pruner.fit_transform(dfc)
            self._drop_cols.extend(pruner.drop_cols)
            self.waterfall_[str(pruner)] = pruner.drop_cols
        self._is_fit = True
