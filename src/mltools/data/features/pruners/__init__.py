"""Feature pruning transformers."""

from abc import ABC, abstractmethod

import pandas as pd

from mltools.data.transform import BaseTransformer


class BasePruner(BaseTransformer, ABC):
    """Provide a base class for feature selection.

    The object contains information on
    the columns to drop.
    """

    def __init__(self):
        self._drop_cols = []
        self._is_fit = False

    @property
    def drop_cols(self):
        """Return columns selected for removal."""
        if not self._is_fit:
            msg = "Must fit before accessing drop_cols"
            raise RuntimeError(msg)
        return self._drop_cols

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Fit the pruner."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop selected columns from the dataframe."""
        return df.drop(columns=self.drop_cols)

    def __repr__(self) -> str:
        """Return the pruner name."""
        return self.__class__.__name__


class FeatureSelectionPipeline(BasePruner):
    """A class for managing multiple feature selection pruners."""

    def __init__(self, pruners: list[BasePruner]):
        super().__init__()
        self.pruners = pruners
        self.waterfall_: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """Fit each pruner in sequence."""
        dfc = df.copy()
        for pruner in self.pruners:
            dfc = pruner.fit_transform(dfc)
            self._drop_cols.extend(pruner.drop_cols)
            self.waterfall_[str(pruner)] = pruner.drop_cols
        self._is_fit = True
