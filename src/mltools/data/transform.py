from abc import ABC, abstractmethod

import pandas as pd

class BaseTransformer(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
