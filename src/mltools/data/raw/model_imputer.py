from typing import List, Optional, Tuple
import logging

import pandas as pd
import sklearn.metrics as sm

from mltools.data.transform import BaseTransformer

logger = logging.getLogger(__name__)

class ModelImputer(BaseTransformer):
    def __init__(self,
        model,
        target_col: str,
        missing_value: float
    ):
        self.model = model
        self.target_col = target_col
        self.missing_value = missing_value
        self.fit_cols: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame):
        self.fit_cols = [x for x in df.columns if x != self.target_col]
        train, test = self._get_train_test(df.copy())
        self.model.fit(train[self.fit_cols], train[self.target_col])

        logger.debug("Imputation model performance:")
        logger.debug("MAE: %s", sm.mean_absolute_error(test[self.target_col], self.model.predict(test[self.fit_cols])))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fit_cols is None:
            raise RuntimeError("Must fit before transforming")
        for c in self.fit_cols:
            if c not in df.columns:
                raise ValueError(f"{c} not in df.columns")
        train, test = self._get_train_test(df)

        mask = df[self.target_col] == self.missing_value
        logger.info("Imputing %s values", mask.sum())
        test.loc[mask, self.target_col] = self.model.predict(test.loc[mask, self.fit_cols])
        return pd.concat([train, test], axis=0).sort_index()

    def _get_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        df["_tmp"] = df[self.target_col] == self.missing_value
        train, test = df.loc[~df["_tmp"], :].copy(), df.loc[df["_tmp"], :].copy()
        train.drop(columns=["_tmp"], inplace=True)
        test.drop(columns=["_tmp"], inplace=True)
        return train, test
