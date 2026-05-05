"""Model-based missing value imputation."""

import logging

import pandas as pd
import sklearn.metrics as sm

from mltools.data.transformers.base import BaseTransformer

logger = logging.getLogger(__name__)


class ModelImputer(BaseTransformer):
    """Impute missing target values with a fitted model."""

    def __init__(
        self,
        model,
        target_col: str,
        missing_value: float,
    ):
        self.model = model
        self.target_col = target_col
        self.missing_value = missing_value
        self.fit_cols: list[str] | None = None

    def fit(self, df: pd.DataFrame):
        """Fit the imputation model."""
        self.fit_cols = [x for x in df.columns if x != self.target_col]
        train, test = self._get_train_test(df.copy())
        self.model.fit(train[self.fit_cols], train[self.target_col])

        logger.debug("Imputation model performance:")
        logger.debug("MAE: %s", sm.mean_absolute_error(test[self.target_col], self.model.predict(test[self.fit_cols])))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in a dataframe."""
        if self.fit_cols is None:
            msg = "Must fit before transforming"
            raise RuntimeError(msg)
        for c in self.fit_cols:
            if c not in df.columns:
                msg = f"{c} not in df.columns"
                raise ValueError(msg)
        train, test = self._get_train_test(df)

        mask = df[self.target_col] == self.missing_value
        logger.info("Imputing %s values", mask.sum())
        test.loc[mask, self.target_col] = self.model.predict(test.loc[mask, self.fit_cols])
        return pd.concat([train, test], axis=0).sort_index()

    def _get_train_test(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        df["_tmp"] = df[self.target_col] == self.missing_value
        train, test = df.loc[~df["_tmp"], :].copy(), df.loc[df["_tmp"], :].copy()
        train = train.drop(columns=["_tmp"])
        test = test.drop(columns=["_tmp"])
        return train, test
