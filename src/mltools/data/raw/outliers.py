from math import sqrt
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2

def mad(x: pd.DataFrame, normalize: bool = False):
    x_arr = np.array(x)
    median = np.quantile(x_arr, 0.5, axis=0)
    absolute_deviations = np.abs(x_arr - median)
    mad_val = np.quantile(absolute_deviations, 0.5, axis=0)
    if normalize:
        # A normalizing constant
        mad_val = mad_val / 0.67476
    return mad_val

class MADMedianOutlierDetector():
    def __init__(self, q: float = 0.975):
        """
        Parameters
        ----------
        q: float
            The quantile to use for the chi-squared distribution used in the rule.
            The default is 0.975, as found in literature.
        """
        self.q = q
        self._obs_median: Optional[float] = None
        self._obs_mad: Optional[float] = None

    def _get_rule_cutoff(self) -> float:
        return sqrt(chi2.ppf(self.q, 1))

    def rule_mask(self, x: pd.DataFrame) -> pd.Series:
        rule = abs(x - self._obs_median) / self._obs_mad
        return (rule < self._get_rule_cutoff()).any(axis=1)

    def fit(self, x: pd.DataFrame) -> None:
        self._obs_median = np.quantile(x, 0.5, axis=0)
        self._obs_mad = mad(x, True)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x[self.rule_mask(x)]

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        self.fit(x)
        out = self.transform(x)
        return out
