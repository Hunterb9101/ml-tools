from typing import List, Dict
import logging

import pandas as pd
import numpy as np

from mltools.data.raw.discretizer import Discretizer

logger = logging.getLogger(__name__)

class Interaction:
    """
    The Interaction class is used to augment a dataset with interaction features in the form
    new_X = X - E[X|Y]
    """
    def __init__(self, interaction_columns: List[str], max_unique_vals: int = 10):
        """
        Initialize the interaction augmentor.

        Parameters
        ----------
        interaction_columns : List[str]
            The columns to include in the interaction augmentor
        max_unique_vals : int
            The maximum number of unique values that any one column can have before it is binned down to
            `max_unique_vals` unique values
        """
        self.cols = interaction_columns
        self.max_unique_vals = max_unique_vals
        self.means: Dict[str, Dict[str, float]] = {}
        self.feature_name: str = "{x}_{y}"
        self.discretizer = Discretizer(self.cols, max_unique_vals=self.max_unique_vals)

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the interaction augmentor to the data. This should be done on the training data only.
        """
        df = df.copy()
        df = self.discretizer.fit_transform(df)
        # Compute a list of tuples of the product of all interaction columns
        pairs = [(col1, col2) for col1 in self.cols for col2 in self.cols if col1 != col2]

        for x, y in pairs:
            q_col = self.discretizer.feature_name.format(x=y)
            col = q_col if y in self.discretizer.bucket_map else y
            if x not in self.means:
                self.means[x] = {}
            try:
                self.means[x][y] = df.groupby(col)[x].mean().to_dict()
            except TypeError:
                logger.debug("Unable to create {x}|{y} interaction feature".format(x=x, y=y))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        old_cols = len(df.columns)
        df = df.copy()
        orig_cols = df.columns
        df = self.discretizer.transform(df)
        rm_cols = list(set(df.columns) - set(orig_cols))

        colnames =  []
        series = []

        for c in self.cols:
            for k, v in self.means[c].items():
                colnames.append(self.feature_name.format(x=c, y=k))
                series.append(df[k].map(v))

        aug =  pd.concat(series, axis=1)
        aug.columns = colnames
        df = pd.concat([df, aug], axis=1)
        logger.debug("Added {n} interaction features".format(n=len(df.columns) - old_cols))
        return df.drop(columns=rm_cols)
