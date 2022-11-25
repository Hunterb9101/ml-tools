from typing import List

import numpy as np
import pandas as pd

def map_to_series(s: pd.Series, categories: List[pd.Interval]) -> pd.Series:
    """
    Maps a continuous variable to a corresponding interval range

    Parameters
    ----------
    s: pd.Series
        A series to map
    categories: List[pd.Interval]
        A list of mapping categories

    Returns
    -------
    pd.Series
        A mapped series
    """
    q = pd.Series(np.zeros(len(s)), dtype="object")
    used = pd.Series(np.zeros(len(s)))
    for c in categories:
        cond_a = s > c.left
        if c.closed_left:
            cond_a = s >= c.left

        cond_b = s < c.right
        if c.closed_right:
            cond_b = s <= c.right
        q.loc[cond_a & cond_b] = c
        used.loc[cond_a & cond_b] = 1
    q.loc[used == 0] = None
    return q
