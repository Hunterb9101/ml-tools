from typing import Literal, Sequence
import logging

import numpy as np
import pandas as pd
import scipy.stats as ss

import mltools.pandas as mtp


def si(old: Sequence, new: Sequence, bins: int=10, is_categorical: bool = False) -> float:
    """
    Calculate a stability index between two series. Input target scores for a
    Population Stability Index (PSI), or continuous feature samples for a Characteristic
    Stability Index (CSI)

    Parameters
    ----------
    old: pd.DataFrame
        A dataframe that a model was trained on
    new: pd.DataFrame
        A dataframe that has been seen recently
    groups: int
        The number of quantiles to use
    is_categorical: bool
        Treat `old` and `new` as categorical variables. No quantiles will be calculated and bins
        will be unique category values.
    Returns
    -------
    float
        The resultant stability index value across all quantiles
    """
    return _si_df(old, new, bins, is_categorical=is_categorical)["si"].sum()


def _si_df(old: Sequence, new: Sequence, bins: int=10, is_categorical: bool = False) -> pd.DataFrame:
    """
    Calculate a stability index between two series. Input target scores for a
    Population Stability Index (PSI), or continuous feature samples for a Characteristic
    Stability Index (CSI)

    Parameters:
    -----------
    old: Sequence
        A dataframe that a model was trained on
    new: Sequence
        A dataframe that has been seen recently
    groups: int
        The number of quantiles to use. This is ignored when `is_categorical` is true.
    is_categorical: bool
        Treat `old` and `new` as categorical variables. No quantiles will be calculated and bins
        will be unique category values.

    Returns:
    --------
    pd.DataFrame
        A dataframe with all of the key components necessary for
        computing a stability index.
    """
    if is_categorical:
        df = _counts_by_category(old=old, new=new)
    else:
        df = _counts_by_quantile(old=old, new=new, bins=bins)
    df["train_pct"] = df["old"] / len(old)
    df["score_pct"] = df["new"] / len(new)
    df["ln_score_train"] = np.log(df["score_pct"] / df["train_pct"])
    df["si"] = (df["score_pct"] - df["train_pct"]) * df["ln_score_train"]
    df.loc[df["si"] == np.inf, "si"] = 0
    return df


def _counts_by_quantile(old: Sequence, new: Sequence, bins: int = 10, clip_bounds: bool = True, tol=1e-3) -> pd.DataFrame:
    """
    Parameters:
    -----------
    old: Sequence
        A dataframe that a model was trained on
    new: Sequence
        A dataframe that has been seen recently
    bins: int
        The number of quantiles to calculate
    clip_bounds: bool
        Manage how outliers are dealt with. When set to false, values greater than the max or less than the min in the
        old sequence will not be included in the final count result. When set to true, values will be clipped into the
        largest bucket.

    Returns
    -------
    pd.DataFrame
        A dataframe with 3 columns: "bin", "old", and "new"
    """
    old = np.array(old)
    new = np.array(new)
    df = pd.DataFrame(old, columns=["old"])
    df["quant"] = pd.qcut(old, bins, duplicates="drop")
    df = df.groupby("quant").count().reset_index()

    if clip_bounds:
        new = np.clip(new, a_min=old.min() + tol, a_max=old.max() - tol)
    new_dist = pd.DataFrame(mtp.map_to_series(pd.Series(new), df["quant"].cat.categories), columns=["quant"])
    new_dist["new"] = 0
    df["new"] = new_dist.groupby("quant").count().reset_index()["new"]
    # New data might not exist in the old quantiles
    df["new"].fillna(1, inplace=True)
    return df


def _counts_by_category(old: Sequence, new: Sequence) -> pd.DataFrame:
    """
    Parameters:
    -----------
    old: Sequence
        A dataframe that a model was trained on
    new: Sequence
        A dataframe that has been seen recently

    Returns
    -------
    pd.DataFrame
        A dataframe with 3 columns: "bin", "old", and "new"
    """
    old_counts = pd.DataFrame(old.value_counts(dropna=False), columns=["old"])
    new_counts = pd.DataFrame(new.value_counts(dropna=False), columns=["new"])
    categories = list(set(old_counts.index.tolist() + new_counts.index.tolist()))
    df = pd.DataFrame(categories, columns=["bin"])
    df = df.merge(old_counts, how="left", left_on="bin", right_on=old_counts.index)
    df = df.merge(new_counts, how="left", left_on="bin", right_on=new_counts.index)
    df = df.fillna(0).astype({"old": 'int64', "new": 'int64'}, copy=False)

    if len(df) > 50:
        logging.warning("Found over 50 unique categories between series %s and %s.", old.name, new.name)
    return df


def si_is_signifcant(
    old: Sequence,
    new: Sequence,
    bins: int = 10,
    is_categorical: bool = False,
    method: Literal["chisq", "norm", "industry"] = "norm",
    quantile: float = 0.95
) -> bool:
    observed = si(old=old, new=new, bins=bins, is_categorical=is_categorical)
    if method == "chisq":
        return observed > critical_value_chi2(len_old=len(old), len_new=len(new), n_bins=bins, quantile=quantile)
    elif method == "norm":
        return observed > critical_value_norm(len_old=len(old), len_new=len(new), n_bins=bins, quantile=quantile)
    elif method == "industry":
        return observed > 0.2
    else:
        raise ValueError(f"Unexpected method: {method}")



def critical_value_norm(len_new: int, len_old: int, n_bins: int, quantile: float = 0.95) -> float:
    """
    The recommended method from https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations
    to determine statistically significant cutoffs for PSI distribution differences.

    This method agrees closely with the Chi-Squared test values.
    """
    z = ss.norm.ppf(quantile)
    return (1/len_new + 1/len_old) * (n_bins - 1) + z * (1/len_new + 1/len_old) * np.sqrt(2* (n_bins - 1))


def critical_value_chi2(len_new: int, len_old: int, n_bins: int, quantile: float = 0.95) -> float:
    """
    The second method described in https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations
    to determine statistically significant cutoffs for PSI distribution differences.

    This method more closely follows the underlying distribution of PSI values, at the expense of being harder to
    visualize quantiles. This method is said to agree closely with critical values obtained from the Normal 
    distribution.
    """
    z = ss.chi2.ppf(q=quantile, df=n_bins - 1)
    return z * (1/len_new + 1/len_old)
