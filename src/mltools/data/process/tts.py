"""
Train Test Split Algorithms

There are some cases where we will need something a little more sophisticated
than what Scikit-learn will give us. This is where we will implement our own
train/validation/test split algorithms.
"""
import logging
from typing import Optional, Dict, Any, Sequence

import numpy as np
import pandas as pd
import sklearn.model_selection as sms

from mltools.types import TrainValTest

logger = logging.getLogger(__name__)


def compute_tts_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    tts_kwargs: Optional[Dict[str, Any]] = None
) -> TrainValTest[pd.DataFrame]:
    """
    Compute a train/val/test split of a DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
    val_size: float
    test_size: float
    tts_kwargs: Optional[Dict[str, Any]]
        Keyword arguments to pass directly to `sklearn.model_selection.train_test_split`.
    
    Returns
    -------
    TrainValTest[pd.DataFrame]
        A TrainValTest object containing the train, validation, and test dataframes.
    """
    is_frac = val_size < 1 or test_size < 1
    if val_size <= 0 or test_size <= 0:
        raise ValueError("val_size and test_size must be positive and non-zero.")
    if val_size + test_size >= 1 and is_frac:
        # If the values look like they should be fractions, but sum to something
        # greater than 1, then we should raise an error..
        raise ValueError("val_size and test_size must sum to less than 1 if using fractions.")
    if val_size + test_size >= len(df):
        raise ValueError("val_size and test_size must sum to less than the number of rows in the dataframe.")

    tts_kwargs = tts_kwargs or {}

    if not is_frac:
        # Normalize the values to be fractions.
        val_size = int(val_size) / len(df)
        test_size = int(test_size) / len(df)
    tvl, test = sms.train_test_split(df, test_size=test_size, **tts_kwargs)
    train, val = sms.train_test_split(tvl, test_size=val_size / (1 - test_size), **tts_kwargs)
    return TrainValTest(train=train, val=val, test=test)


def compute_oos_tts_split(
    df: pd.DataFrame,
    split_col: str,
    values_val: Sequence[Any],
    values_test: Sequence[Any],
    val_to_test_ratio: float = 0.5,
    tts_kwargs: Optional[Dict[str, Any]] = None,
    shuffle_seed: int = 42
) -> TrainValTest[pd.DataFrame]:
    """
    Create validation and test sets from predetermined values from the dataframe.
    This is intended to be used for creating out-of-time/out-of-sample datasets.

    Parameters
    ----------
    df: pd.DataFrame
    split_col: The column that will be used to split the data.
    values_val: Sequence[Any]
        The values to put in the validation set.
    values_test: Sequence[Any]
        The values to put in the test set.
    val_to_test_ratio: float
        When the validation and test sets have shared values, this is the ratio of
        the shared values that will go to the validation set. The rest will go to the
        test set.
    tts_kwargs: Optional[Dict[str, Any]]
        Keyword arguments to pass directly to `sklearn.model_selection.train_test_split`.
        Note that `stratify` will have little effect here.
    shuffle_seed: int
        The seed to use for shuffling the data.
    
    Returns
    -------
    TrainValTest[pd.DataFrame]
        A TrainValTest object containing the train, validation, and test dataframes.
    """
    if not 0 < val_to_test_ratio < 1:
        raise ValueError("val_to_test_ratio must be between 0 and 1, exclusive.")

    tts_kwargs = tts_kwargs or {}
    values_valtest = set(values_val) | set(values_test)
    shared_values = set(values_val) & set(values_test)

    # Make sure values are getting used.
    unq_vals = df[split_col].unique()
    unused_vals = values_valtest - set(unq_vals)
    if len(unused_vals) > 0:
        logger.warning("The following split values were not found in the dataframe: %s", unused_vals)

    # Split the data where the values are clear-cut into each category.
    train = df.loc[~df[split_col].isin(list(values_valtest)), :]
    val = df.loc[df[split_col].isin(list(values_valtest - set(values_test))), :]
    test = df.loc[df[split_col].isin(list(values_valtest - set(values_val))), :]

    # Add in Val/Test shared values.
    if len(shared_values) > 0:
        valtest = df.loc[df[split_col].isin(list(shared_values)), :]
        valtest_val, valtest_test = sms.train_test_split(valtest, train_size=val_to_test_ratio, **tts_kwargs)

        val = pd.concat([val, valtest_val]).sample(frac=1, random_state=shuffle_seed)
        test = pd.concat([test, valtest_test]).sample(frac=1, random_state=shuffle_seed)

    tvt = TrainValTest(train=train, val=val, test=test)
    _validate_split(tvt=tvt)
    return tvt


def _validate_split(tvt: TrainValTest[pd.DataFrame]) -> None:
    """
    Make sure that no splits are empty.
    """
    for f in tvt.model_fields:
        if len(getattr(tvt, f)) == 0:
            raise ValueError(f"The {f} set is empty.")
