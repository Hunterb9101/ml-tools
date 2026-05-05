"""Deterministic fold and holdout assignment helpers."""

import numpy as np
import pandas as pd
import sklearn.model_selection as sms

MIN_ROWS_PER_SPLIT = 2


def assign_holdout(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    test_size: float,
    random_state: int,
    holdout_col: str = "is_holdout",
) -> pd.DataFrame:
    """Assign a deterministic holdout indicator to a dataframe copy.

    Parameters
    ----------
    df
        Source dataframe.
    target_col
        Target column used for stratification. If ``None``, assignment is
        unstratified.
    test_size
        Fraction of rows assigned to holdout.
    random_state
        Random seed used by sklearn's splitter.
    holdout_col
        Output boolean column name.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with a holdout indicator column.
    """
    _validate_output_column(df, holdout_col)
    _validate_fraction(test_size, name="test_size")

    if len(df) < MIN_ROWS_PER_SPLIT:
        msg = "Holdout assignment requires at least 2 rows."
        raise ValueError(msg)

    stratify = _stratify_values(df, target_col, split_name="holdout")
    row_positions = np.arange(len(df))
    _, holdout_positions = sms.train_test_split(
        row_positions,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    assigned = df.copy()
    is_holdout = pd.Series(data=np.zeros(len(assigned), dtype=bool), index=assigned.index)
    is_holdout.iloc[holdout_positions] = True
    assigned[holdout_col] = is_holdout
    return assigned


def assign_folds(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    n_splits: int,
    random_state: int,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Assign deterministic validation fold ids to a dataframe copy.

    Parameters
    ----------
    df
        Source dataframe.
    target_col
        Target column used for stratification. If ``None``, assignment is
        unstratified.
    n_splits
        Number of validation folds.
    random_state
        Random seed used by sklearn's splitter.
    fold_col
        Output integer fold column name.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with a fold id column.
    """
    _validate_output_column(df, fold_col)
    _validate_n_splits(df, n_splits)

    folds = pd.Series(-1, index=df.index, dtype="int64")
    row_positions = np.arange(len(df))

    if target_col is None:
        splitter = sms.KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(row_positions)
    else:
        y = _stratify_values(df, target_col, split_name="fold")
        _validate_stratified_fold_counts(y, n_splits)
        splitter = sms.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(row_positions, y)

    for fold_id, (_, val_positions) in enumerate(split_iterator):
        folds.iloc[val_positions] = fold_id

    assigned = df.copy()
    assigned[fold_col] = folds
    return assigned


def _validate_output_column(df: pd.DataFrame, output_col: str) -> None:
    """Raise when an assignment output column would overwrite input data."""
    if output_col in df.columns:
        msg = f"Output column already exists: {output_col}"
        raise ValueError(msg)


def _validate_fraction(value: float, *, name: str) -> None:
    """Raise when a split fraction is outside the supported range."""
    if not 0 < value < 1:
        msg = f"{name} must be between 0 and 1, exclusive."
        raise ValueError(msg)


def _validate_n_splits(df: pd.DataFrame, n_splits: int) -> None:
    """Raise when the requested fold count is impossible."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        msg = "n_splits must be an integer."
        raise TypeError(msg)
    if n_splits < MIN_ROWS_PER_SPLIT:
        msg = "n_splits must be at least 2."
        raise ValueError(msg)
    if n_splits > len(df):
        msg = "n_splits cannot exceed the number of rows."
        raise ValueError(msg)


def _stratify_values(df: pd.DataFrame, target_col: str | None, *, split_name: str) -> pd.Series | None:
    """Return target values for stratification after validation."""
    if target_col is None:
        return None
    if target_col not in df.columns:
        msg = f"target_col is missing from dataframe: {target_col}"
        raise ValueError(msg)

    y = df[target_col]
    class_counts = y.value_counts(dropna=False)
    if class_counts.empty or class_counts.min() < MIN_ROWS_PER_SPLIT:
        msg = f"Stratified {split_name} assignment requires at least 2 rows per target class."
        raise ValueError(msg)
    return y


def _validate_stratified_fold_counts(y: pd.Series, n_splits: int) -> None:
    """Raise when a stratified fold assignment has too few rows per class."""
    class_counts = y.value_counts(dropna=False)
    if class_counts.min() < n_splits:
        msg = "n_splits cannot exceed the count of the least frequent target class."
        raise ValueError(msg)
