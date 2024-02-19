from typing import List

import pandas as pd
import numpy as np

import mltools.data.validation.schema as mts


_FLOAT_DTYPES = [f"float{x}" for x in [8, 16, 32, 64]]
_INT_DTYPES = [f"int{x}" for x in [8, 16, 32, 64]]


def _validate_dtype(col: pd.Series, s: mts.SchemaObj) -> bool:
    if col.dtype != s.dtype and s.dtype not in _FLOAT_DTYPES:
        return False
    if col.dtype != s.dtype and s.dtype in _FLOAT_DTYPES and col.dtype not in _INT_DTYPES:
        if col.dtype == 'object':
            try:
                col.astype("float64")
            except ValueError:
                return False
        else:
            return False
    return True


def _illegal_values_idx(col: pd.Series, s: mts.SchemaObj) -> pd.Series:
    """
    Return the indices where invalid values exist

    Parameters
    ----------
    col: pd.Series
    s: hfs.SchemaObj

    Returns
    -------
    pd.Series
        A series of indices with illegal values.
    """
    sers = []
    if len(s.valid_vals) == 0:
        return pd.Series([], dtype='float64')
    for chk in s.valid_vals:
        sers.append(pd.Series(chk.contains(col)))
    cond_df = pd.concat(sers, axis=1).sum(axis=1)
    return pd.Series(cond_df[cond_df == 0].index)


def _illegal_values(col: pd.Series, s: mts.SchemaObj) -> pd.Series:
    illegal = _illegal_values_idx(col=col, s=s)
    illegal_vals = pd.Series(col.iloc[illegal].unique())
    if s.nullable:
        illegal_vals = illegal_vals[~pd.isnull(illegal_vals)]
    return illegal_vals


def validate_data(data: pd.DataFrame, schema: List[mts.SchemaObj]) -> List[str]:
    """
    Validate that all columns are in their valid ranges.

    Parameters
    ----------
    data: pd.DataFrame
        Data to validate
    schema: List[SchemaObj]
        The schema to validate the data against. Columns not found in this list will
        be ignored.

    Returns
    -------
    List[str]
        A list of all errors found during validation. If the length is 0, then
        there are no errors.
    """
    messages = []
    for s in schema:
        col = s.column
        if col not in data.columns:
            messages.append(f"Required column {col} not found.")
            continue
        if not _validate_dtype(data[col], s):
            messages.append(f"Invalid datatype for {col}: {data[col].dtype}, expected: {s.dtype}.")
            continue

        illegal_vals = _illegal_values(data[col], s).tolist()
        if not len(illegal_vals) == 0:
            messages.append(f"Found illegal values {illegal_vals} in {col}, " \
                f"expected value in [{', '.join([str(x) for x in s.valid_vals])}]."
            )
        if not s.nullable and data[col].isna().sum() > 0:
            messages.append(f"Found null values in non-nullable column {col}.")
    return messages
