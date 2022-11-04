# pylint: disable=W0621,W0212
import pytest
import pandas as pd
import numpy as np

import ml_tools.data.validation as mtv
import ml_tools.data.validation.schema as mts

@pytest.fixture
def schema_obj_1():
    return mts.SchemaObj(
        column="a",
        dtype="int64",
        valid_vals=[
            mts.SchemaRange(0, 4),
            mts.SchemaList([10, 11, 12])
        ]
    )

@pytest.fixture
def int_schema():
    schema = [
        mts.SchemaObj(column=x, dtype='int64', valid_vals=[mts.SchemaRange(minval=i)]) for i, x in enumerate('abc')
    ]
    return schema


@pytest.fixture
def mixed_numeric_schema():
    schema = [
        mts.SchemaObj(
            column=x,
            dtype='int64' if i % 2 == 0 else 'float64',
        ) for i, x in enumerate('ab')
    ]
    return schema


@pytest.fixture
def int_schema_with_exclusions():
    schema = [
        mts.SchemaObj(
            column='a',
            dtype='int64',
            valid_vals=[
                mts.SchemaRange(0, 1, include_ub=False),
                mts.SchemaRange(1, 99, include_lb=False)
            ]
        )
    ]
    return schema

@pytest.mark.parametrize("ser, expected",
    [
        (pd.Series([0, 1, 2]), True),
        (pd.Series([0, 1, 5]), False),
        (pd.Series([1, 10, 3]), True),
        (pd.Series([-1, -1, -1]), False)
    ]
)
def test_range_validation(ser, expected, schema_obj_1):
    cond = len(mtv._illegal_values(ser, schema_obj_1)) == 0
    assert cond == expected


def test_range_valdation_scrambled_indices(schema_obj_1):
    df = pd.DataFrame([{"idx": 3, "a": 2}, {"idx": 1, "a": 5}, {"idx": 2, "a": 1}])
    df.set_index("idx", inplace=True)
    illegal = mtv._illegal_values(df["a"], schema_obj_1)
    assert illegal.tolist() == [5]


def test_validate_data(int_schema):
    data = pd.DataFrame([{"a": i + 1, "b": i * 2 + 1, "c": i * 3 + 2} for i in range(20)])
    msgs = mtv.validate_data(data=data, schema=int_schema)
    assert len(msgs) == 0


@pytest.mark.parametrize("data", [
    # Allow float columns to be represented by an integer
    (pd.DataFrame([{"a": i, "b": i} for i in range(20)])),
    # Allow float columns to be represented by a float
    (pd.DataFrame([{"a": i, "b": i / 2} for i in range(20)])),
    # Allow float columns to be null
    (pd.DataFrame([{"a": i, "b": None} for i in range(20)])),
])
def test_validate_data_mixed(data, mixed_numeric_schema):
    msgs = mtv.validate_data(data=data, schema=mixed_numeric_schema)
    print(msgs)
    assert len(msgs) == 0


def test_validate_data_invalid_dtype(int_schema):
    data = pd.DataFrame([{"a": "a", "b": i * 2 + 1, "c": i * 3 + 2} for i in range(20)])
    msgs = mtv.validate_data(data=data, schema=int_schema)
    # Single datatype error present
    assert len(msgs) == 1


def test_validate_data_non_nullable():
    schema = [mts.SchemaObj(column='a', dtype='int64', valid_vals=[mts.SchemaRange(minval=0)])]
    data = pd.DataFrame([{"a": None} for i in range(20)])
    msgs = mtv.validate_data(data=data, schema=schema)
    assert len(msgs) == 1


def test_validate_data_nullable():
    schema = [mts.SchemaObj(column='a', dtype='float64', valid_vals=[mts.SchemaRange(minval=0)])]
    data = pd.DataFrame([{"a": None if i % 2 == 0 else np.NaN} for i in range(20)])
    msgs = mtv.validate_data(data=data, schema=schema)
    assert len(msgs) == 0

@pytest.mark.parametrize("data", [
    # Do not allow decimals in integer columns
    (pd.DataFrame([{"a": i / 2, "b": i} for i in range(20)])),
    # Do not allow integer columns to be null
    (pd.DataFrame([{"a": None , "b": i} for i in range(20)])),
    # Do not allow float columns to have extraneous strings
    (pd.DataFrame([{"a": i , "b": None if i % 2 == 0 else 'b'} for i in range(20)])),
])
def test_validate_data_mixed_invalid_dtype(data, mixed_numeric_schema):
    msgs = mtv.validate_data(data=data, schema=mixed_numeric_schema)
    # Single datatype error present
    assert len(msgs) == 1


def test_validate_data_multi_invalid_dtype(int_schema):
    data = pd.DataFrame([{"a": "a", "b": "b", "c": (i + 1) / 3 * 12} for i in range(20)])
    msgs = mtv.validate_data(data=data, schema=int_schema)
    # Multiple datatype errors present (string instead of integer and float instead of integer)
    assert len(msgs) == 3


def test_validate_data_lower_than_min(int_schema):
    data = pd.DataFrame([{"a": i * -1, "b": i * 2 + 1, "c": i * 3 + 2} for i in range(20)])
    msgs = mtv.validate_data(data=data, schema=int_schema)
    # Column `a` has values below the allowed minimum
    assert len(msgs) == 1


def test_validate_data_exclusion_value(int_schema_with_exclusions):
    data = pd.DataFrame([{"a": i} for i in range(20)])
    msgs = mtv.validate_data(data=data, schema=int_schema_with_exclusions)
    # Column `a` has an exclusion value at 1.
    assert len(msgs) == 1


def test_validate_data_missing_col(int_schema):
    data = pd.DataFrame([{"b": i * 2 + 1, "c": i * 3 + 2} for i in range(20)])
    msgs = mtv.validate_data(data=data, schema=int_schema)
    # Column `a` is missing
    assert len(msgs) == 1
