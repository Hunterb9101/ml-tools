import pytest
import numpy as np
import pandas as pd

import ml_tools.data.validation.schema as mts


@pytest.mark.parametrize("minval, maxval, include_lb, include_ub",
    [
        (0, 1, True, True),
        (0, None, True, True),
        (None, 1, True, True)
    ]
)
def test_schema_range_init(minval, maxval, include_lb, include_ub):
    mts.SchemaRange(minval, maxval, include_lb, include_ub)


@pytest.mark.parametrize("input_fn", [np.array, pd.Series, None])
@pytest.mark.parametrize("schemarange, val, expected",
    [
        # Check that float ranges (Inclusive) work correctly
        (mts.SchemaRange(0.0, 2.0, True, True), 0.0, True),
        (mts.SchemaRange(0.0, 2.0, True, True), 1.0, True),
        (mts.SchemaRange(0.0, 2.0, True, True), 2.0, True),
        (mts.SchemaRange(0.0, 2.0, True, True), -1.0, False),
        (mts.SchemaRange(0.0, 2.0, True, True), -1, False),
        (mts.SchemaRange(0.0, 2.0, True, True), 5.0, False),
        (mts.SchemaRange(0.0, 2.0, True, True), 5, False),
        (mts.SchemaRange(0.0, 2.0, True, True), 1, True),
        # Check float ranges (Inclusive, Exclusive) work correctly
        (mts.SchemaRange(0.0, 2.0, True, False), 1.0, True),
        (mts.SchemaRange(0.0, 2.0, True, False), 2.0, False),
        # Check float ranges (Exclusive, Inclusive) work correctly
        (mts.SchemaRange(0.0, 2.0, False, True), 0.0, False),
        (mts.SchemaRange(0.0, 2.0, False, True), 1.0, True),
        # Check that integer ranges (Inclusive) work correctly
        (mts.SchemaRange(0, 10, True, True), 0, True),
        (mts.SchemaRange(0, 10, True, True), 5, True),
        (mts.SchemaRange(0, 10, True, True), 10, True),
        (mts.SchemaRange(0, 10, True, True), -1.0, False),
        (mts.SchemaRange(0, 10, True, True), -1, False),
        (mts.SchemaRange(0, 10, True, True), 11.0, False),
        (mts.SchemaRange(0, 10, True, True), 11, False),
        (mts.SchemaRange(0, 10, True, True), 2.333333, True),
        # Check float ranges (Inclusive, Exclusive) work correctly
        (mts.SchemaRange(0, 10, True, False), 5, True),
        (mts.SchemaRange(0, 10, True, False), 10, False),
        # Check float ranges (Exclusive, Inclusive) work correctly
        (mts.SchemaRange(0, 10, False, True), 0, False),
        (mts.SchemaRange(0, 10, False, True), 5, True),
        # Check single min/single max
        (mts.SchemaRange(minval=0), -5, False),
        (mts.SchemaRange(minval=0), 5, True),
        (mts.SchemaRange(maxval=10), 0, True),
        (mts.SchemaRange(maxval=10), 11, False)
    ]
)
def test_schema_range_contains(input_fn, schemarange, val, expected):
    data = input_fn([val]) if input_fn is not None else [val]
    assert schemarange.contains(data).tolist() == [expected]


@pytest.mark.parametrize("input_fn", [np.array, pd.Series, None])
@pytest.mark.parametrize("schemarange, val",
    [
        (mts.SchemaRange(0, 100, True, True), "a"),
        (mts.SchemaRange(0, 100, True, True), [0, 1, "a"])
    ]
)
def test_schema_bad_inputs(schemarange, val, input_fn):
    if not isinstance(val, list):
        val = [val]
    with pytest.raises(Exception):
        schemarange.contains(input_fn(val))


@pytest.mark.parametrize("input_fn", [np.array, pd.Series, None])
@pytest.mark.parametrize("schemalist, val, expected",
    [
        (mts.SchemaList(["a", "b", "c"]), "a", True),
        (mts.SchemaList(["a", "b", "c"]), 1, False),
        (mts.SchemaList([0, 1, 2]), "a", False),
        (mts.SchemaList([0, 1, 2]), 1, True)
    ]
)
def test_schema_list_contains(schemalist, val, expected, input_fn):
    data = input_fn([val]) if input_fn is not None else [val]
    assert schemalist.contains(data).tolist() == [expected]


def test_unpack_pack_schemalist():
    sl = mts.SchemaList(["a", "b", "c"])
    serialized = mts.dict_to_valid_vals(sl.to_dict())
    assert sl == serialized


def test_unpack_pack_schemarange():
    sl = mts.SchemaRange(minval=3, maxval=10)
    serialized = mts.dict_to_valid_vals(sl.to_dict())
    assert sl == serialized


def test_bad_dict_to_valid_vals():
    with pytest.raises(ValueError):
        mts.dict_to_valid_vals({"minval": 10, "vals": [1, 2, 3]})
