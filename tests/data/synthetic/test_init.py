# pylint: disable=W0621,W0212

import pytest
import pandas as pd
import numpy as np

import ml_tools.data.synthetic as core

@pytest.fixture
def schema_1():
    return [
        core.MockColumn("a", distribution=np.random.random),
        core.MockColumn("b", distribution=np.random.normal, distribution_kwargs={"loc": 5}, is_useful=False),
        core.DupColumn("c", redundant_of="a")
    ]


@pytest.fixture
def schema_2():
    return [
        core.MockColumn("a", distribution=np.random.random),
        core.MockColumn("b", distribution=np.random.random),
        core.MockColumn("c", distribution=np.random.random, is_useful=False),
        core.DupColumn("d", redundant_of="a")
    ]


@pytest.mark.parametrize(
    "init",
    [
        {"name": "a", "distribution": np.random.randn, "distribution_kwargs": None, "redundant_of": "b"},
        {"name": "a", "distribution": None, "distribution_kwargs": {"mu": 0.5}, "redundant_of": "b"},
        {"name": "a", "distribution": None, "distribution_kwargs": None},
        {"name": "a", "distribution": None, "distribution_kwargs": None, "is_useful": False}
    ]
)
def test_mock_column_early_errors(init):
    """ Throw errors when certain column configurations are not available. """
    with pytest.raises(ValueError):
        core.MockColumn(**init)


def test_mock_column_warns():
    """ Warn when a column is "useful", but is also duplicated from another. """
    with pytest.warns():
        core.MockColumn(name="a", redundant_of="b")


def test_dup_column_init():
    dup_col = core.DupColumn("a", "b")

    assert dup_col.redundant_of == "b"
    assert not dup_col.is_useful


def test_mock_mgr_init(schema_1):
    core.MockManagerClassification(n_rows=100, columns=schema_1, idx_cols=["idx"])


def test_mock_mgr_gen_X_unique_col_values(schema_2):
    mock_mgr = core.MockManagerClassification(n_rows=100, columns=schema_2, idx_cols=["idx"])
    df = mock_mgr._generate_X()
    assert not np.equal(df["a"].values, df["b"].values).all()
    assert not np.equal(df["a"].values, df["c"].values).all()
    assert np.equal(df["a"].values, df["d"].values).all()


def test_mock_mgr_gen_X(schema_1):
    mock_mgr = core.MockManagerClassification(n_rows=100, columns=schema_1, idx_cols=["idx"])
    df = mock_mgr._generate_X()

    # "c" should be a duplicate of "a" as defined in schema_1
    assert np.equal(df["a"].values, df["c"].values).all()
    assert len(df["idx"]) == len(df["idx"].unique())


def test_mock_mgr_gen_X_correct_size(schema_1):
    mock_mgr = core.MockManagerClassification(n_rows=100, columns=schema_1, idx_cols=["idx"])
    df = mock_mgr._generate_X()
    # 3 generated columns from schema_1 and 1 index column
    assert len(df.columns) == 4
    # Assert length is correct
    assert len(df) == 100


def test_mock_mgr_gen_X_single_idx(schema_1):
    mock_mgr = core.MockManagerClassification(n_rows=100, columns=schema_1, idx_cols=["idx"])
    df = mock_mgr._generate_X()
    assert len(df["idx"]) == len(df["idx"].unique())


def test_mock_mgr_gen_X_reproducible(schema_1):
    mock_mgr = core.MockManagerClassification(n_rows=100, columns=schema_1)
    df = mock_mgr._generate_X()

    mock_mgr2 = core.MockManagerClassification(n_rows=100, columns=schema_1)
    df2 = mock_mgr2._generate_X()
    assert df.equals(df2)


def test_mock_mgr_gen_y(schema_1):
    mock_mgr = core.MockManagerClassification(n_rows=100, columns=schema_1, idx_cols=["idx"])
    df = mock_mgr._generate_X()
    y = mock_mgr._generate_Y(df)
    assert len(y.unique()) == 2


def test_mock_mgr_gen_y_reproducible(schema_1):
    mock_mgr = core.MockManagerClassification(n_rows=100, columns=schema_1, idx_cols=["idx"])
    df = mock_mgr._generate_X()
    y = mock_mgr._generate_Y(df)

    mock_mgr2 = core.MockManagerClassification(n_rows=100, columns=schema_1, idx_cols=["idx"])
    df2 = mock_mgr._generate_X()
    y2 = mock_mgr2._generate_Y(df2)
    assert y.equals(y2)


@pytest.mark.parametrize("threshold", [(0.01), (0.1), (0.25), (0.5), (0.75), (0.99)])
def test_mock_mgr_gen_y_thresholds(schema_1, threshold):
    mock_mgr = core.MockManagerClassification(
        n_rows=1000,
        columns=schema_1,
        idx_cols=["idx"],
        class_balance=threshold
    )
    df = mock_mgr._generate_X()
    y = mock_mgr._generate_Y(df)
    assert np.isclose(y.mean(), threshold, atol=0.02)
