import pandas as pd

from mltools.data.raw.encoder import *

def test_target_encoder():
    df = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": ["a", "b", "c", "a", "b", "c"], "y": [1, 1, 1, 0, 0, 0]})
    te = TargetEncoder(["a", "b"], "y")
    te.fit(df)

    out = te.transform(df)

    assert all(df.columns == out.columns)
    pd.testing.assert_series_equal(out["a"], pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], name="a"))
    pd.testing.assert_series_equal(out["b"], pd.Series([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], name="b"))


def test_target_encoder_extraneous_cols():
    """
    Make sure we aren't overwriting columns that we shouldn't be
    """
    b = [1, 2, 3, 1, 2, 3]
    df = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": b, "y": [1, 1, 1, 0, 0, 0]})
    te = TargetEncoder(["a"], "y")
    te.fit(df)

    out = te.transform(df)

    pd.testing.assert_series_equal(out["a"], pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], name="a"))
    pd.testing.assert_series_equal(out["b"], pd.Series(b, name="b"))

def test_fit_transform_target_encoder():
    df = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 1, 2, 3], "y": [1, 1, 1, 0, 0, 0]})
    te = TargetEncoder(["a", "b"], "y")
    out = te.fit_transform(df)

    assert all(df.columns == out.columns)
    pd.testing.assert_series_equal(out["a"], pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], name="a"))
    pd.testing.assert_series_equal(out["b"], pd.Series([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], name="b"))
