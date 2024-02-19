# pylint: disable=redefined-outer-name
import numpy as np
import pandas as pd
import pytest

import mltools.data.process.tts as mdpt

@pytest.fixture
def data() -> pd.DataFrame:
    x = np.repeat(["A", "B", "C", "D"], 10)
    return pd.DataFrame({"x": x, "y": np.random.rand(40)})


@pytest.mark.parametrize("val_size, test_size, val_exp, test_exp", [
    (0.2, 0.2, 8, 8),
    (0.25, 0.5, 10, 20),
    (5, 5, 5, 5)
])
def test_compute_tts_split(data, val_size, test_size, val_exp, test_exp):
    tvt = mdpt.compute_tts_split(data, val_size=val_size, test_size=test_size, tts_kwargs=None)
    assert np.isclose(len(tvt.train), len(data) - val_exp - test_exp, atol=2)
    assert np.isclose(len(tvt.val), val_exp, atol=2)
    assert np.isclose(len(tvt.test), test_exp, atol=2)


@pytest.mark.parametrize("val_size, test_size", [
    (-1, 0.0),
    (1.0, 0.5),
    (0.0, 0.0),
    (0.0, 0.25),
    (0.5, 0.5),
    (20, 20)
])
def test_compute_tts_split_invalid(data, val_size, test_size):
    with pytest.raises(ValueError):
        mdpt.compute_tts_split(data, val_size=val_size, test_size=test_size, tts_kwargs=None)


@pytest.mark.parametrize("val_values, test_values, val_to_test_ratio, val_exp, test_exp", [
    (["A"], ["B"], 0.5, 10, 10),
    (["A", "B"], ["B", "D"], 0.5, 15, 15),
    (["A", "B"], ["D"], 0.5, 20, 10),
    (["A", "B"], ["A", "B"], 0.75, 15, 5),
    (["A", "B"], ["A", "B"], 0.25, 5, 15)
])
def test_compute_oos_tts_split(data, val_values, test_values, val_to_test_ratio, val_exp, test_exp):
    tvt = mdpt.compute_oos_tts_split(
        data,
        "x",
        values_val=val_values,
        values_test=test_values,
        val_to_test_ratio=val_to_test_ratio,
        tts_kwargs=None
    )
    assert np.isclose(len(tvt.val), val_exp, atol=2)
    assert np.isclose(len(tvt.test), test_exp, atol=2)

@pytest.mark.parametrize("val_values, test_values, val_to_test_ratio", [
    (["A"], ["A"], 1.5),
    (["A", "B", "C", "D"], ["A", "B", "C", "D"], 0.5),
    (["A", "B", "C", "D"], [], 0.0),
    ([], ["A", "B", "C", "D"], 0.5)
])
def test_compute_oos_tts_split_invalid(data, val_values, test_values, val_to_test_ratio):
    with pytest.raises(ValueError):
        mdpt.compute_oos_tts_split(
            data,
            "x",
            values_val=val_values,
            values_test=test_values,
            val_to_test_ratio=val_to_test_ratio,
            tts_kwargs=None
        )
