import numpy as np
import pandas as pd
import pytest

import mltools.data.features.moving as mdfm


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, 4, 5]})


def test_moving_average(data):
    ma = mdfm.MovingAverage(cols=["a"], windows=[1, 2])
    trans = ma.transform(data)

    exp = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "a_ma1": [1, 2, 3, 4, 5],
            "a_ma2": [-1, 1.5, 2.5, 3.5, 4.5],
        },
    )

    assert len(trans.columns) == 3  # 1 for the original column, 2 for the moving average
    assert np.equal(trans.fillna(-1).values, exp.values).all()


def test_expanding_average(data):
    e = mdfm.Expanding(cols=["a"], min_periods=2, agg_fn="sum")
    trans = e.transform(data)

    assert len(trans.columns) == 2  # 1 for the original column, 1 for the expanding average
    assert np.equal(trans.loc[:, trans.columns[-1]].fillna(-1).values, [-1, 3, 6, 10, 15]).all()
