# pylint: disable=redefined-outer-name
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from mltools.data.transformers.pruners.rfe import RFEPruner


@pytest.fixture
def data() -> pd.DataFrame:
    useful = np.random.uniform(100, size=(100, 4))
    useless = np.random.uniform(100, size=(100, 5))
    target = useful.sum(axis=1).reshape(-1, 1)

    data_ = np.hstack([useful, useless, target])
    return pd.DataFrame(
        data_,
        columns=[f"useful_{i}" for i in range(4)] + [f"useless_{i}" for i in range(5)] + ["target"],
    )


def test_rfe_fit(data):
    rfe = RFEPruner(model=lgb.LGBMRegressor(), n_features_to_select=4, target_col="target")
    rfe.fit(data)
    assert rfe.drop_cols == ["useless_0", "useless_1", "useless_2", "useless_3", "useless_4"]
