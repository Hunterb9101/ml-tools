import numpy as np
import pandas as pd
import pytest

import ml_tools.data.stability as mtds


def test_counts_by_category():
    n = pd.Series(1 if x < 5 else 0 for x in range(10))
    m = pd.Series(2 if x < 5 else 1 for x in range(10))
    df = mtds._counts_by_category(n, m)
    expected = pd.DataFrame([
        {"bin": 0, "old": 5, "new": 0},
        {"bin": 1, "old": 5, "new": 5},
        {"bin": 2, "old": 0, "new": 5}
    ])
    assert df.equals(expected)


def test_si():
    n = np.random.uniform(0, 1, size=1000)
    m = np.random.uniform(0, 1, size=1000)
    mtds.si(new=n, old=m, bins=10)


@pytest.mark.parametrize("method", ['chisq', 'industry', 'norm'])
@pytest.mark.parametrize("quantile", [0.95, 0.99, 0.999])
@pytest.mark.parametrize("old, new, expected",
    [
        (np.random.uniform(0, 1, size=1000), np.random.uniform(0, 1, size=500), False),
        (np.random.uniform(0, 1, size=1000), np.random.beta(3, 1, size=500), True),
        (np.random.uniform(0, 1, size=1000), np.random.uniform(-0.5, 1.5, size=500), True),
        (np.random.beta(0.5, 0.5, size=1000), np.random.normal(0, 1, size=500), True),
        (np.random.normal(0, 1.1, size=1000), np.random.normal(0, 1, size=500), False),
        (np.random.normal(0.05, 1, size=1000), np.random.normal(0, 1, size=500), False),
    ]
)
def test_si_significance(old, new, expected, method, quantile):
    assert mtds.si_is_signifcant(old=old, new=new, method=method, quantile=quantile) == expected


def test_si_calculation():
    """
    Check that worked example from https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations
    on table 1.2 returns the same value.
    """
    n = pd.Series([1 if x < 10085 else 0 for x in range(39786)])
    m = pd.Series([1 if x < 32000 else 0 for x in range(181231)])
    df = mtds._si_df(n, m, is_categorical=True)
    assert np.round(df[df["bin"] == 1]["si"].tolist()[0], decimals=3) == 0.028
