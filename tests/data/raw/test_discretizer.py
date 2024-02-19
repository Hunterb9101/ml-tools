# pylint: disable=redefined-outer-name
import pytest
import numpy as np
import pandas as pd

import mltools.data.raw.interaction as interaction

@pytest.fixture
def discretize_df():
    a = np.array([int(x) for x in list("1"*20 + "2"*20 + "3"*20)]).reshape(60, 1)
    b = np.array([int(x) for x in list("1"*30 + "2"*30)]).reshape(60, 1)

    df = pd.DataFrame(np.hstack([a, b]), columns=["a", "b"])
    return df


def test_discretizer_fit(discretize_df):
    ed = interaction.Discretizer(["a", "b"], max_unique_vals=2)
    ed.fit(discretize_df)

    # 'a' has more than 2 unique vals, and should be bucketed
    assert 'a' in ed.bucket_map.keys()
    # We should persist the quantiles.
    assert len(ed.bucket_map['a']) == 3


def test_discretizer_transform_column(discretize_df):
    ed = interaction.Discretizer(["a", "b"], max_unique_vals=2)
    ed.fit(discretize_df)

    out = ed.transform_column(discretize_df['a'])
    assert len(out) == 60
    for v, c in [(0, 40), (1, 20)]:
        print(v)
        assert v in out.unique()
        assert (out == v).sum() == c


def test_discretizer_transform(discretize_df):
    ed = interaction.Discretizer(["a", "b"], max_unique_vals=2)
    ed.fit(discretize_df)

    out = ed.transform(discretize_df)
    for c in ['q_a']:
        assert c in out.columns
