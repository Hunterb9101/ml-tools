import pytest
import pandas as pd
import numpy as np

import mltools.data.raw.interaction as interaction


@pytest.fixture
def df():
    a = np.array([int(x) for x in list("7"*20 + "8"*20 + "9"*20)]).reshape(60, 1)
    b = np.array([int(x) for x in list("8"*30 + "9"*30)]).reshape(60, 1)

    df = pd.DataFrame(np.hstack([a, b]), columns=["a", "b"])
    return df

@pytest.fixture
def df2(df):
    c = np.array([int(x) for x in list("4"*15 + "5"*15 + "6"*15 + "7"*15)]).reshape(60, 1)
    df['c'] = c
    return df

def test_interaction_fit(df):
    i = interaction.Interaction(["a", "b"], max_unique_vals=2)
    i.fit(df)

    assert len(i.means) == 2


def test_interaction_transform_column(df2):
    """
    Assert that the mapping is working correctly for an interaction column.
    
    The math is shown below for the expected means.
    """
    #     a b
    # 20x 7 8
    # 10x 8 8
    # 10x 8 9
    # 20x 9 9

    # E[a|b=8] = 20*7 + 10*8 = 220 / 30 = 7.333
    # E[a|b=9] = 10*8 + 20*9 = 260 / 30 = 8.666
    i = interaction.Interaction(["a", "b", "c"], max_unique_vals=2)
    i.fit(df2)

    out = i.transform(df2)
    assert out['a_b'].isna().sum() == 0
    assert np.all(np.isclose(out[out['b'] == 8]['a_b'], 7.333, atol=0.01))
    assert np.all(np.isclose(out[out['b'] == 9]['a_b'], 8.666, atol=0.01))
