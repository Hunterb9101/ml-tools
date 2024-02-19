import numpy as np
import pandas as pd


from mltools.data.features.moment import MomentPruner


def test_moment_fit():
    df = pd.DataFrame(np.random.uniform(100, size=(100, 3)), columns=list('abc'), index=np.arange(100))
    mp = MomentPruner(min_variance=0.0, target_col='c')
    mp.fit(df)
    assert mp.drop_cols == []


def test_moment_fit_drop_high_correlation():
    """
    Remove columns with a high correlation with another column
    """
    a = np.random.uniform(100, size=(100,))
    c = np.random.uniform(100, size=(100,))
    df = pd.DataFrame({'a': a, 'b': a, 'c': c}, columns=list('abc'), index=np.arange(100))
    mp = MomentPruner(min_variance=0.0, max_corr=0.95, target_col='c')
    mp.fit(df)
    assert mp.drop_cols == ['b']


def test_moment_fit_drop_low_variance():
    """
    Remove columns that have a constant integer value
    """
    a = np.random.uniform(100, size=(100,))
    b = np.random.choice([0], size=(100,))
    c = np.random.uniform(100, size=(100,))
    df = pd.DataFrame({'a': a, 'b': b, 'c': c}, index=np.arange(100))
    mp = MomentPruner(max_corr=1.0, target_col='c')
    mp.fit(df)
    assert mp.drop_cols == ['b']
