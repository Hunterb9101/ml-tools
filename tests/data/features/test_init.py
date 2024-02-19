# pylint: disable=redefined-outer-name
import pandas as pd
import pytest

import mltools.data.features as kpf

@pytest.fixture
def ex_fsp() -> kpf.FeatureSelectionPipeline:
    class A(kpf.BasePruner):
        def __init__(self):
            super().__init__()
            self._drop_cols = []
            self._is_fit = False

        def fit(self, df): # pylint: disable=unused-argument
            self._drop_cols = ['a']
            self._is_fit = True

    class B(kpf.BasePruner):
        def __init__(self):
            super().__init__()
            self._drop_cols = []
            self._is_fit = False

        def fit(self, df): # pylint: disable=unused-argument
            self._drop_cols = ['b']
            self._is_fit = True

    fsp = kpf.FeatureSelectionPipeline([A(), B()])
    return fsp


def test_feature_selection_pipeline():
    """ Make sure it initializes"""
    kpf.FeatureSelectionPipeline([])


def test_feature_selection_pipeline_fit(ex_fsp):
    """ Make sure that all columns are added to drop_cols"""
    df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    ex_fsp.fit(df)
    assert ex_fsp.drop_cols == ['a', 'b']


def test_feature_selection_pipeline_fit_transform(ex_fsp):
    """ Make sure that the result is returned correctly. """
    df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
    df = ex_fsp.fit_transform(df)
    assert df.columns.tolist() == ['c']
