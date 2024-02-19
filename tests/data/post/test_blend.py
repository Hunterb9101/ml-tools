import pytest
import pandas as pd
import numpy as np

import mltools.data.post.blend as kb

def test_blender_init():
    kb.Blender([1, 1, 1], ["a", "b", "c"])

@pytest.mark.parametrize("weights", [(0, 0 ,0), (-1, 1, 1)])
def test_blender_init_bad_weights(weights):
    with pytest.raises(ValueError):
        kb.Blender(weights, ["a", "b", "c"])

def test_blender_transform():
    blend = kb.Blender([1, 1, 1], ["a", "b", "c"], out_col='blend')
    df = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 1, 2, 3], "c": [3, 2, 1, 3, 2, 1]})
    out = blend.transform(df)

    assert 'blend' in out.columns
    assert all(np.isclose(out['blend'], [5/3, 5/3, 5/3, 2, 2, 2]))


def test_blender_transform_variable_weights():
    blend = kb.Blender([3, 1, 0], ["a", "b", "c"], out_col='blend')
    df = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 1, 2, 3], "c": [3, 2, 1, 3, 2, 1]})
    out = blend.transform(df)

    assert 'blend' in out.columns
    assert all(np.isclose(out['blend'], [1, 1.25, 1.5, 1.75, 2, 2.25]))

def test_blender_transform_variable_weights_not_normalized():
    blend = kb.Blender([3, 1, 0], ["a", "b", "c"], out_col='blend', normalize=False)
    df = pd.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 1, 2, 3], "c": [3, 2, 1, 3, 2, 1]})
    out = blend.transform(df)

    assert 'blend' in out.columns
    assert all(np.isclose(out['blend'], [4, 5, 6, 7, 8, 9]))
