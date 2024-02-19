import pytest

import mltools.config as mc

@pytest.mark.parametrize("cfg, kwargs", [
    (mc.PathConfig, {}),
    (mc.ModelPathConfig, {"model": "gbm"}),
    (mc.ModelConfig, {})
])
def test_path_config(cfg, kwargs):
    cfg(**kwargs) # Everything initializes...
