"""Tests for shared model wrapper behavior."""

from __future__ import annotations

from typing import Self

import pandas as pd
import pytest

from mltools.data.schema import DatasetSchema, FittedTransformerSet, FoldDesignMatrix
from mltools.models.arch.base import BaseModelWrapper, Task
from mltools.models.arch.base import FoldDesignMatrix as BaseFoldDesignMatrix


class DummyModel(BaseModelWrapper):
    """Concrete wrapper for exercising base behavior."""

    def fit(self, fold: BaseFoldDesignMatrix) -> Self:
        """Record fold context without backend fitting."""
        self._record_fit_context(fold)
        if self.task is Task.CLASSIFICATION:
            self.class_order_ = [0, 1]
        self.is_fit = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return deterministic binary scores."""
        self._select_recorded_features(df)
        return self._classification_prediction_frame(df, [0.25] * len(df))


def test_task_public_contract() -> None:
    assert Task.CLASSIFICATION.value == "classification"
    assert Task.REGRESSION.value == "regression"


def test_base_params_are_copied() -> None:
    params = {"alpha": 1}
    model = DummyModel(name="dummy", task=Task.CLASSIFICATION, params=params)
    params["alpha"] = 2
    assert model.params == {"alpha": 1}


def test_predict_before_fit_fails_clearly() -> None:
    model = DummyModel(name="dummy", task=Task.CLASSIFICATION)
    df = pd.DataFrame({"id": [1], "x": [0.1]})
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict(df)


def test_missing_predict_feature_fails_clearly() -> None:
    fold = FoldDesignMatrix(
        fold_id=0,
        schema=DatasetSchema(id_col="id", target_col="target"),
        train=pd.DataFrame({"id": [1, 2], "target": [0, 1], "x": [0.1, 0.2], "y": [1.1, 1.2]}),
        val=pd.DataFrame({"id": [3], "target": [0], "x": [0.3], "y": [1.3]}),
        fitted=FittedTransformerSet(),
    )
    model = DummyModel(name="dummy", task=Task.CLASSIFICATION).fit(fold)

    with pytest.raises(ValueError, match="Missing feature columns"):
        model.predict(pd.DataFrame({"id": [3], "target": [0], "x": [0.3]}))


def test_prediction_frame_omits_target_and_uses_binary_score() -> None:
    fold = FoldDesignMatrix(
        fold_id=0,
        schema=DatasetSchema(id_col="id", target_col="target"),
        train=pd.DataFrame({"id": [1, 2], "target": [0, 1], "x": [0.1, 0.2]}),
        val=pd.DataFrame({"id": [3], "target": [0], "x": [0.3]}),
        fitted=FittedTransformerSet(),
    )
    model = DummyModel(name="dummy", task=Task.CLASSIFICATION).fit(fold)
    pred = model.predict(pd.DataFrame({"target": [0], "extra": [10], "x": [0.3], "id": [3]}))

    assert pred.columns.tolist() == ["id", "score_1"]
    assert pred["score_1"].tolist() == [0.25]
