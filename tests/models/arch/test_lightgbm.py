"""Tests for native LightGBM model architecture wrappers."""

from __future__ import annotations

import pandas as pd

from mltools.data.schema import DatasetSchema, FittedTransformerSet, FoldDesignMatrix
from mltools.models.arch import LightGBMModel, Task


def _binary_fold(weight_col: str | None = None) -> FoldDesignMatrix:
    train = pd.DataFrame(
        {
            "id": list(range(1, 9)),
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
            "weight": [1.0] * 8,
            "x1": [0.0, 0.1, 0.2, 0.15, 1.0, 1.1, 1.2, 1.15],
            "x2": [1.0, 0.9, 0.8, 0.85, 0.0, 0.1, 0.2, 0.15],
        },
    )
    val = pd.DataFrame(
        {
            "id": [9, 10],
            "target": [0, 1],
            "weight": [1.0, 1.0],
            "x1": [0.12, 1.12],
            "x2": [0.88, 0.08],
        },
    )
    return FoldDesignMatrix(
        fold_id=0,
        schema=DatasetSchema(id_col="id", target_col="target", weight_col=weight_col),
        train=train,
        val=val,
        fitted=FittedTransformerSet(),
    )


def test_lightgbm_binary_prediction_frame_uses_id_and_score_column() -> None:
    model = LightGBMModel(
        name="lgbm",
        task=Task.CLASSIFICATION,
        params={"num_boost_round": 3, "learning_rate": 0.2, "min_data_in_leaf": 1, "num_leaves": 3, "seed": 0},
    )
    fold = _binary_fold(weight_col="weight")

    result = model.fit(fold)
    pred = result.predict(fold.val.loc[:, ["target", "x2", "weight", "id", "x1"]].assign(extra=1))

    assert result is model
    assert model.feature_names_ == ["x1", "x2"]
    assert model.class_order_ == [0, 1]
    assert pred.columns.tolist() == ["id", "score_1"]
    assert pred["id"].tolist() == [9, 10]


def test_lightgbm_feature_importance_gain_schema() -> None:
    model = LightGBMModel(
        name="lgbm_gain",
        task=Task.CLASSIFICATION,
        params={"num_boost_round": 3, "learning_rate": 0.2, "min_data_in_leaf": 1, "num_leaves": 3, "seed": 0},
    ).fit(_binary_fold(weight_col="weight"))

    importance = model.feature_importance()

    assert importance.columns.tolist() == ["model_name", "feature", "importance", "importance_type", "class_label"]
    assert importance["model_name"].unique().tolist() == ["lgbm_gain"]
    assert importance["feature"].tolist() == ["x1", "x2"]
    assert importance["importance_type"].unique().tolist() == ["gain"]
