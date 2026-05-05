"""Tests for sklearn model architecture wrappers."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from mltools.data.schema import DatasetSchema, FittedTransformerSet, FoldDesignMatrix
from mltools.models.arch import SklearnModel, Task


def _binary_fold() -> FoldDesignMatrix:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "target": [0, 0, 0, 1, 1, 1],
            "x1": [0.0, 0.2, 0.1, 1.0, 1.2, 0.9],
            "x2": [1.0, 0.7, 1.2, 0.1, 0.2, 0.0],
        },
    )
    val = pd.DataFrame(
        {
            "id": [7, 8],
            "target": [0, 1],
            "x1": [0.15, 1.1],
            "x2": [0.9, 0.1],
        },
    )
    return FoldDesignMatrix(
        fold_id=0,
        schema=DatasetSchema(id_col="id", target_col="target"),
        train=train,
        val=val,
        fitted=FittedTransformerSet(),
    )


def test_sklearn_logistic_regression_binary_prediction_frame_is_stable() -> None:
    model = SklearnModel(
        name="logreg",
        task=Task.CLASSIFICATION,
        estimator=LogisticRegression(random_state=0),
        params={"solver": "lbfgs"},
    )
    fold = _binary_fold()

    result = model.fit(fold)
    shuffled = fold.val.loc[:, ["target", "x2", "id", "x1"]].assign(extra=99)
    pred = result.predict(shuffled)

    assert result is model
    assert model.feature_names_ == ["x1", "x2"]
    assert model.class_order_ == [0, 1]
    assert pred.columns.tolist() == ["id", "score_1"]
    assert pred["id"].tolist() == [7, 8]


def test_sklearn_random_forest_feature_importance() -> None:
    model = SklearnModel(
        name="forest",
        task=Task.CLASSIFICATION,
        estimator=RandomForestClassifier(n_estimators=5, random_state=0),
    ).fit(_binary_fold())

    importance = model.feature_importance()

    assert importance.columns.tolist() == ["model_name", "feature", "importance", "importance_type", "class_label"]
    assert importance["model_name"].unique().tolist() == ["forest"]
    assert importance["feature"].tolist() == ["x1", "x2"]
    assert importance["importance_type"].unique().tolist() == ["feature_importances_"]


def test_sklearn_multiclass_emits_one_score_per_class_position() -> None:
    train = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "target": [0, 1, 2, 0, 1, 2],
            "x1": [0.0, 1.0, 2.0, 0.1, 1.1, 2.1],
            "x2": [0.0, 0.2, 0.4, 0.1, 0.3, 0.5],
        },
    )
    fold = FoldDesignMatrix(
        fold_id=0,
        schema=DatasetSchema(id_col="id", target_col="target"),
        train=train,
        val=train.iloc[:3].copy(),
        fitted=FittedTransformerSet(),
    )

    model = SklearnModel(
        name="multi",
        task=Task.CLASSIFICATION,
        estimator=LogisticRegression(random_state=0),
    ).fit(fold)
    pred = model.predict(fold.val)

    assert pred.columns.tolist() == ["id", "score_0", "score_1", "score_2"]


def test_sklearn_classification_requires_predict_proba() -> None:
    fold = _binary_fold()
    model = SklearnModel(name="no_proba", task=Task.CLASSIFICATION, estimator=SVC()).fit(fold)

    with pytest.raises(NotImplementedError, match="predict_proba"):
        model.predict(fold.val)
