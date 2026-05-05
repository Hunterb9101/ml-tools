from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mltools.data.contracts import DatasetSchema, FittedTransformerSet, FoldDesignMatrix
from mltools.models.arch.base import BaseModelWrapper, Task
from mltools.models.cv import CVTrainingResult, FoldTrainingResult, train_cv


class RecordingModel(BaseModelWrapper):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, task=Task.CLASSIFICATION)
        self.fit_fold_ids: list[int] = []
        self.predicted_ids: list[list[int]] = []

    def fit(self, fold: Any) -> RecordingModel:
        self.fit_fold_ids.append(fold.fold_id)
        self.is_fit = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.predicted_ids.append(df["id"].tolist())
        return pd.DataFrame({"id": df["id"], "score_1": df["feature"]})


class SklearnClassifierModel(BaseModelWrapper):
    def __init__(self) -> None:
        super().__init__(name="sklearn_classifier", task=Task.CLASSIFICATION)
        self.estimator = LogisticRegression(solver="liblinear", random_state=42)
        self.feature_names_: list[str] | None = None

    def fit(self, fold: Any) -> SklearnClassifierModel:
        self.feature_names_ = ["x0", "x1"]
        self.estimator.fit(fold.train[self.feature_names_], fold.train["target"])
        self.is_fit = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is None:
            msg = "model is not fit."
            raise ValueError(msg)
        probabilities = self.estimator.predict_proba(df[self.feature_names_])
        return pd.DataFrame({"id": df["id"], "score_1": probabilities[:, 1]})


def _fold(fold_id: int, train_ids: list[int], val_ids: list[int]) -> FoldDesignMatrix:
    schema = DatasetSchema(id_col="id", target_col="target")
    return FoldDesignMatrix(
        fold_id=fold_id,
        schema=schema,
        train=pd.DataFrame({"id": train_ids, "target": [int(value % 2) for value in train_ids], "feature": train_ids}),
        val=pd.DataFrame({"id": val_ids, "target": [int(value % 2) for value in val_ids], "feature": val_ids}),
        fitted=FittedTransformerSet(),
    )


def _model(name: str = "result_model") -> RecordingModel:
    return RecordingModel(name=name)


def test_train_cv_calls_model_factory_once_per_fold_and_preserves_order() -> None:
    folds = [_fold(2, [10, 11], [20]), _fold(1, [20, 21], [10])]
    created_models: list[RecordingModel] = []

    def model_factory() -> RecordingModel:
        model = RecordingModel(name=f"model_{len(created_models)}")
        created_models.append(model)
        return model

    result = train_cv(model_factory=model_factory, folds=folds)

    assert len(created_models) == 2
    assert result.models() == created_models
    assert [fold_result.fold_id for fold_result in result.fold_results] == [2, 1]
    assert [model.fit_fold_ids for model in created_models] == [[2], [1]]


def test_train_cv_rejects_reused_model_instances() -> None:
    folds = [_fold(0, [1], [2]), _fold(1, [2], [1])]
    reused_model = RecordingModel(name="reused")

    def model_factory() -> RecordingModel:
        return reused_model

    with pytest.raises(ValueError, match="fresh model instance"):
        train_cv(model_factory=model_factory, folds=folds)


def test_oof_predictions_concatenate_validation_predictions() -> None:
    result = CVTrainingResult(
        id_col="id",
        fold_results=[
            FoldTrainingResult(
                fold_id=0,
                model=_model("fold_0"),
                train_predictions=pd.DataFrame({"id": [2, 3], "score_1": [0.2, 0.3]}),
                val_predictions=pd.DataFrame({"id": [1], "score_1": [0.1]}),
            ),
            FoldTrainingResult(
                fold_id=1,
                model=_model("fold_1"),
                train_predictions=pd.DataFrame({"id": [1, 3], "score_1": [0.4, 0.5]}),
                val_predictions=pd.DataFrame({"id": [2], "score_1": [0.6]}),
            ),
        ],
    )

    pd.testing.assert_frame_equal(
        result.oof_predictions().reset_index(drop=True),
        pd.DataFrame({"id": [1, 2], "score_1": [0.1, 0.6]}),
    )


def test_train_predictions_concatenate_train_predictions() -> None:
    result = CVTrainingResult(
        id_col="id",
        fold_results=[
            FoldTrainingResult(
                fold_id=0,
                model=_model("fold_0"),
                train_predictions=pd.DataFrame({"id": [2, 3], "score_1": [0.2, 0.3]}),
                val_predictions=pd.DataFrame({"id": [1], "score_1": [0.1]}),
            ),
            FoldTrainingResult(
                fold_id=1,
                model=_model("fold_1"),
                train_predictions=pd.DataFrame({"id": [1, 3], "score_1": [0.4, 0.5]}),
                val_predictions=pd.DataFrame({"id": [2], "score_1": [0.6]}),
            ),
        ],
    )

    pd.testing.assert_frame_equal(
        result.train_predictions().reset_index(drop=True),
        pd.DataFrame({"id": [2, 3, 1, 3], "score_1": [0.2, 0.3, 0.4, 0.5]}),
    )


def test_oof_predictions_reject_duplicate_ids_when_id_column_is_available() -> None:
    result = CVTrainingResult(
        id_col="id",
        fold_results=[
            FoldTrainingResult(
                fold_id=0,
                model=_model("fold_0"),
                train_predictions=pd.DataFrame(),
                val_predictions=pd.DataFrame({"id": [1], "score_1": [0.1]}),
            ),
            FoldTrainingResult(
                fold_id=1,
                model=_model("fold_1"),
                train_predictions=pd.DataFrame(),
                val_predictions=pd.DataFrame({"id": [1], "score_1": [0.2]}),
            ),
        ],
    )

    with pytest.raises(ValueError, match="duplicate ids"):
        result.oof_predictions()


def test_train_cv_rejects_empty_folds() -> None:
    def model_factory() -> RecordingModel:
        return RecordingModel(name="unused")

    with pytest.raises(ValueError, match="at least one fold"):
        train_cv(model_factory=model_factory, folds=[])


def test_train_cv_rejects_duplicate_fold_ids() -> None:
    folds = [_fold(0, [1], [2]), _fold(0, [2], [1])]

    def model_factory() -> RecordingModel:
        return RecordingModel(name="unused")

    with pytest.raises(ValueError, match="duplicated fold ids"):
        train_cv(model_factory=model_factory, folds=folds)


def test_train_cv_trains_small_sklearn_classifier_across_synthetic_folds() -> None:
    df = pd.DataFrame(
        {
            "id": list(range(12)),
            "x0": [0.0, 0.1, 1.0, 1.1, 0.2, 0.3, 1.2, 1.3, 0.4, 0.5, 1.4, 1.5],
            "x1": [1.0, 0.9, 0.0, 0.1, 0.8, 0.7, 0.2, 0.3, 0.6, 0.5, 0.4, 0.5],
            "target": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        },
    )
    schema = DatasetSchema(id_col="id", target_col="target")
    folds = [
        FoldDesignMatrix(
            fold_id=0,
            schema=schema,
            train=df.iloc[4:].copy(),
            val=df.iloc[:4].copy(),
            fitted=FittedTransformerSet(),
        ),
        FoldDesignMatrix(
            fold_id=1,
            schema=schema,
            train=pd.concat([df.iloc[:4], df.iloc[8:]]),
            val=df.iloc[4:8].copy(),
            fitted=FittedTransformerSet(),
        ),
        FoldDesignMatrix(
            fold_id=2,
            schema=schema,
            train=df.iloc[:8].copy(),
            val=df.iloc[8:].copy(),
            fitted=FittedTransformerSet(),
        ),
    ]

    result = train_cv(model_factory=SklearnClassifierModel, folds=folds)
    oof_predictions = result.oof_predictions()

    assert len(result.fold_results) == 3
    assert all(isinstance(model, SklearnClassifierModel) for model in result.models())
    assert sorted(oof_predictions["id"].tolist()) == list(range(12))
    assert oof_predictions["score_1"].between(0, 1).all()


def test_train_cv_does_not_catch_backend_training_errors() -> None:
    class FailingModel(BaseModelWrapper):
        def __init__(self) -> None:
            super().__init__(name="failing", task=Task.CLASSIFICATION)

        def fit(self, fold: Any) -> Any:
            _ = fold
            msg = "backend failed"
            raise RuntimeError(msg)

        def predict(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    with pytest.raises(RuntimeError, match="backend failed"):
        train_cv(model_factory=FailingModel, folds=[_fold(0, [1], [2])])
