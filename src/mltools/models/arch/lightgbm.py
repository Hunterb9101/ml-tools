"""Mutable native LightGBM model wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import lightgbm as lgb
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from mltools.models.arch.base import BINARY_CLASS_COUNT, BaseModelWrapper, FoldDesignMatrix, Task

if TYPE_CHECKING:
    import pandas as pd

PROBABILITY_THRESHOLD = 0.5


_TRAINING_CONTROL_KEYS = {
    "balanced_accuracy_metric_only",
    "diagnostic_metric",
    "early_stopping_rounds",
    "log_evaluation_period",
    "n_iter",
    "num_boost_round",
    "use_balanced_accuracy_eval",
}


class LightGBMModel(BaseModelWrapper):
    """Wrap native LightGBM training and prediction."""

    def __init__(self, name: str, task: Task, params: dict[str, Any] | None = None) -> None:
        """Initialize a native LightGBM model wrapper."""
        super().__init__(name=name, task=task, params=params)
        self.booster_: lgb.Booster | None = None
        self._label_encoder: LabelEncoder | None = None

    def fit(self, fold: FoldDesignMatrix) -> Self:
        """Fit a LightGBM booster on fold training data."""
        features = self._record_fit_context(fold)
        model_params, control = _split_params(self.params)
        model_params = _with_task_defaults(model_params, self.task, fold.train[self._target_col])

        x_train = fold.train.loc[:, features]
        x_val = fold.val.loc[:, features]
        y_train = fold.train[self._target_col]
        y_val = fold.val[self._target_col]

        if self.task is Task.CLASSIFICATION:
            self._label_encoder = LabelEncoder()
            y_train_values = self._label_encoder.fit_transform(y_train)
            y_val_values = self._label_encoder.transform(y_val)
            self.class_order_ = list(self._label_encoder.classes_)
            if len(self.class_order_) > BINARY_CLASS_COUNT:
                model_params.setdefault("num_class", len(self.class_order_))
        else:
            y_train_values = y_train.to_numpy()
            y_val_values = y_val.to_numpy()
            self.class_order_ = None

        train_weight = _weight_values(fold.train, self._weight_col)
        val_weight = _weight_values(fold.val, self._weight_col)
        train_set = lgb.Dataset(x_train, label=y_train_values, weight=train_weight, feature_name=features)
        val_set = lgb.Dataset(x_val, label=y_val_values, weight=val_weight, reference=train_set, feature_name=features)

        callbacks = _callbacks(control)
        feval = _balanced_accuracy_eval if control["use_balanced_accuracy_eval"] else None
        if control["balanced_accuracy_metric_only"]:
            model_params["metric"] = "None"
        if control["diagnostic_metric"] and "metric" not in model_params:
            model_params["metric"] = control["diagnostic_metric"]

        self.booster_ = lgb.train(
            model_params,
            train_set,
            num_boost_round=control["num_boost_round"],
            valid_sets=[val_set],
            valid_names=["valid"],
            feval=feval,
            callbacks=callbacks,
        )
        self.is_fit = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a prediction frame for a model-ready dataframe."""
        self._require_fit()
        if self.booster_ is None:
            msg = f"{self.name} has no fitted LightGBM booster."
            raise RuntimeError(msg)
        x_pred = self._select_recorded_features(df)
        predictions = self.booster_.predict(x_pred, num_iteration=self.booster_.best_iteration or None)
        if self.task is Task.CLASSIFICATION:
            return self._classification_prediction_frame(df, predictions)
        return self._regression_prediction_frame(df, predictions)

    def feature_importance(self) -> pd.DataFrame:
        """Return standardized native LightGBM gain importance values."""
        self._require_fit()
        if self.booster_ is None:
            msg = f"{self.name} has no fitted LightGBM booster."
            raise RuntimeError(msg)
        features = list(self.booster_.feature_name())
        gains = self.booster_.feature_importance(importance_type="gain")
        return self._feature_importance_frame(features, gains, importance_type="gain")


def _split_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_params = {key: value for key, value in params.items() if key not in _TRAINING_CONTROL_KEYS}
    control = {
        "balanced_accuracy_metric_only": bool(params.get("balanced_accuracy_metric_only", False)),
        "diagnostic_metric": params.get("diagnostic_metric"),
        "early_stopping_rounds": params.get("early_stopping_rounds"),
        "log_evaluation_period": params.get("log_evaluation_period", 0),
        "num_boost_round": int(params.get("num_boost_round", params.get("n_iter", 100))),
        "use_balanced_accuracy_eval": bool(params.get("use_balanced_accuracy_eval", False)),
    }
    return model_params, control


def _with_task_defaults(params: dict[str, Any], task: Task, y_train: pd.Series) -> dict[str, Any]:
    model_params = dict(params)
    model_params.setdefault("verbosity", -1)
    if task is Task.CLASSIFICATION:
        n_classes = int(y_train.nunique())
        model_params.setdefault("objective", "binary" if n_classes == BINARY_CLASS_COUNT else "multiclass")
        model_params.setdefault("metric", "binary_logloss" if n_classes == BINARY_CLASS_COUNT else "multi_logloss")
    else:
        model_params.setdefault("objective", "regression")
        model_params.setdefault("metric", "rmse")
    return model_params


def _callbacks(control: dict[str, Any]) -> list[Any]:
    callbacks: list[Any] = []
    early_stopping_rounds = control["early_stopping_rounds"]
    if early_stopping_rounds is not None:
        callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))

    log_period = int(control["log_evaluation_period"] or 0)
    if log_period > 0:
        callbacks.append(lgb.log_evaluation(period=log_period))
    return callbacks


def _weight_values(df: pd.DataFrame, weight_col: str | None) -> np.ndarray[Any, Any] | None:
    if weight_col is None:
        return None
    if weight_col not in df.columns:
        msg = f"Weight column {weight_col!r} is missing from design matrix."
        raise ValueError(msg)
    return df[weight_col].to_numpy()


def _balanced_accuracy_eval(preds: np.ndarray[Any, Any], data: lgb.Dataset) -> tuple[str, float, bool]:
    labels = data.get_label()
    if labels is None:
        msg = "LightGBM evaluation data is missing labels."
        raise ValueError(msg)
    predicted_labels = (preds >= PROBABILITY_THRESHOLD).astype(int) if preds.ndim == 1 else np.argmax(preds, axis=1)
    score = balanced_accuracy_score(labels, predicted_labels)
    return "balanced_accuracy", float(score), True
