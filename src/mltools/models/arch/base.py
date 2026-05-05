"""Shared contracts for mutable model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd

from mltools.data.schema import feature_columns as schema_feature_columns

BINARY_CLASS_COUNT = 2
TWO_DIMENSIONS = 2


class FoldDesignMatrix(Protocol):
    """Structural type for PRD 01 fold design matrices."""

    schema: Any
    train: pd.DataFrame
    val: pd.DataFrame


class Task(Enum):
    """Supported supervised learning tasks."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class BaseModelWrapper(ABC):
    """Mutable base class for one-fold model training and prediction."""

    name: str
    task: Task
    params: dict[str, Any]
    is_fit: bool
    feature_names_: list[str] | None
    class_order_: list[Any] | None

    def __init__(self, name: str, task: Task, params: dict[str, Any] | None = None) -> None:
        """Initialize common model wrapper state.

        Parameters
        ----------
        name
            Stable model name used in downstream metadata.
        task
            Supervised learning task implemented by the wrapper.
        params
            Backend parameters. Values are defensively copied.
        """
        if not name:
            msg = "Model name must be a non-empty string."
            raise ValueError(msg)
        if not isinstance(task, Task):
            msg = "task must be an instance of Task."
            raise TypeError(msg)
        self.name = name
        self.task = task
        self.params = dict(params or {})
        self.is_fit = False
        self.feature_names_ = None
        self.class_order_ = None
        self.schema_: Any | None = None

    @abstractmethod
    def fit(self, fold: FoldDesignMatrix) -> Self:
        """Fit the model on a fold design matrix."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict on a model-ready design matrix."""

    def feature_importance(self) -> pd.DataFrame:
        """Return standardized feature importance values.

        Raises
        ------
        NotImplementedError
            If the backend cannot provide feature importance.
        """
        msg = f"{self.__class__.__name__} does not support feature importance."
        raise NotImplementedError(msg)

    def _record_fit_context(self, fold: FoldDesignMatrix) -> list[str]:
        features = _feature_columns(fold.train, fold.schema)
        self.feature_names_ = features
        self.schema_ = fold.schema
        return features

    def _require_fit(self) -> None:
        if not self.is_fit:
            msg = f"{self.name} must be fit before calling predict or feature_importance."
            raise RuntimeError(msg)

    def _select_recorded_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fit()
        if self.feature_names_ is None:
            msg = f"{self.name} has no recorded feature columns."
            raise RuntimeError(msg)
        missing = [feature for feature in self.feature_names_ if feature not in df.columns]
        if missing:
            msg = f"Missing feature columns for {self.name}: {missing}."
            raise ValueError(msg)
        return df.loc[:, self.feature_names_]

    def _id_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fit()
        id_col = self._id_col
        if id_col not in df.columns:
            msg = f"Prediction input for {self.name} is missing id column {id_col!r}."
            raise ValueError(msg)
        return pd.DataFrame({id_col: df[id_col].to_numpy()})

    @property
    def _id_col(self) -> str:
        if self.schema_ is None:
            msg = f"{self.name} has no fitted DatasetSchema."
            raise RuntimeError(msg)
        return str(self.schema_.id_col)

    @property
    def _target_col(self) -> str:
        if self.schema_ is None:
            msg = f"{self.name} has no fitted DatasetSchema."
            raise RuntimeError(msg)
        return str(self.schema_.target_col)

    @property
    def _weight_col(self) -> str | None:
        if self.schema_ is None:
            msg = f"{self.name} has no fitted DatasetSchema."
            raise RuntimeError(msg)
        return self.schema_.weight_col

    def _classification_prediction_frame(self, df: pd.DataFrame, probabilities: Any) -> pd.DataFrame:
        frame = self._id_frame(df)
        class_order = self.class_order_
        if class_order is None:
            msg = f"{self.name} has no recorded class order."
            raise RuntimeError(msg)

        probs = np.asarray(probabilities)
        if len(class_order) == BINARY_CLASS_COUNT:
            if probs.ndim == 1:
                frame["score_1"] = probs
            else:
                frame["score_1"] = probs[:, 1]
            return frame

        if probs.ndim != TWO_DIMENSIONS or probs.shape[1] != len(class_order):
            msg = f"Expected probabilities with {len(class_order)} columns for {self.name}; got shape {probs.shape}."
            raise ValueError(msg)
        for class_index in range(len(class_order)):
            frame[f"score_{class_index}"] = probs[:, class_index]
        return frame

    def _regression_prediction_frame(self, df: pd.DataFrame, predictions: Any) -> pd.DataFrame:
        frame = self._id_frame(df)
        frame["prediction"] = np.asarray(predictions)
        return frame

    def _feature_importance_frame(
        self,
        features: list[str],
        importances: Any,
        *,
        importance_type: str,
        class_label: Any | None = None,
    ) -> pd.DataFrame:
        values = np.asarray(importances, dtype=float)
        if values.ndim != 1:
            msg = "Feature importance values must be one-dimensional."
            raise ValueError(msg)
        if len(features) != len(values):
            msg = f"Expected {len(features)} importance values for {self.name}; got {len(values)}."
            raise ValueError(msg)
        return pd.DataFrame(
            {
                "model_name": self.name,
                "feature": features,
                "importance": values,
                "importance_type": importance_type,
                "class_label": class_label,
            },
        )


def _feature_columns(df: pd.DataFrame, schema: Any) -> list[str]:
    return list(schema_feature_columns(df, schema))
