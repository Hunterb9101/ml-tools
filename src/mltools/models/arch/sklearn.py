"""Mutable wrappers for already-constructed sklearn estimators."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd

from mltools.models.arch.base import BaseModelWrapper, FoldDesignMatrix, Task


class SklearnModel(BaseModelWrapper):
    """Wrap an existing sklearn estimator or pipeline."""

    def __init__(
        self,
        name: str,
        task: Task,
        estimator: Any,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a sklearn model wrapper.

        Parameters
        ----------
        name
            Stable model name used in downstream metadata.
        task
            Supervised learning task.
        estimator
            Already-constructed sklearn estimator or pipeline.
        params
            Optional metadata parameters copied onto the wrapper.
        """
        super().__init__(name=name, task=task, params=params)
        self.estimator = estimator

    def fit(self, fold: FoldDesignMatrix) -> Self:
        """Fit the wrapped estimator on fold training data."""
        features = self._record_fit_context(fold)
        x_train = fold.train.loc[:, features]
        y_train = fold.train[self._target_col]
        self.estimator.fit(x_train, y_train)

        if self.task is Task.CLASSIFICATION:
            classes = getattr(self.estimator, "classes_", None)
            if classes is None:
                classes = getattr(_final_estimator(self.estimator), "classes_", None)
            if classes is None:
                msg = f"{self.name} fitted classifier does not expose classes_."
                raise RuntimeError(msg)
            self.class_order_ = list(classes)
        else:
            self.class_order_ = None

        self.is_fit = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a prediction frame for a model-ready dataframe."""
        x_pred = self._select_recorded_features(df)
        if self.task is Task.CLASSIFICATION:
            if not hasattr(self.estimator, "predict_proba"):
                msg = f"{self.name} requires predict_proba for classification predictions."
                raise NotImplementedError(msg)
            probabilities = self.estimator.predict_proba(x_pred)
            return self._classification_prediction_frame(df, probabilities)

        predictions = self.estimator.predict(x_pred)
        return self._regression_prediction_frame(df, predictions)

    def feature_importance(self) -> pd.DataFrame:
        """Return standardized sklearn feature importance values."""
        self._require_fit()
        if self.feature_names_ is None:
            msg = f"{self.name} has no recorded feature columns."
            raise RuntimeError(msg)

        estimator = _final_estimator(self.estimator)
        if hasattr(estimator, "feature_importances_"):
            return self._feature_importance_frame(
                self.feature_names_,
                estimator.feature_importances_,
                importance_type="feature_importances_",
            )

        if hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_, dtype=float)
            if coef.ndim == 1:
                return self._feature_importance_frame(
                    self.feature_names_,
                    np.abs(coef),
                    importance_type="abs_coef_",
                )

            if coef.shape[0] == 1:
                class_label = self.class_order_[1] if self.class_order_ and len(self.class_order_) > 1 else None
                return self._feature_importance_frame(
                    self.feature_names_,
                    np.abs(coef[0]),
                    importance_type="abs_coef_",
                    class_label=class_label,
                )

            frames = []
            class_order = self.class_order_ or list(range(coef.shape[0]))
            for class_index, values in enumerate(coef):
                frames.append(
                    self._feature_importance_frame(
                        self.feature_names_,
                        np.abs(values),
                        importance_type="abs_coef_",
                        class_label=class_order[class_index],
                    ),
                )
            return pd.concat(frames, ignore_index=True)

        msg = f"{self.name} does not expose feature_importances_ or coef_."
        raise NotImplementedError(msg)


def _final_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "steps") and estimator.steps:
        return estimator.steps[-1][1]
    return estimator
