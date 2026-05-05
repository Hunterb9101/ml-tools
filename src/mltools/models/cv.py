"""Cross-validation training helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import pandas as pd
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mltools.data.contracts import FoldDesignMatrix
    from mltools.models.arch.base import BaseModelWrapper
else:
    FoldDesignMatrix: TypeAlias = Any
    BaseModelWrapper: TypeAlias = Any


class FoldTrainingResult(BaseModel):
    """Training output for one cross-validation fold."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fold_id: int
    model: BaseModelWrapper
    train_predictions: pd.DataFrame
    val_predictions: pd.DataFrame


class CVTrainingResult(BaseModel):
    """Training output for a full cross-validation run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fold_results: list[FoldTrainingResult]

    def oof_predictions(self) -> pd.DataFrame:
        """Concatenate validation predictions across folds.

        Returns
        -------
        pd.DataFrame
            Out-of-fold prediction frame.
        """
        frames = [fold_result.val_predictions for fold_result in self.fold_results]
        return _concat_prediction_frames(frames, validate_duplicate_ids=True)

    def train_predictions(self) -> pd.DataFrame:
        """Concatenate train predictions across folds.

        Returns
        -------
        pd.DataFrame
            Training prediction frame.
        """
        frames = [fold_result.train_predictions for fold_result in self.fold_results]
        return _concat_prediction_frames(frames, validate_duplicate_ids=False)

    def models(self) -> list[BaseModelWrapper]:
        """Return fitted fold model wrappers in fold order.

        Returns
        -------
        list[BaseModelWrapper]
            Fitted model wrappers.
        """
        return [fold_result.model for fold_result in self.fold_results]


def train_cv(
    *,
    model_factory: Callable[[], BaseModelWrapper],
    folds: Sequence[FoldDesignMatrix],
) -> CVTrainingResult:
    """Train a fresh model wrapper for each fold.

    Parameters
    ----------
    model_factory
        Callable that creates an unfitted model wrapper.
    folds
        Fold design matrices to train and predict in order.

    Returns
    -------
    CVTrainingResult
        Fitted fold models and their train/validation predictions.

    Raises
    ------
    ValueError
        If no folds are provided, fold ids are duplicated, or the factory
        reuses a model instance.
    """
    fold_list = list(folds)
    if len(fold_list) == 0:
        msg = "folds must contain at least one fold."
        raise ValueError(msg)

    fold_ids = [fold.fold_id for fold in fold_list]
    duplicated_fold_ids = _duplicates(fold_ids)
    if duplicated_fold_ids:
        msg = f"fold ids must be unique; duplicated fold ids: {duplicated_fold_ids}."
        raise ValueError(msg)

    fold_results: list[FoldTrainingResult] = []
    model_instance_ids: set[int] = set()
    for fold in fold_list:
        model = model_factory()
        model_instance_id = id(model)
        if model_instance_id in model_instance_ids:
            msg = "model_factory must return a fresh model instance for each fold."
            raise ValueError(msg)
        model_instance_ids.add(model_instance_id)

        fitted_model = model.fit(fold)
        train_predictions = fitted_model.predict(fold.train)
        val_predictions = fitted_model.predict(fold.val)
        fold_results.append(
            FoldTrainingResult(
                fold_id=fold.fold_id,
                model=fitted_model,
                train_predictions=train_predictions,
                val_predictions=val_predictions,
            ),
        )

    return CVTrainingResult(fold_results=fold_results)


def _concat_prediction_frames(frames: Sequence[pd.DataFrame], *, validate_duplicate_ids: bool) -> pd.DataFrame:
    """Concatenate prediction frames and optionally validate inferred ids."""
    if len(frames) == 0:
        return pd.DataFrame()

    result = pd.concat(frames, copy=False)
    if validate_duplicate_ids:
        id_col = _infer_id_column(result)
        if id_col is not None and result[id_col].duplicated().any():
            msg = f"prediction frame contains duplicate ids in column {id_col!r}."
            raise ValueError(msg)
    return result


def _infer_id_column(df: pd.DataFrame) -> str | None:
    """Infer a prediction-frame id column when it is unambiguous."""
    candidate_columns = [
        column
        for column in df.columns
        if column != "prediction" and not column.startswith("score_")
    ]
    if len(candidate_columns) != 1:
        return None
    return candidate_columns[0]


def _duplicates(values: Sequence[int]) -> list[int]:
    """Return duplicated integer values in first duplicate order."""
    seen: set[int] = set()
    duplicates: list[int] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates
