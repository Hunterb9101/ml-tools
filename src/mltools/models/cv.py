"""Cross-validation training helpers."""

from __future__ import annotations

from collections import Counter
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
    id_col: str | None = None

    def oof_predictions(self) -> pd.DataFrame:
        """Concatenate validation predictions across folds.

        Returns
        -------
        pd.DataFrame
            Out-of-fold prediction frame.
        """
        frames = [fold_result.val_predictions for fold_result in self.fold_results]
        return _concat_prediction_frames(frames, id_col=self.id_col, validate_duplicate_ids=True)

    def train_predictions(self) -> pd.DataFrame:
        """Concatenate train predictions across folds.

        Returns
        -------
        pd.DataFrame
            Training prediction frame.
        """
        frames = [fold_result.train_predictions for fold_result in self.fold_results]
        return _concat_prediction_frames(frames, id_col=self.id_col, validate_duplicate_ids=False)

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

    return CVTrainingResult(fold_results=fold_results, id_col=_common_id_col(fold_list))


def _concat_prediction_frames(
    frames: Sequence[pd.DataFrame],
    *,
    id_col: str | None,
    validate_duplicate_ids: bool,
) -> pd.DataFrame:
    """Concatenate prediction frames and optionally validate duplicate ids."""
    if len(frames) == 0:
        return pd.DataFrame()

    result = pd.concat(frames, copy=False)
    if validate_duplicate_ids and id_col is not None:
        if id_col not in result.columns:
            msg = f"prediction frame is missing id column {id_col!r}."
            raise ValueError(msg)
        if result[id_col].duplicated().any():
            msg = f"prediction frame contains duplicate ids in column {id_col!r}."
            raise ValueError(msg)
    return result


def _common_id_col(folds: Sequence[FoldDesignMatrix]) -> str | None:
    """Return the common schema id column when every fold exposes one."""
    raw_id_cols = [getattr(getattr(fold, "schema", None), "id_col", None) for fold in folds]
    if any(id_col is None for id_col in raw_id_cols):
        return None
    id_cols = [str(id_col) for id_col in raw_id_cols]
    unique_id_cols = set(id_cols)
    if len(unique_id_cols) > 1:
        msg = f"fold schemas must use the same id column, got: {sorted(unique_id_cols)}."
        raise ValueError(msg)
    return id_cols[0]


def _duplicates(values: Sequence[int]) -> list[int]:
    """Return duplicated integer values in first occurrence order."""
    counts = Counter(values)
    return [value for value in dict.fromkeys(values) if counts[value] > 1]
