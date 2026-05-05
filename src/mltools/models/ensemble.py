"""Utilities for validating predictions and building stack matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mltools.data.schema import DatasetSchema


def validate_prediction_frame(
    preds: pd.DataFrame,
    *,
    id_col: str,
    output_cols: Sequence[str],
    context: str = "prediction frame",
) -> None:
    """Validate the common prediction-frame contract.

    Parameters
    ----------
    preds
        Prediction frame to validate.
    id_col
        Name of the id column.
    output_cols
        Model output columns that must be finite.
    context
        Human-readable context included in validation errors.
    """
    required_cols = [id_col, *output_cols]
    missing_cols = [col for col in required_cols if col not in preds.columns]
    if missing_cols:
        msg = f"{context} is missing required columns: {missing_cols}."
        raise ValueError(msg)

    if preds[id_col].duplicated().any():
        duplicate_ids = preds.loc[preds[id_col].duplicated(keep=False), id_col].unique().tolist()
        msg = f"{context} contains duplicate ids in {id_col}: {duplicate_ids}."
        raise ValueError(msg)

    if not output_cols:
        msg = f"{context} must include at least one output column."
        raise ValueError(msg)

    outputs = preds.loc[:, list(output_cols)]
    try:
        output_values = outputs.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        msg = f"{context} contains non-numeric output values in columns: {list(output_cols)}."
        raise ValueError(msg) from exc

    if not np.isfinite(output_values).all():
        msg = f"{context} contains non-finite output values in columns: {list(output_cols)}."
        raise ValueError(msg)


def stack_feature_names(model_name: str, score_cols: Sequence[str]) -> list[str]:
    """Return level-2 feature names for a base model's output columns.

    Parameters
    ----------
    model_name
        Base model name used as the stack-feature namespace.
    score_cols
        Prediction output columns in the order they should appear in the stack matrix.
    """
    if not model_name:
        msg = "model_name must be a non-empty string."
        raise ValueError(msg)
    if not score_cols:
        msg = "score_cols must include at least one output column."
        raise ValueError(msg)

    if list(score_cols) == ["score_1"]:
        return [model_name]
    return [f"{model_name}_{score_col}" for score_col in score_cols]


def build_oof_stack_matrix(
    *,
    schema: DatasetSchema,
    labels: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    output_cols_by_model: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Build a supervised out-of-fold level-2 stack matrix.

    Parameters
    ----------
    schema
        Dataset schema with id and target column names.
    labels
        Label frame containing exactly the ids and targets for OOF training.
    predictions
        Mapping from base model name to OOF prediction frame.
    output_cols_by_model
        Optional explicit output columns for each base model.
    """
    _validate_labels(labels, schema=schema)
    if not predictions:
        msg = "predictions must include at least one base model."
        raise ValueError(msg)

    output_cols = _resolve_output_cols_by_model(
        predictions=predictions,
        output_cols_by_model=output_cols_by_model,
        id_col=schema.id_col,
        target_col=schema.target_col,
    )
    _validate_stack_feature_collisions(output_cols)

    stack_matrix = labels.loc[:, [schema.id_col, schema.target_col]].copy()
    expected_ids = labels[schema.id_col]

    for model_name, prediction_frame in predictions.items():
        model_output_cols = output_cols[model_name]
        validate_prediction_frame(
            prediction_frame,
            id_col=schema.id_col,
            output_cols=model_output_cols,
            context=f"{model_name} OOF predictions",
        )
        _raise_for_id_mismatch(
            actual_ids=prediction_frame[schema.id_col],
            expected_ids=expected_ids,
            context=f"{model_name} OOF predictions",
        )
        aligned_predictions = _align_prediction_frame(
            prediction_frame=prediction_frame,
            id_col=schema.id_col,
            output_cols=model_output_cols,
            ordered_ids=expected_ids,
        )
        feature_names = stack_feature_names(model_name, model_output_cols)
        for output_col, feature_name in zip(model_output_cols, feature_names, strict=True):
            stack_matrix[feature_name] = aligned_predictions[output_col].to_numpy()

    return stack_matrix


def average_fold_predictions(
    *,
    schema: DatasetSchema,
    prediction_frames: Sequence[pd.DataFrame],
    output_cols: Sequence[str],
) -> pd.DataFrame:
    """Average fold prediction frames by id.

    Parameters
    ----------
    schema
        Dataset schema with the id column name.
    prediction_frames
        Fold prediction frames to average.
    output_cols
        Output columns to average.
    """
    if not prediction_frames:
        msg = "prediction_frames must include at least one fold prediction frame."
        raise ValueError(msg)
    if not output_cols:
        msg = "output_cols must include at least one output column."
        raise ValueError(msg)

    first_frame = prediction_frames[0]
    validate_prediction_frame(first_frame, id_col=schema.id_col, output_cols=output_cols, context="fold 0 predictions")

    ordered_ids = first_frame[schema.id_col]
    totals = _align_prediction_frame(
        prediction_frame=first_frame,
        id_col=schema.id_col,
        output_cols=output_cols,
        ordered_ids=ordered_ids,
    ).loc[:, list(output_cols)]

    for fold_index, prediction_frame in enumerate(prediction_frames[1:], start=1):
        context = f"fold {fold_index} predictions"
        validate_prediction_frame(prediction_frame, id_col=schema.id_col, output_cols=output_cols, context=context)
        _raise_for_id_mismatch(actual_ids=prediction_frame[schema.id_col], expected_ids=ordered_ids, context=context)
        aligned_predictions = _align_prediction_frame(
            prediction_frame=prediction_frame,
            id_col=schema.id_col,
            output_cols=output_cols,
            ordered_ids=ordered_ids,
        )
        totals += aligned_predictions.loc[:, list(output_cols)]

    averaged = totals / len(prediction_frames)
    return pd.concat([ordered_ids.reset_index(drop=True), averaged.reset_index(drop=True)], axis=1)


def build_serving_stack_matrix(
    *,
    schema: DatasetSchema,
    base_predictions: Mapping[str, pd.DataFrame],
    output_cols_by_model: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Build an unsupervised serving level-2 stack matrix.

    Parameters
    ----------
    schema
        Dataset schema with id and target column names.
    base_predictions
        Mapping from base model name to serving prediction frame.
    output_cols_by_model
        Optional explicit output columns for each base model.
    """
    if not base_predictions:
        msg = "base_predictions must include at least one base model."
        raise ValueError(msg)

    output_cols = _resolve_output_cols_by_model(
        predictions=base_predictions,
        output_cols_by_model=output_cols_by_model,
        id_col=schema.id_col,
        target_col=schema.target_col,
    )
    _validate_stack_feature_collisions(output_cols)

    first_model_name = next(iter(base_predictions))
    first_frame = base_predictions[first_model_name]
    first_output_cols = output_cols[first_model_name]
    validate_prediction_frame(
        first_frame,
        id_col=schema.id_col,
        output_cols=first_output_cols,
        context=f"{first_model_name} serving predictions",
    )
    ordered_ids = first_frame[schema.id_col]
    stack_matrix = ordered_ids.reset_index(drop=True).to_frame()

    for model_name, prediction_frame in base_predictions.items():
        model_output_cols = output_cols[model_name]
        context = f"{model_name} serving predictions"
        validate_prediction_frame(
            prediction_frame,
            id_col=schema.id_col,
            output_cols=model_output_cols,
            context=context,
        )
        _raise_for_id_mismatch(actual_ids=prediction_frame[schema.id_col], expected_ids=ordered_ids, context=context)
        aligned_predictions = _align_prediction_frame(
            prediction_frame=prediction_frame,
            id_col=schema.id_col,
            output_cols=model_output_cols,
            ordered_ids=ordered_ids,
        )
        feature_names = stack_feature_names(model_name, model_output_cols)
        for output_col, feature_name in zip(model_output_cols, feature_names, strict=True):
            stack_matrix[feature_name] = aligned_predictions[output_col].to_numpy()

    return stack_matrix


class StackingEnsemble(BaseModel):
    """Lineage metadata for a level-2 stack matrix."""

    name: str
    base_model_names: list[str]
    stack_feature_columns: list[str]
    class_order_by_model: dict[str, list[Any]] = Field(default_factory=dict)


def _validate_labels(labels: pd.DataFrame, *, schema: DatasetSchema) -> None:
    """Validate the id and target columns required for supervised stacking."""
    missing_cols = [col for col in [schema.id_col, schema.target_col] if col not in labels.columns]
    if missing_cols:
        msg = f"labels is missing required columns: {missing_cols}."
        raise ValueError(msg)

    if labels[schema.id_col].duplicated().any():
        duplicate_ids = labels.loc[labels[schema.id_col].duplicated(keep=False), schema.id_col].unique().tolist()
        msg = f"labels contains duplicate ids in {schema.id_col}: {duplicate_ids}."
        raise ValueError(msg)


def _resolve_output_cols_by_model(
    *,
    predictions: Mapping[str, pd.DataFrame],
    output_cols_by_model: Mapping[str, Sequence[str]] | None,
    id_col: str,
    target_col: str,
) -> dict[str, list[str]]:
    """Return explicit or inferred output columns for every model."""
    if output_cols_by_model is None:
        return {
            model_name: [col for col in prediction_frame.columns if col not in {id_col, target_col}]
            for model_name, prediction_frame in predictions.items()
        }

    missing_models = [model_name for model_name in predictions if model_name not in output_cols_by_model]
    if missing_models:
        msg = f"output_cols_by_model is missing entries for models: {missing_models}."
        raise ValueError(msg)

    extra_models = [model_name for model_name in output_cols_by_model if model_name not in predictions]
    if extra_models:
        msg = f"output_cols_by_model contains unknown models: {extra_models}."
        raise ValueError(msg)

    return {model_name: list(output_cols_by_model[model_name]) for model_name in predictions}


def _validate_stack_feature_collisions(output_cols_by_model: Mapping[str, Sequence[str]]) -> None:
    """Raise when stack feature naming would produce duplicate columns."""
    feature_names = [
        feature_name
        for model_name, output_cols in output_cols_by_model.items()
        for feature_name in stack_feature_names(model_name, output_cols)
    ]
    if len(feature_names) != len(set(feature_names)):
        duplicate_features = sorted(
            {feature_name for feature_name in feature_names if feature_names.count(feature_name) > 1},
        )
        msg = f"Stack feature names must be unique, found duplicates: {duplicate_features}."
        raise ValueError(msg)


def _align_prediction_frame(
    *,
    prediction_frame: pd.DataFrame,
    id_col: str,
    output_cols: Sequence[str],
    ordered_ids: pd.Series,
) -> pd.DataFrame:
    """Return prediction outputs aligned to the requested id order."""
    return (
        prediction_frame.loc[:, [id_col, *output_cols]]
        .set_index(id_col)
        .loc[ordered_ids.tolist(), list(output_cols)]
        .reset_index(drop=True)
    )


def _raise_for_id_mismatch(*, actual_ids: pd.Series, expected_ids: pd.Series, context: str) -> None:
    """Raise when two prediction id sets do not match exactly."""
    actual_set = set(actual_ids.tolist())
    expected_set = set(expected_ids.tolist())

    missing_ids = sorted(expected_set - actual_set)
    extra_ids = sorted(actual_set - expected_set)
    if missing_ids or extra_ids:
        msg = f"{context} id coverage does not match expected ids; missing ids: {missing_ids}; extra ids: {extra_ids}."
        raise ValueError(msg)
