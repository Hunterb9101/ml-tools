"""Optional path layout helpers for ML artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pydantic as pdt


class ArtifactLayout(pdt.BaseModel):
    """Return standard artifact paths under a caller-supplied root."""

    root: Path

    def data_dir(self) -> Path:
        """Return the top-level data artifact directory."""
        return self.root / "data"

    def models_dir(self) -> Path:
        """Return the top-level model artifact directory."""
        return self.root / "models"

    def stacks_dir(self) -> Path:
        """Return the top-level stack artifact directory."""
        return self.root / "stacks"

    def model_path(self, *, model_name: str, fold_id: int = 0) -> Path:
        """Return a fitted model artifact path."""
        return self._model_fold_dir(model_name=model_name, fold_id=fold_id) / "mdl.pkl"

    def transformer_path(self, *, fold_id: int = 0, name: str) -> Path:
        """Return a fitted transformer artifact path."""
        transformer_name = _validate_path_segment(name, field_name="name")
        return self.models_dir() / f"fold_{fold_id}" / "transformers" / f"{transformer_name}.pkl"

    def dmatrix_path(self, *, split: str, fold_id: int = 0) -> Path:
        """Return a design matrix artifact path."""
        split_name = _validate_path_segment(split, field_name="split")
        return self.models_dir() / f"fold_{fold_id}" / "dmatrix" / f"{split_name}.parquet"

    def prediction_path(self, *, model_name: str, split: str, fold_id: int = 0) -> Path:
        """Return a model prediction artifact path."""
        split_name = _validate_path_segment(split, field_name="split")
        return self._model_fold_dir(model_name=model_name, fold_id=fold_id) / "preds" / f"{split_name}.pkl"

    def stack_matrix_path(self, *, ensemble_name: str, split: str) -> Path:
        """Return a stack matrix artifact path."""
        split_name = _validate_path_segment(split, field_name="split")
        return self._stack_dir(ensemble_name=ensemble_name) / "dmatrix" / f"{split_name}.parquet"

    def stack_model_path(self, *, ensemble_name: str) -> Path:
        """Return a stack model artifact path."""
        return self._stack_dir(ensemble_name=ensemble_name) / "mdl.pkl"

    def metrics_path(self, *, name: str) -> Path:
        """Return a metrics artifact path."""
        metrics_name = _validate_path_segment(name, field_name="name")
        return self.models_dir() / f"{metrics_name}.json"

    def _model_fold_dir(self, *, model_name: str, fold_id: int) -> Path:
        """Return the model-specific fold directory."""
        safe_model_name = _validate_path_segment(model_name, field_name="model_name")
        return self.models_dir() / safe_model_name / f"fold_{fold_id}"

    def _stack_dir(self, *, ensemble_name: str) -> Path:
        """Return the ensemble-specific stack directory."""
        safe_ensemble_name = _validate_path_segment(ensemble_name, field_name="ensemble_name")
        return self.stacks_dir() / safe_ensemble_name


def _validate_path_segment(value: str, *, field_name: str) -> str:
    """Validate a caller-supplied path segment."""
    if value != value.strip() or not value:
        msg = f"{field_name} must be a non-empty path segment without leading or trailing whitespace."
        raise ValueError(msg)
    if value in {".", ".."}:
        msg = f"{field_name} must not be a relative path segment: {value!r}."
        raise ValueError(msg)
    if os.sep in value or (os.altsep is not None and os.altsep in value):
        msg = f"{field_name} must not contain path separators: {value!r}."
        raise ValueError(msg)

    value_path = Path(value)
    if value_path.is_absolute() or len(value_path.parts) != 1:
        msg = f"{field_name} must be a single path segment: {value!r}."
        raise ValueError(msg)

    return value
