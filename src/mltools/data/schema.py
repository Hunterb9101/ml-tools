"""Reusable data contracts for model-ready tabular datasets."""

import warnings
from collections.abc import ItemsView, KeysView, ValuesView
from typing import Any

import pandas as pd
import pydantic as pdt


class DatasetSchema(pdt.BaseModel):
    """Describe the required columns in a model-ready dataset."""

    id_col: str
    target_col: str
    weight_col: str | None = None

    @pdt.field_validator("id_col", "target_col")
    @classmethod
    def _validate_required_name(cls, value: str) -> str:
        """Validate required column names."""
        if not value.strip():
            msg = "Column names must be non-empty strings."
            raise ValueError(msg)
        return value

    @pdt.field_validator("weight_col")
    @classmethod
    def _validate_optional_name(cls, value: str | None) -> str | None:
        """Validate optional column names."""
        if value is not None and not value.strip():
            msg = "Column names must be non-empty strings."
            raise ValueError(msg)
        return value


class FittedTransformerSet(pdt.BaseModel):
    """Container for named fitted, sklearn-like transformers."""

    model_config = pdt.ConfigDict(arbitrary_types_allowed=True)

    transformers: dict[str, Any] = pdt.Field(default_factory=dict)

    def __getitem__(self, name: str) -> Any:
        """Return a transformer by name."""
        return self.transformers[name]

    def __contains__(self, name: str) -> bool:
        """Return whether a transformer name exists."""
        return name in self.transformers

    def __len__(self) -> int:
        """Return the number of named transformers."""
        return len(self.transformers)

    def get(self, name: str, default: Any = None) -> Any:
        """Return a transformer by name, or a default value."""
        return self.transformers.get(name, default)

    def items(self) -> ItemsView[str, Any]:
        """Return transformer name and value pairs."""
        return self.transformers.items()

    def keys(self) -> KeysView[str]:
        """Return transformer names."""
        return self.transformers.keys()

    def values(self) -> ValuesView[Any]:
        """Return transformer values."""
        return self.transformers.values()

# Pydantic warns because the required public field name `schema` shadows
# BaseModel.schema(). The PRD contract uses `fold.schema`, so the class-level
# warning is intentionally suppressed at definition time.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='Field name "schema".*', category=UserWarning)

    class FoldDesignMatrix(pdt.BaseModel):
        """Store model-ready training, validation, and optional holdout frames."""

        model_config = pdt.ConfigDict(arbitrary_types_allowed=True)

        fold_id: int = pdt.Field(ge=0)
        schema: DatasetSchema  # type: ignore[assignment]
        train: pd.DataFrame
        val: pd.DataFrame
        fitted: FittedTransformerSet
        holdout: pd.DataFrame | None = None

        @pdt.model_validator(mode="after")
        def _validate_frames(self) -> "FoldDesignMatrix":
            """Validate that frames contain columns required by the schema."""
            validate_required_columns(self.train, self.schema, frame_name="train")
            validate_required_columns(self.val, self.schema, frame_name="val")
            if self.holdout is not None:
                validate_required_columns(self.holdout, self.schema, frame_name="holdout", require_target=False)
            return self


def schema_columns(schema: DatasetSchema, *, require_target: bool = True, require_weight: bool = True) -> list[str]:
    """Return schema columns that should be present in a dataframe.

    Parameters
    ----------
    schema
        Dataset schema that defines id, target, and optional weight columns.
    require_target
        Whether to include the target column.
    require_weight
        Whether to include the configured weight column.

    Returns
    -------
    list[str]
        Required schema columns in contract order.
    """
    columns = [schema.id_col]
    if require_target:
        columns.append(schema.target_col)
    if require_weight and schema.weight_col is not None:
        columns.append(schema.weight_col)
    return columns


def missing_schema_columns(
    df: pd.DataFrame,
    schema: DatasetSchema,
    *,
    require_target: bool = True,
    require_weight: bool = True,
) -> list[str]:
    """Return schema columns missing from a dataframe.

    Parameters
    ----------
    df
        Dataframe to validate.
    schema
        Dataset schema that defines id, target, and optional weight columns.
    require_target
        Whether to require the target column.
    require_weight
        Whether to require the configured weight column.

    Returns
    -------
    list[str]
        Missing required schema columns.
    """
    columns = schema_columns(schema, require_target=require_target, require_weight=require_weight)
    return [col for col in columns if col not in df]


def validate_required_columns(
    df: pd.DataFrame,
    schema: DatasetSchema,
    *,
    frame_name: str = "dataframe",
    require_target: bool = True,
    require_weight: bool = True,
) -> None:
    """Raise if a dataframe does not contain required schema columns.

    Parameters
    ----------
    df
        Dataframe to validate.
    schema
        Dataset schema that defines id, target, and optional weight columns.
    frame_name
        Human-readable dataframe name for error messages.
    require_target
        Whether to require the target column.
    require_weight
        Whether to require the configured weight column.
    """
    missing = missing_schema_columns(
        df,
        schema,
        require_target=require_target,
        require_weight=require_weight,
    )
    if missing:
        msg = f"{frame_name} is missing required schema columns: {missing}"
        raise ValueError(msg)


def feature_columns(df: pd.DataFrame, schema: DatasetSchema) -> list[str]:
    """Return model feature columns in dataframe order.

    Parameters
    ----------
    df
        Model-ready dataframe.
    schema
        Dataset schema that identifies non-feature columns.

    Returns
    -------
    list[str]
        Feature columns in the same order as the dataframe.
    """
    validate_required_columns(df, schema)
    excluded = set(schema_columns(schema))
    return [col for col in df.columns if col not in excluded]
