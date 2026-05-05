import pandas as pd
import pydantic as pdt
import pytest

from mltools.data.schema import (
    DatasetSchema,
    FittedTransformerSet,
    FoldDesignMatrix,
    feature_columns,
    missing_schema_columns,
)


def test_dataset_schema_accepts_valid_columns():
    schema = DatasetSchema(id_col="id", target_col="target", weight_col="weight")

    assert schema.id_col == "id"
    assert schema.target_col == "target"
    assert schema.weight_col == "weight"


@pytest.mark.parametrize(
    "schema_kwargs",
    [
        {"id_col": "", "target_col": "target"},
        {"id_col": "   ", "target_col": "target"},
        {"id_col": "id", "target_col": ""},
        {"id_col": "id", "target_col": "target", "weight_col": ""},
    ],
)
def test_dataset_schema_rejects_empty_column_names(schema_kwargs):
    with pytest.raises(pdt.ValidationError, match="Column names must be non-empty strings"):
        DatasetSchema(**schema_kwargs)


def test_missing_schema_columns_reports_required_columns():
    schema = DatasetSchema(id_col="id", target_col="target", weight_col="weight")
    df = pd.DataFrame({"id": [1], "feature": [0.5]})

    assert missing_schema_columns(df, schema) == ["target", "weight"]


def test_feature_columns_preserve_order_without_weight():
    schema = DatasetSchema(id_col="id", target_col="target")
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "feature_b": [3.0, 4.0],
            "target": [0, 1],
            "feature_a": [5.0, 6.0],
        },
    )

    assert feature_columns(df, schema) == ["feature_b", "feature_a"]


def test_feature_columns_exclude_weight_col():
    schema = DatasetSchema(id_col="id", target_col="target", weight_col="weight")
    df = pd.DataFrame(
        {
            "weight": [1.0, 0.5],
            "feature_b": [3.0, 4.0],
            "id": [1, 2],
            "target": [0, 1],
            "feature_a": [5.0, 6.0],
        },
    )

    assert feature_columns(df, schema) == ["feature_b", "feature_a"]


def test_feature_columns_raise_for_missing_required_columns():
    schema = DatasetSchema(id_col="id", target_col="target", weight_col="weight")
    df = pd.DataFrame({"id": [1], "feature": [0.5]})

    with pytest.raises(ValueError, match=r"missing required schema columns: \['target', 'weight'\]"):
        feature_columns(df, schema)


def test_fitted_transformer_set_supports_named_lookup_and_items():
    fitted = FittedTransformerSet(transformers={"scale": object(), "encode": object()})

    assert "scale" in fitted
    assert fitted["scale"] is fitted.transformers["scale"]
    assert list(fitted.keys()) == ["scale", "encode"]
    assert list(fitted.items()) == list(fitted.transformers.items())


def test_fold_design_matrix_validates_train_and_val_columns():
    schema = DatasetSchema(id_col="id", target_col="target")
    train = pd.DataFrame({"id": [1], "target": [0], "feature": [0.5]})
    val = pd.DataFrame({"id": [2], "feature": [0.6]})

    with pytest.raises(pdt.ValidationError, match="val is missing required schema columns"):
        FoldDesignMatrix(
            fold_id=0,
            schema=schema,
            train=train,
            val=val,
            fitted=FittedTransformerSet(),
        )


def test_fold_design_matrix_allows_holdout_without_target():
    schema = DatasetSchema(id_col="id", target_col="target")
    train = pd.DataFrame({"id": [1], "target": [0], "feature": [0.5]})
    val = pd.DataFrame({"id": [2], "target": [1], "feature": [0.6]})
    holdout = pd.DataFrame({"id": [3], "feature": [0.7]})

    matrix = FoldDesignMatrix(
        fold_id=0,
        schema=schema,
        train=train,
        val=val,
        holdout=holdout,
        fitted=FittedTransformerSet(),
    )

    assert matrix.holdout is holdout
