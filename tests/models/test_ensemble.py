from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from mltools.models.ensemble import (
    StackingEnsemble,
    average_fold_predictions,
    build_oof_stack_matrix,
    build_serving_stack_matrix,
    stack_feature_names,
    validate_prediction_frame,
)


@dataclass(frozen=True)
class DatasetSchema:
    id_col: str = "id"
    target_col: str = "target"


def test_validate_prediction_frame_accepts_id_and_outputs_without_target():
    preds = pd.DataFrame({"id": [1, 2], "score_1": [0.1, 0.9]})

    validate_prediction_frame(preds, id_col="id", output_cols=["score_1"])


@pytest.mark.parametrize(
    ("preds", "match"),
    [
        (pd.DataFrame({"score_1": [0.1]}), "missing required columns"),
        (pd.DataFrame({"id": [1, 1], "score_1": [0.1, 0.2]}), "duplicate ids"),
        (pd.DataFrame({"id": [1]}), "missing required columns"),
        (pd.DataFrame({"id": [1], "score_1": [np.inf]}), "non-finite"),
    ],
)
def test_validate_prediction_frame_raises_for_contract_violations(preds, match):
    with pytest.raises(ValueError, match=match):
        validate_prediction_frame(preds, id_col="id", output_cols=["score_1"])


def test_stack_feature_names_uses_binary_model_name_only():
    assert stack_feature_names("lgbm", ["score_1"]) == ["lgbm"]


def test_stack_feature_names_uses_multiclass_score_suffixes():
    assert stack_feature_names("lgbm", ["score_0", "score_1", "score_2"]) == [
        "lgbm_score_0",
        "lgbm_score_1",
        "lgbm_score_2",
    ]


def test_stack_feature_names_uses_regression_prediction_suffix():
    assert stack_feature_names("ridge", ["prediction"]) == ["ridge_prediction"]


def test_build_oof_stack_matrix_from_two_base_models_aligns_by_id():
    schema = DatasetSchema()
    labels = pd.DataFrame({"id": [10, 20, 30], "target": [0, 1, 0]})
    predictions = {
        "lgbm": pd.DataFrame({"id": [30, 10, 20], "score_1": [0.3, 0.1, 0.2]}),
        "rf": pd.DataFrame({"id": [20, 30, 10], "score_1": [0.8, 0.7, 0.9]}),
    }

    stack_matrix = build_oof_stack_matrix(schema=schema, labels=labels, predictions=predictions)

    expected = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "target": [0, 1, 0],
            "lgbm": [0.1, 0.2, 0.3],
            "rf": [0.9, 0.8, 0.7],
        },
    )
    pd.testing.assert_frame_equal(stack_matrix, expected)


def test_build_oof_stack_matrix_supports_multiclass_outputs():
    schema = DatasetSchema()
    labels = pd.DataFrame({"id": [1, 2], "target": [0, 2]})
    predictions = {
        "lgbm": pd.DataFrame(
            {
                "id": [2, 1],
                "score_0": [0.1, 0.8],
                "score_1": [0.2, 0.1],
                "score_2": [0.7, 0.1],
            },
        ),
    }

    stack_matrix = build_oof_stack_matrix(schema=schema, labels=labels, predictions=predictions)

    expected = pd.DataFrame(
        {
            "id": [1, 2],
            "target": [0, 2],
            "lgbm_score_0": [0.8, 0.1],
            "lgbm_score_1": [0.1, 0.2],
            "lgbm_score_2": [0.1, 0.7],
        },
    )
    pd.testing.assert_frame_equal(stack_matrix, expected)


def test_average_fold_predictions_aligns_by_id_and_averages_outputs_only():
    schema = DatasetSchema()
    folds = [
        pd.DataFrame({"id": [1, 2, 3], "score_1": [0.1, 0.2, 0.3], "ignored": [9, 9, 9]}),
        pd.DataFrame({"id": [3, 1, 2], "score_1": [0.9, 0.7, 0.5], "ignored": [1, 1, 1]}),
    ]

    averaged = average_fold_predictions(schema=schema, prediction_frames=folds, output_cols=["score_1"])

    expected = pd.DataFrame({"id": [1, 2, 3], "score_1": [0.4, 0.35, 0.6]})
    pd.testing.assert_frame_equal(averaged, expected)


def test_build_serving_stack_matrix_aligns_two_base_models_by_id_without_target():
    schema = DatasetSchema()
    base_predictions = {
        "lgbm": pd.DataFrame({"id": [2, 1], "score_1": [0.8, 0.7]}),
        "ridge": pd.DataFrame({"id": [1, 2], "prediction": [12.0, 14.0]}),
    }

    stack_matrix = build_serving_stack_matrix(schema=schema, base_predictions=base_predictions)

    expected = pd.DataFrame({"id": [2, 1], "lgbm": [0.8, 0.7], "ridge_prediction": [14.0, 12.0]})
    pd.testing.assert_frame_equal(stack_matrix, expected)


@pytest.mark.parametrize(
    "predictions",
    [
        {"lgbm": pd.DataFrame({"id": [1, 2], "score_1": [0.1, 0.2]})},
        {"lgbm": pd.DataFrame({"id": [1, 2, 4], "score_1": [0.1, 0.2, 0.4]})},
        {"lgbm": pd.DataFrame({"id": [1, 2, 2], "score_1": [0.1, 0.2, 0.3]})},
    ],
)
def test_build_oof_stack_matrix_raises_for_missing_extra_and_duplicate_ids(predictions):
    schema = DatasetSchema()
    labels = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})

    with pytest.raises(ValueError, match=r"missing ids|extra ids|duplicate ids"):
        build_oof_stack_matrix(schema=schema, labels=labels, predictions=predictions)


@pytest.mark.parametrize(
    "folds",
    [
        [
            pd.DataFrame({"id": [1, 2, 3], "score_1": [0.1, 0.2, 0.3]}),
            pd.DataFrame({"id": [1, 2], "score_1": [0.1, 0.2]}),
        ],
        [
            pd.DataFrame({"id": [1, 2, 3], "score_1": [0.1, 0.2, 0.3]}),
            pd.DataFrame({"id": [1, 2, 4], "score_1": [0.1, 0.2, 0.4]}),
        ],
        [
            pd.DataFrame({"id": [1, 2, 3], "score_1": [0.1, 0.2, 0.3]}),
            pd.DataFrame({"id": [1, 2, 2], "score_1": [0.1, 0.2, 0.3]}),
        ],
    ],
)
def test_average_fold_predictions_raises_for_missing_extra_and_duplicate_ids(folds):
    schema = DatasetSchema()

    with pytest.raises(ValueError, match=r"missing ids|extra ids|duplicate ids"):
        average_fold_predictions(schema=schema, prediction_frames=folds, output_cols=["score_1"])


@pytest.mark.parametrize(
    "base_predictions",
    [
        {
            "lgbm": pd.DataFrame({"id": [1, 2, 3], "score_1": [0.1, 0.2, 0.3]}),
            "rf": pd.DataFrame({"id": [1, 2], "score_1": [0.1, 0.2]}),
        },
        {
            "lgbm": pd.DataFrame({"id": [1, 2, 3], "score_1": [0.1, 0.2, 0.3]}),
            "rf": pd.DataFrame({"id": [1, 2, 4], "score_1": [0.1, 0.2, 0.4]}),
        },
        {
            "lgbm": pd.DataFrame({"id": [1, 2, 3], "score_1": [0.1, 0.2, 0.3]}),
            "rf": pd.DataFrame({"id": [1, 2, 2], "score_1": [0.1, 0.2, 0.3]}),
        },
    ],
)
def test_build_serving_stack_matrix_raises_for_missing_extra_and_duplicate_ids(base_predictions):
    schema = DatasetSchema()

    with pytest.raises(ValueError, match=r"missing ids|extra ids|duplicate ids"):
        build_serving_stack_matrix(schema=schema, base_predictions=base_predictions)


def test_stacking_ensemble_records_lineage():
    ensemble = StackingEnsemble(
        name="stack",
        base_model_names=["lgbm", "rf"],
        stack_feature_columns=["lgbm", "rf"],
        class_order_by_model={"lgbm": [0, 1]},
    )

    assert ensemble.base_model_names == ["lgbm", "rf"]
    assert ensemble.stack_feature_columns == ["lgbm", "rf"]
    assert ensemble.class_order_by_model == {"lgbm": [0, 1]}
