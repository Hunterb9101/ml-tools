import pytest

from mltools.artifacts import ArtifactLayout


def test_artifact_layout_generates_standard_paths(tmp_path):
    layout = ArtifactLayout(root=tmp_path)

    assert layout.data_dir() == tmp_path / "data"
    assert layout.models_dir() == tmp_path / "models"
    assert layout.stacks_dir() == tmp_path / "stacks"
    assert layout.model_path(model_name="lgbm", fold_id=2) == tmp_path / "models" / "lgbm" / "fold_2" / "mdl.pkl"
    assert (
        layout.transformer_path(fold_id=2, name="encoder")
        == tmp_path / "models" / "fold_2" / "transformers" / "encoder.pkl"
    )
    assert layout.dmatrix_path(split="train", fold_id=2) == tmp_path / "models" / "fold_2" / "dmatrix" / "train.parquet"
    assert (
        layout.prediction_path(model_name="lgbm", split="oof", fold_id=2)
        == tmp_path / "models" / "lgbm" / "fold_2" / "preds" / "oof.pkl"
    )
    assert layout.stack_matrix_path(ensemble_name="blend", split="train") == (
        tmp_path / "stacks" / "blend" / "dmatrix" / "train.parquet"
    )
    assert layout.stack_model_path(ensemble_name="blend") == tmp_path / "stacks" / "blend" / "mdl.pkl"
    assert layout.metrics_path(name="lgbm") == tmp_path / "models" / "lgbm.json"


def test_artifact_layout_fold_id_defaults_to_zero(tmp_path):
    layout = ArtifactLayout(root=tmp_path)

    assert layout.model_path(model_name="lgbm") == tmp_path / "models" / "lgbm" / "fold_0" / "mdl.pkl"
    assert layout.transformer_path(name="encoder") == tmp_path / "models" / "fold_0" / "transformers" / "encoder.pkl"
    assert layout.dmatrix_path(split="train") == tmp_path / "models" / "fold_0" / "dmatrix" / "train.parquet"
    assert (
        layout.prediction_path(model_name="lgbm", split="test")
        == tmp_path / "models" / "lgbm" / "fold_0" / "preds" / "test.pkl"
    )


def test_artifact_layout_methods_do_not_create_files(tmp_path):
    layout = ArtifactLayout(root=tmp_path / "artifacts")

    paths = [
        layout.data_dir(),
        layout.models_dir(),
        layout.stacks_dir(),
        layout.model_path(model_name="lgbm"),
        layout.transformer_path(name="encoder"),
        layout.dmatrix_path(split="train"),
        layout.prediction_path(model_name="lgbm", split="test"),
        layout.stack_matrix_path(ensemble_name="blend", split="train"),
        layout.stack_model_path(ensemble_name="blend"),
        layout.metrics_path(name="lgbm"),
    ]

    assert not layout.root.exists()
    assert all(not path.exists() for path in paths)


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("model_path", {"model_name": "../lgbm"}),
        ("transformer_path", {"name": "transformers/encoder"}),
        ("dmatrix_path", {"split": ""}),
        ("prediction_path", {"model_name": "lgbm", "split": " train"}),
        ("stack_matrix_path", {"ensemble_name": "blend/test", "split": "train"}),
        ("stack_model_path", {"ensemble_name": ".."}),
        ("metrics_path", {"name": "metrics.json/extra"}),
    ],
)
def test_artifact_layout_rejects_nested_or_empty_names(tmp_path, method_name, kwargs):
    layout = ArtifactLayout(root=tmp_path)
    method = getattr(layout, method_name)

    with pytest.raises(ValueError, match=r"path segment|path separators|non-empty"):
        method(**kwargs)
