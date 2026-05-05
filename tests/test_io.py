from pathlib import Path

import pandas as pd
import pytest

from mltools.io import read_dataframe, read_file, write_dataframe, write_file


def test_pickle_file_round_trip(tmp_path):
    artifact = {"model": "demo", "weights": [1.0, 2.5, 3.0]}
    path = tmp_path / "nested" / "artifact.pkl"

    written_path = write_file(artifact, path)
    loaded = read_file(path)

    assert written_path == path.resolve()
    assert loaded == artifact


def test_json_file_round_trip_is_deterministic(tmp_path):
    artifact = {"z": 2, "a": {"b": 1}}
    path = tmp_path / "metrics.json"

    written_path = write_file(artifact, path)
    loaded = read_file(path)

    assert written_path == path.resolve()
    assert loaded == artifact
    assert path.read_text(encoding="utf-8") == '{\n  "a": {\n    "b": 1\n  },\n  "z": 2\n}\n'


def test_text_file_round_trip_with_format_override(tmp_path):
    path = tmp_path / "notes.artifact"

    write_file("hello", path, format="text")

    assert read_file(path, format="txt") == "hello"


def test_missing_object_artifact_read_includes_path(tmp_path):
    path = tmp_path / "missing.pkl"

    with pytest.raises(FileNotFoundError, match=str(path)):
        read_file(path)


def test_parquet_dataframe_round_trip(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "score": [0.1, 0.9], "label": ["a", "b"]})
    path = tmp_path / "matrices" / "train.parquet"

    written_path = write_dataframe(df, path)
    loaded = read_dataframe(path)

    assert written_path == path.resolve()
    pd.testing.assert_frame_equal(loaded, df)


def test_csv_dataframe_round_trip(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "score": [0.1, 0.9], "label": ["a", "b"]})
    path = tmp_path / "preds" / "train.csv"

    written_path = write_dataframe(df, path)
    loaded = read_dataframe(path)

    assert written_path == path.resolve()
    pd.testing.assert_frame_equal(loaded, df)


def test_dataframe_kwargs_are_passed_to_pandas(tmp_path):
    df = pd.DataFrame({"id": [1, 2], "score": [0.1, 0.9]})
    path = tmp_path / "indexed.csv"

    write_dataframe(df, path, index=True)
    loaded = read_dataframe(path, index_col=0)

    pd.testing.assert_frame_equal(loaded, df)


@pytest.mark.parametrize(
    "path",
    [
        Path("artifact.unknown"),
        Path("artifact"),
    ],
)
def test_object_artifact_unsupported_suffix_errors(tmp_path, path):
    with pytest.raises(ValueError, match="object artifact"):
        write_file({"a": 1}, tmp_path / path)


def test_object_artifact_unsupported_format_errors(tmp_path):
    with pytest.raises(ValueError, match="Unsupported object artifact format"):
        write_file({"a": 1}, tmp_path / "artifact.pkl", format="yaml")


def test_dataframe_unsupported_suffix_errors(tmp_path):
    with pytest.raises(ValueError, match="dataframe artifact"):
        write_dataframe(pd.DataFrame({"a": [1]}), tmp_path / "frame.json")


def test_missing_dataframe_read_includes_path(tmp_path):
    path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match=str(path)):
        read_dataframe(path)
