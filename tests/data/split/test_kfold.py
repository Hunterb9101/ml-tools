import numpy as np
import pandas as pd
import pytest

from mltools.data.split.kfold import assign_folds, assign_holdout


def _classification_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(30),
            "target": np.repeat([0, 1, 2], 10),
            "feature": np.linspace(0, 1, 30),
        },
        index=np.arange(100, 130),
    )


def test_assign_folds_is_deterministic_and_stratified():
    df = _classification_frame()

    assigned = assign_folds(df, target_col="target", n_splits=5, random_state=11)
    assigned_again = assign_folds(df, target_col="target", n_splits=5, random_state=11)

    assert assigned.index.tolist() == df.index.tolist()
    assert assigned["fold"].tolist() == assigned_again["fold"].tolist()
    assert sorted(assigned["fold"].unique().tolist()) == [0, 1, 2, 3, 4]
    for _, fold_frame in assigned.groupby("fold"):
        assert fold_frame["target"].value_counts().sort_index().tolist() == [2, 2, 2]


def test_assign_folds_supports_deterministic_unstratified_assignment():
    df = pd.DataFrame({"id": np.arange(12), "feature": np.linspace(0, 1, 12)}, index=np.arange(50, 62))

    assigned = assign_folds(df, target_col=None, n_splits=3, random_state=7)
    assigned_again = assign_folds(df, target_col=None, n_splits=3, random_state=7)

    assert assigned.index.tolist() == df.index.tolist()
    assert assigned["fold"].tolist() == assigned_again["fold"].tolist()
    assert assigned["fold"].value_counts().sort_index().tolist() == [4, 4, 4]


def test_assign_holdout_is_deterministic_and_stratified():
    df = _classification_frame()

    assigned = assign_holdout(df, target_col="target", test_size=0.3, random_state=19)
    assigned_again = assign_holdout(df, target_col="target", test_size=0.3, random_state=19)
    holdout = assigned.loc[assigned["is_holdout"]]

    assert assigned.index.tolist() == df.index.tolist()
    assert assigned["is_holdout"].tolist() == assigned_again["is_holdout"].tolist()
    assert len(holdout) == 9
    assert holdout["target"].value_counts().sort_index().tolist() == [3, 3, 3]


def test_assign_holdout_supports_deterministic_unstratified_assignment():
    df = pd.DataFrame({"id": np.arange(10), "feature": np.linspace(0, 1, 10)}, index=np.arange(20, 30))

    assigned = assign_holdout(df, target_col=None, test_size=0.2, random_state=23)
    assigned_again = assign_holdout(df, target_col=None, test_size=0.2, random_state=23)

    assert assigned.index.tolist() == df.index.tolist()
    assert assigned["is_holdout"].tolist() == assigned_again["is_holdout"].tolist()
    assert assigned["is_holdout"].sum() == 2


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1])
def test_assign_holdout_rejects_invalid_test_size(test_size):
    df = _classification_frame()

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        assign_holdout(df, target_col="target", test_size=test_size, random_state=1)


def test_assign_holdout_rejects_missing_target_column():
    df = _classification_frame().drop(columns=["target"])

    with pytest.raises(ValueError, match="target_col is missing"):
        assign_holdout(df, target_col="target", test_size=0.2, random_state=1)


def test_assign_holdout_rejects_duplicate_output_column():
    df = _classification_frame()
    df["is_holdout"] = False

    with pytest.raises(ValueError, match="Output column already exists"):
        assign_holdout(df, target_col="target", test_size=0.2, random_state=1)


@pytest.mark.parametrize("n_splits", [1, 31])
def test_assign_folds_rejects_impossible_fold_counts(n_splits):
    df = _classification_frame()

    with pytest.raises(ValueError, match="n_splits"):
        assign_folds(df, target_col="target", n_splits=n_splits, random_state=1)


def test_assign_folds_rejects_impossible_stratified_fold_counts():
    df = pd.DataFrame({"target": [0, 0, 1, 1, 1], "feature": np.arange(5)})

    with pytest.raises(ValueError, match="least frequent target class"):
        assign_folds(df, target_col="target", n_splits=3, random_state=1)


def test_assign_folds_rejects_duplicate_output_column():
    df = _classification_frame()
    df["fold"] = -1

    with pytest.raises(ValueError, match="Output column already exists"):
        assign_folds(df, target_col="target", n_splits=3, random_state=1)
