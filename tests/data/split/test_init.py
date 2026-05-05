from mltools.data import split


def test_split_exports():
    assert split.__all__ == [
        "assign_folds",
        "assign_holdout",
        "compute_oos_tts_split",
        "compute_tts_split",
    ]
