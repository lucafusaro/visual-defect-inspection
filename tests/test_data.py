import pandas as pd
import pytest

from visual_inspection.data import (
    split_normal_train_validation,
    validate_dataset_index,
)


def test_validate_dataset_index_accepts_valid_dataframe():
    df = pd.DataFrame(
        {
            "path": [
                "train/good/001.png",
                "test/good/001.png",
                "test/broken_large/001.png",
                "test/contamination/001.png",
            ],
            "split": [
                "train",
                "test",
                "test",
                "test",
            ],
            "label": [
                "normal",
                "normal",
                "defective",
                "defective",
            ],
            "defect_type": [
                "good",
                "good",
                "broken_large",
                "contamination",
            ],
        }
    )

    validate_dataset_index(df)


def test_validate_dataset_index_rejects_duplicate_paths():
    df = pd.DataFrame(
        {
            "path": [
                "train/good/001.png",
                "test/good/001.png",
                "train/good/001.png",
                "test/contamination/001.png",
            ],
            "split": [
                "train",
                "test",
                "train",
                "test",
            ],
            "label": [
                "normal",
                "normal",
                "normal",
                "defective",
            ],
            "defect_type": [
                "good",
                "good",
                "good",
                "contamination",
            ],
        }
    )

    with pytest.raises(ValueError, match="duplicated paths"):
        validate_dataset_index(df)


def test_split_normal_train_validation_has_expected_sizes():
    df = pd.DataFrame(
        {
            "path": [f"train/good/{i:03d}.png" for i in range(10)],
            "split": ["train"] * 10,
            "label": ["normal"] * 10,
            "defect_type": ["good"] * 10,
        }
    )

    train_df, validation_df = split_normal_train_validation(
        df,
        validation_size=0.2,
        random_state=42,
    )

    assert len(train_df) == 8
    assert len(validation_df) == 2


def test_split_normal_train_validation_is_reproducible():
    df = pd.DataFrame(
        {
            "path": [f"train/good/{i:03d}.png" for i in range(10)],
            "split": ["train"] * 10,
            "label": ["normal"] * 10,
            "defect_type": ["good"] * 10,
        }
    )

    train_1, validation_1 = split_normal_train_validation(
        df,
        validation_size=0.2,
        random_state=42,
    )

    train_2, validation_2 = split_normal_train_validation(
        df,
        validation_size=0.2,
        random_state=42,
    )

    assert train_1.equals(train_2)
    assert validation_1.equals(validation_2)


def test_split_normal_train_validation_rejects_defective_training_samples():
    df = pd.DataFrame(
        {
            "path": [
                "train/good/001.png",
                "train/bad/002.png",
            ],
            "split": [
                "train",
                "train",
            ],
            "label": [
                "normal",
                "defective",
            ],
            "defect_type": [
                "good",
                "contamination",
            ],
        }
    )

    with pytest.raises(ValueError, match="defective"):
        split_normal_train_validation(df)
