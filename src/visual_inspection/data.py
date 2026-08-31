from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataset_index(data_dir: Path) -> pd.DataFrame:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    if not data_dir.is_dir():
        raise ValueError(f"Expected a directory, got: {data_dir}")

    train_dir = data_dir / "train" / "good"
    test_dir = data_dir / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    train_images = sorted(train_dir.glob("*.png"))

    if not train_images:
        raise ValueError(
            f"No training PNG images found in: {train_dir}"
        )

    train_records = []

    for image_path in train_images:
        train_records.append(
            {
                "path": image_path,
                "split": "train",
                "label": "normal",
                "defect_type": "good",
            }
        )

    test_images = sorted(test_dir.rglob("*.png"))

    if not test_images:
        raise ValueError(
            f"No test PNG images found in: {test_dir}"
        )

    test_records = []

    for image_path in test_images:
        defect_type = image_path.parent.name

        test_records.append(
            {
                "path": image_path,
                "split": "test",
                "label": "normal" if defect_type == "good" else "defective",
                "defect_type": defect_type,
            }
        )

    return pd.DataFrame(train_records + test_records)



def validate_dataset_index(df: pd.DataFrame) -> None:
    required_columns = {"path", "split", "label", "defect_type"}

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset index is empty")

    if df.isna().any().any():
        raise ValueError("Dataset index contains missing values")

    if df["path"].duplicated().any():
        raise ValueError("Dataset index contains duplicated paths")

    valid_splits = {"train", "test"}
    if not set(df["split"]).issubset(valid_splits):
        raise ValueError("Dataset index contains invalid split values")

    valid_labels = {"normal", "defective"}
    if not set(df["label"]).issubset(valid_labels):
        raise ValueError("Dataset index contains invalid label values")

    train_df = df[df["split"] == "train"]

    if (train_df["label"] != "normal").any():
        raise ValueError("Defective image found in training data")


def split_normal_train_validation(
    df: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"] == "train"]

    if (train_df["label"] != "normal").any():
        raise ValueError("Training data contains defective samples")

    train_split, validation_split = train_test_split(
        train_df,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )

    return (
        train_split.reset_index(drop=True),
        validation_split.reset_index(drop=True),
    )