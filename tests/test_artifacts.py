import pytest
import torch

from visual_inspection.artifacts import load_model_artifact, save_model_artifact


def test_save_and_load_model_artifact_roundtrip(tmp_path):
    path = tmp_path / "artifacts" / "bottle_knn_model.pt"
    reference_features = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    threshold = torch.tensor(1.5)

    save_model_artifact(
        path=path,
        reference_features=reference_features,
        threshold=threshold,
        k=5,
        method="knn",
        category="bottle",
    )

    artifact = load_model_artifact(path)

    torch.testing.assert_close(artifact["reference_features"], reference_features)
    assert artifact["threshold"] == 1.5
    assert artifact["k"] == 5
    assert artifact["method"] == "knn"
    assert artifact["category"] == "bottle"


def test_load_model_artifact_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model_artifact(path)


def test_load_model_artifact_rejects_missing_required_key(tmp_path):
    path = tmp_path / "invalid.pt"
    torch.save(
        {
            "reference_features": torch.tensor([[1.0, 2.0]]),
            "threshold": 1.0,
            "k": 5,
            "method": "knn",
        },
        path,
    )

    with pytest.raises(ValueError, match="missing required keys"):
        load_model_artifact(path)


def test_save_model_artifact_rejects_empty_reference_features(tmp_path):
    path = tmp_path / "model.pt"

    with pytest.raises(ValueError, match="Reference features tensor is empty"):
        save_model_artifact(
            path=path,
            reference_features=torch.empty((0, 2)),
            threshold=1.0,
            k=5,
            method="knn",
            category="bottle",
        )


def test_save_model_artifact_rejects_non_2d_reference_features(tmp_path):
    path = tmp_path / "model.pt"

    with pytest.raises(ValueError, match="Reference features tensor must be 2D"):
        save_model_artifact(
            path=path,
            reference_features=torch.tensor([1.0, 2.0]),
            threshold=1.0,
            k=5,
            method="knn",
            category="bottle",
        )


def test_save_model_artifact_rejects_non_scalar_threshold(tmp_path):
    path = tmp_path / "model.pt"

    with pytest.raises(ValueError, match="Threshold must be a scalar"):
        save_model_artifact(
            path=path,
            reference_features=torch.tensor([[1.0, 2.0]]),
            threshold=torch.tensor([1.0, 2.0]),
            k=5,
            method="knn",
            category="bottle",
        )


@pytest.mark.parametrize("k", [0, -1, 1.5])
def test_save_model_artifact_rejects_invalid_k(tmp_path, k):
    path = tmp_path / "model.pt"

    with pytest.raises(ValueError, match="k must be a positive integer"):
        save_model_artifact(
            path=path,
            reference_features=torch.tensor([[1.0, 2.0]]),
            threshold=1.0,
            k=k,
            method="knn",
            category="bottle",
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("method", ""),
        ("category", ""),
    ],
)
def test_save_model_artifact_rejects_empty_metadata(
    tmp_path,
    field_name,
    field_value,
):
    path = tmp_path / "model.pt"
    kwargs = {
        "path": path,
        "reference_features": torch.tensor([[1.0, 2.0]]),
        "threshold": 1.0,
        "k": 5,
        "method": "knn",
        "category": "bottle",
    }
    kwargs[field_name] = field_value

    with pytest.raises(ValueError, match=f"{field_name} must be a non-empty string"):
        save_model_artifact(**kwargs)
