from pathlib import Path

import torch

REQUIRED_ARTIFACT_KEYS = {
    "reference_features",
    "threshold",
    "k",
    "method",
    "category",
}


def save_model_artifact(
    path: Path,
    reference_features: torch.Tensor,
    threshold,
    k: int,
    method: str,
    category: str,
) -> None:
    _validate_reference_features(reference_features)
    threshold = _validate_threshold(threshold)
    _validate_k(k)
    _validate_non_empty_string(method, "method")
    _validate_non_empty_string(category, "category")

    artifact = {
        "reference_features": reference_features.detach().cpu(),
        "threshold": threshold,
        "k": k,
        "method": method,
        "category": category,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_model_artifact(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    artifact = torch.load(path, map_location="cpu")
    _validate_artifact(artifact)
    return artifact


def _validate_artifact(artifact: dict) -> None:
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary")

    missing_keys = REQUIRED_ARTIFACT_KEYS - set(artifact)
    if missing_keys:
        raise ValueError(f"Model artifact missing required keys: {missing_keys}")

    _validate_reference_features(artifact["reference_features"])
    _validate_threshold(artifact["threshold"])
    _validate_k(artifact["k"])
    _validate_non_empty_string(artifact["method"], "method")
    _validate_non_empty_string(artifact["category"], "category")


def _validate_reference_features(reference_features: torch.Tensor) -> None:
    if not isinstance(reference_features, torch.Tensor):
        raise ValueError("Reference features must be a tensor")
    if reference_features.numel() == 0:
        raise ValueError("Reference features tensor is empty")
    if reference_features.dim() != 2:
        raise ValueError("Reference features tensor must be 2D")


def _validate_threshold(threshold) -> float:
    threshold_tensor = torch.as_tensor(threshold)
    if threshold_tensor.dim() != 0:
        raise ValueError("Threshold must be a scalar")
    return float(threshold_tensor.item())


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _validate_non_empty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
