from pathlib import Path
from unittest.mock import Mock

import torch

from visual_inspection import inference


def test_predict_features_returns_normal_prediction():
    artifact = {
        "reference_features": torch.tensor([[0.0], [2.0], [10.0]]),
        "threshold": 1.0,
        "k": 2,
    }

    result = inference.predict_features(torch.tensor([[1.0]]), artifact)

    assert result == {
        "prediction": "normal",
        "anomaly_score": 1.0,
        "threshold": 1.0,
    }


def test_predict_image_extracts_features_and_delegates_prediction(monkeypatch):
    features = torch.tensor([[1.0]])
    artifact = {"reference_features": torch.tensor([[0.0]])}
    prediction = {"prediction": "normal"}
    extract_image_features = Mock(return_value=features)
    predict_features = Mock(return_value=prediction)

    monkeypatch.setattr(inference, "extract_image_features", extract_image_features)
    monkeypatch.setattr(inference, "predict_features", predict_features)

    result = inference.predict_image(
        Path("image.png"),
        model=None,
        preprocess=None,
        artifact=artifact,
    )

    extract_image_features.assert_called_once_with(Path("image.png"), None, None)
    predict_features.assert_called_once_with(features, artifact)
    assert result == prediction


def test_predict_image_returns_normal_prediction(monkeypatch):
    monkeypatch.setattr(
        inference,
        "extract_image_features",
        lambda image_path, model, preprocess: torch.tensor([[1.0]]),
    )
    artifact = {
        "reference_features": torch.tensor([[0.0], [2.0], [10.0]]),
        "threshold": 1.0,
        "k": 2,
    }

    result = inference.predict_image(
        Path("image.png"),
        model=None,
        preprocess=None,
        artifact=artifact,
    )

    assert result == {
        "prediction": "normal",
        "anomaly_score": 1.0,
        "threshold": 1.0,
    }


def test_predict_image_returns_defective_prediction(monkeypatch):
    monkeypatch.setattr(
        inference,
        "extract_image_features",
        lambda image_path, model, preprocess: torch.tensor([[3.0]]),
    )
    artifact = {
        "reference_features": torch.tensor([[0.0], [2.0], [10.0]]),
        "threshold": 1.5,
        "k": 2,
    }

    result = inference.predict_image(
        Path("image.png"),
        model=None,
        preprocess=None,
        artifact=artifact,
    )

    assert result == {
        "prediction": "defective",
        "anomaly_score": 2.0,
        "threshold": 1.5,
    }


def test_predict_image_uses_k_from_artifact(monkeypatch):
    monkeypatch.setattr(
        inference,
        "extract_image_features",
        lambda image_path, model, preprocess: torch.tensor([[3.0]]),
    )
    artifact = {
        "reference_features": torch.tensor([[0.0], [2.0], [10.0]]),
        "threshold": 1.5,
        "k": 1,
    }

    result = inference.predict_image(
        Path("image.png"),
        model=None,
        preprocess=None,
        artifact=artifact,
    )

    assert result["prediction"] == "normal"
    assert result["anomaly_score"] == 1.0
