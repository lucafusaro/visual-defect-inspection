from io import BytesIO
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

from visual_inspection import api


def _mock_startup_dependencies(monkeypatch):
    artifact = object()
    model = object()
    preprocess = object()

    monkeypatch.setattr(api, "load_model_artifact", Mock(return_value=artifact))
    monkeypatch.setattr(
        api,
        "create_feature_extractor",
        Mock(return_value=(model, preprocess)),
    )

    return artifact, model, preprocess


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_returns_ok(monkeypatch):
    _mock_startup_dependencies(monkeypatch)

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_for_valid_image(monkeypatch):
    artifact, model, preprocess = _mock_startup_dependencies(monkeypatch)
    features = object()
    prediction = {
        "prediction": "normal",
        "anomaly_score": 1.25,
        "threshold": 2.0,
    }
    extract_pil_image_features = Mock(return_value=features)
    predict_features = Mock(return_value=prediction)
    monkeypatch.setattr(
        api,
        "extract_pil_image_features",
        extract_pil_image_features,
    )
    monkeypatch.setattr(api, "predict_features", predict_features)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            files={"image": ("image.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == prediction
    extract_pil_image_features.assert_called_once()
    image, received_model, received_preprocess = (
        extract_pil_image_features.call_args.args
    )
    assert image.mode == "RGB"
    assert received_model is model
    assert received_preprocess is preprocess
    predict_features.assert_called_once_with(features, artifact)


def test_predict_rejects_invalid_image(monkeypatch):
    _mock_startup_dependencies(monkeypatch)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            files={"image": ("not-an-image.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid image"}
