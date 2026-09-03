import json
import sys
from pathlib import Path
from unittest.mock import Mock

from visual_inspection import cli


def test_predict_command_prints_prediction_as_json(monkeypatch, capsys):
    artifact = {"reference_features": "reference-features"}
    model = object()
    preprocess = object()
    prediction = {
        "prediction": "normal",
        "anomaly_score": 1.25,
        "threshold": 2.0,
    }
    load_model_artifact = Mock(return_value=artifact)
    create_feature_extractor = Mock(return_value=(model, preprocess))
    predict_image = Mock(return_value=prediction)

    monkeypatch.setattr(cli, "load_model_artifact", load_model_artifact)
    monkeypatch.setattr(cli, "create_feature_extractor", create_feature_extractor)
    monkeypatch.setattr(cli, "predict_image", predict_image)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visual-inspection",
            "predict",
            "--image",
            "image.png",
            "--artifact",
            "model.pt",
        ],
    )

    cli.main()

    load_model_artifact.assert_called_once_with(Path("model.pt"))
    create_feature_extractor.assert_called_once_with()
    predict_image.assert_called_once_with(
        Path("image.png"),
        model,
        preprocess,
        artifact,
    )
    assert json.loads(capsys.readouterr().out) == prediction
