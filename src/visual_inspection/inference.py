from visual_inspection.features import extract_image_features
from visual_inspection.scoring import knn_anomaly_score, predict_from_scores


def predict_features(query_features, artifact) -> dict:
    anomaly_score = knn_anomaly_score(
        artifact["reference_features"],
        query_features,
        k=artifact["k"],
    )
    threshold = artifact["threshold"]
    prediction = predict_from_scores(anomaly_score, threshold)[0]
    return {
        "prediction": prediction,
        "anomaly_score": float(anomaly_score.item()),
        "threshold": float(threshold),
    }


def predict_image(image_path, model, preprocess, artifact) -> dict:
    features = extract_image_features(image_path, model, preprocess)
    return predict_features(features, artifact)
