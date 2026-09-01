import pytest
import torch

from visual_inspection.scoring import (
    calibrate_threshold,
    centroid_anomaly_score,
    knn_anomaly_score,
    predict_from_scores,
)


def test_centroid_anomaly_score_returns_expected_distances():
    reference_features = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
        ]
    )
    query_features = torch.tensor(
        [
            [1.0, 0.0],
            [4.0, 0.0],
        ]
    )

    scores = centroid_anomaly_score(reference_features, query_features)

    expected_scores = torch.tensor([0.0, 3.0])
    torch.testing.assert_close(scores, expected_scores)
    assert scores.shape == (2,)


def test_centroid_anomaly_score_rejects_empty_reference_features():
    reference_features = torch.empty((0, 2))
    query_features = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="reference features tensor is empty"):
        centroid_anomaly_score(reference_features, query_features)


@pytest.mark.parametrize(
    ("reference_features", "query_features"),
    [
        (torch.tensor([0.0, 1.0]), torch.tensor([[0.0, 1.0]])),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([0.0, 1.0])),
    ],
)
def test_centroid_anomaly_score_rejects_non_2d_features(
    reference_features,
    query_features,
):
    with pytest.raises(ValueError, match="must be 2D"):
        centroid_anomaly_score(reference_features, query_features)


def test_centroid_anomaly_score_rejects_different_embedding_dimensions():
    reference_features = torch.tensor([[0.0, 1.0]])
    query_features = torch.tensor([[0.0, 1.0, 2.0]])

    with pytest.raises(ValueError, match="dimension.*must be equal"):
        centroid_anomaly_score(reference_features, query_features)


def test_knn_anomaly_score_returns_mean_distance_to_k_nearest_neighbors():
    reference_features = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
        ]
    )
    query_features = torch.tensor(
        [
            [1.0, 0.0],
            [5.0, 0.0],
        ]
    )

    scores = knn_anomaly_score(reference_features, query_features, k=2)

    expected_scores = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(scores, expected_scores)
    assert scores.shape == (2,)


@pytest.mark.parametrize("k", [0, -1, 1.5])
def test_knn_anomaly_score_rejects_invalid_k(k):
    reference_features = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    query_features = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="k must be a positive integer"):
        knn_anomaly_score(reference_features, query_features, k=k)


def test_knn_anomaly_score_rejects_k_larger_than_reference_set():
    reference_features = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    query_features = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="number of reference samples"):
        knn_anomaly_score(reference_features, query_features, k=3)


def test_calibrate_threshold_returns_expected_quantile():
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])

    threshold = calibrate_threshold(scores, quantile=0.5)

    torch.testing.assert_close(threshold, torch.tensor(2.5))


def test_calibrate_threshold_rejects_empty_scores():
    scores = torch.empty(0)

    with pytest.raises(ValueError, match="scores tensor is empty"):
        calibrate_threshold(scores)


def test_calibrate_threshold_rejects_non_1d_scores():
    scores = torch.tensor([[1.0, 2.0]])

    with pytest.raises(ValueError, match="scores tensor must be 1D"):
        calibrate_threshold(scores)


@pytest.mark.parametrize("quantile", [-0.1, 1.1])
def test_calibrate_threshold_rejects_quantile_outside_valid_range(quantile):
    scores = torch.tensor([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="quantile value must be between 0 and 1"):
        calibrate_threshold(scores, quantile=quantile)


def test_predict_from_scores_maps_scores_to_labels():
    scores = torch.tensor([0.5, 1.0, 1.5])
    threshold = torch.tensor(1.0)

    predictions = predict_from_scores(scores, threshold)

    assert predictions == ["normal", "normal", "defective"]


def test_predict_from_scores_accepts_float_threshold():
    scores = torch.tensor([0.5, 1.5])

    predictions = predict_from_scores(scores, threshold=1.0)

    assert predictions == ["normal", "defective"]


def test_predict_from_scores_rejects_empty_scores():
    scores = torch.empty(0)

    with pytest.raises(ValueError, match="scores tensor is empty"):
        predict_from_scores(scores, threshold=1.0)


def test_predict_from_scores_rejects_non_1d_scores():
    scores = torch.tensor([[0.5, 1.5]])

    with pytest.raises(ValueError, match="scores tensor must be 1D"):
        predict_from_scores(scores, threshold=1.0)


def test_predict_from_scores_rejects_non_scalar_threshold():
    scores = torch.tensor([0.5, 1.5])
    threshold = torch.tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="threshold must be a scalar"):
        predict_from_scores(scores, threshold)
