import pytest
import torch

from visual_inspection.scoring import centroid_anomaly_score, knn_anomaly_score


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
