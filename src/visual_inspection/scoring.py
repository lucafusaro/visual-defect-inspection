import torch


def centroid_anomaly_score(reference_features, query_features):
    if reference_features.ndim != 2:
        raise ValueError("Reference features tensor must be 2D")
    if query_features.ndim != 2:
        raise ValueError("Query features tensor must be 2D")
    if reference_features.numel() == 0:
        raise ValueError("The reference features tensor is empty")
    if reference_features.shape[1] != query_features.shape[1]:
        raise ValueError(
            "Embedding dimension of reference and query tensors must be equal"
        )
    centroid = torch.mean(reference_features, dim=0)
    return torch.linalg.vector_norm(query_features - centroid, dim=1)


def knn_anomaly_score(reference_features, query_features, k=5):
    if reference_features.ndim != 2:
        raise ValueError("Reference features tensor must be 2D")
    if query_features.ndim != 2:
        raise ValueError("Query features tensor must be 2D")
    if reference_features.numel() == 0:
        raise ValueError("The reference features tensor is empty")
    if reference_features.shape[1] != query_features.shape[1]:
        raise ValueError(
            "Embedding dimension of reference and query tensors must be equal"
        )
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if k > reference_features.shape[0]:
        raise ValueError("k must not exceed the number of reference samples")

    distances = torch.cdist(query_features, reference_features)
    nearest_distances = torch.topk(
        distances,
        k=k,
        dim=1,
        largest=False,
    ).values
    return nearest_distances.mean(dim=1)


def calibrate_threshold(scores, quantile=0.95):
    if scores.numel() == 0:
        raise ValueError("The scores tensor is empty")
    if scores.dim() != 1:
        raise ValueError("The scores tensor must be 1D")
    if not 0 <= quantile <= 1:
        raise ValueError("The quantile value must be between 0 and 1")
    threshold = torch.quantile(scores, quantile)
    return threshold


def predict_from_scores(scores, threshold):
    if scores.numel() == 0:
        raise ValueError("The scores tensor is empty")
    if scores.dim() != 1:
        raise ValueError("The scores tensor must be 1D")

    threshold_tensor = torch.as_tensor(threshold)
    if threshold_tensor.dim() != 0:
        raise ValueError("The threshold must be a scalar")

    threshold = float(threshold_tensor.item())
    return [
        "defective" if float(score) > threshold else "normal"
        for score in scores
    ]
