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
