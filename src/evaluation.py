from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def evaluate_clustering(X, labels):
    """
    Calculate clustering performance metrics.

    Returns None when fewer than two valid clusters exist.
    """
    unique_labels = set(labels)

    # Exclude DBSCAN noise label
    valid_clusters = unique_labels - {-1}

    if len(valid_clusters) < 2:
        return None

    return {
        "silhouette_score": silhouette_score(X, labels),
        "davies_bouldin_score": davies_bouldin_score(X, labels),
        "calinski_harabasz_score": calinski_harabasz_score(X, labels),
    }
