from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


def run_kmeans(X, n_clusters=2, random_state=42):
    """Fit K-Means clustering and return the model and labels."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = model.fit_predict(X)

    return model, labels


def run_hierarchical(X, n_clusters=2):
    """Fit hierarchical clustering and return the model and labels."""
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )
    labels = model.fit_predict(X)

    return model, labels


def run_dbscan(X, eps=0.5, min_samples=5):
    """Fit DBSCAN and return the model and labels."""
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )
    labels = model.fit_predict(X)

    return model, labels
