import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_features(feature_dir):
    """Load all feature files from the directory."""
    features = []
    for file in os.listdir(feature_dir):
        if file.endswith('_features.json'):
            with open(os.path.join(feature_dir, file), 'r') as f:
                data = json.load(f)
                combined_features = np.concatenate(
                    [np.array(data['fourier']),
                     np.array(data['stft']).flatten(),
                     np.array(data['mfcc']).flatten(),
                     np.array(data['wavelet']).flatten()]
                )
                features.append(combined_features)
    return np.array(features)

def perform_clustering(features, n_clusters=5):
    """Perform K-means clustering on the extracted features."""
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reduced_features)

    return reduced_features, kmeans.labels_

def visualize_clusters(reduced_features, labels):
    """Visualize the clustered data."""
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.title("Song Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()

if __name__ == "__main__":
    features = load_features('features/')
    reduced_features, labels = perform_clustering(features, n_clusters=5)
    visualize_clusters(reduced_features, labels)
