import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load all extracted features
def load_features():
    features_list = []
    feature_dir = './features/'
    for file_name in os.listdir(feature_dir):
        if file_name.endswith('_features.json'):
            with open(os.path.join(feature_dir, file_name), 'r') as f:
                features = json.load(f)
                # Combine features into a single vector
                combined_features = features["fourier"] + features["stft"] + features["mfcc"] + features["wavelet"]
                features_list.append(combined_features)
    return np.array(features_list)

# Normalize features
def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Perform K-means clustering and return labels
def perform_kmeans(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)

# Visualize clusters using PCA for dimensionality reduction
def visualize_clusters(features, labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.title('K-means Clustering of Songs')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()

# Main function
if __name__ == "__main__":
    print("Loading features...")
    features = load_features()
    print("Normalizing features...")
    normalized_features = normalize_features(features)
    print("Performing K-means clustering...")
    labels = perform_kmeans(normalized_features, n_clusters=5)
    print("Visualizing clusters...")
    visualize_clusters(normalized_features, labels)

