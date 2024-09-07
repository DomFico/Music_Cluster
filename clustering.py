import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Global variables for feature selection and padding/truncating
USE_FOURIER = True
USE_STFT = False
USE_MFCC = False
USE_WAVELET = False

TARGET_FEATURE_LENGTH = 600  # Fixed length for each feature vector (adjustable)

def load_features(feature_dir):
    """Load all feature files from the directory and pad/truncate to fixed size."""
    features = []
    song_names = []
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('_features.json')]
    
    print(f"Loading features from {feature_dir} for the selected files: {feature_files}")
    
    for file in feature_files:
        print(f"Processing feature file: {file}")
        with open(os.path.join(feature_dir, file), 'r') as f:
            data = json.load(f)
            
            combined_features = []

            # Extract the song name from the filename
            song_name = os.path.splitext(file)[0]
            song_names.append(song_name)

            # Handle Fourier Transform
            if USE_FOURIER and 'fourier' in data:
                fourier_features = np.array(data['fourier'])
                print(f" - Fourier features original shape: {fourier_features.shape}")
                fourier_padded = pad_or_truncate(fourier_features, TARGET_FEATURE_LENGTH)
                print(f" - Fourier features padded/truncated to shape: {fourier_padded.shape}")
                combined_features.append(fourier_padded)

            # Handle STFT (flattened and padded)
            if USE_STFT and 'stft' in data:
                stft_features = np.array(data['stft']).flatten()
                print(f" - STFT features original shape: {stft_features.shape}")
                stft_padded = pad_or_truncate(stft_features, TARGET_FEATURE_LENGTH)
                print(f" - STFT features padded/truncated to shape: {stft_padded.shape}")
                combined_features.append(stft_padded)

            # Handle MFCC (flattened and padded)
            if USE_MFCC and 'mfcc' in data:
                mfcc_features = np.array(data['mfcc']).flatten()
                print(f" - MFCC features original shape: {mfcc_features.shape}")
                mfcc_padded = pad_or_truncate(mfcc_features, TARGET_FEATURE_LENGTH)
                print(f" - MFCC features padded/truncated to shape: {mfcc_padded.shape}")
                combined_features.append(mfcc_padded)

            # Handle Wavelet (flattened and padded)
            if USE_WAVELET and 'wavelet' in data:
                wavelet_features = np.array(data['wavelet']).flatten()
                print(f" - Wavelet features original shape: {wavelet_features.shape}")
                wavelet_padded = pad_or_truncate(wavelet_features, TARGET_FEATURE_LENGTH)
                print(f" - Wavelet features padded/truncated to shape: {wavelet_padded.shape}")
                combined_features.append(wavelet_padded)

            if combined_features:
                # Concatenate the selected features
                combined_array = np.concatenate(combined_features)
                print(f" - Combined feature shape: {combined_array.shape}")
                features.append(combined_array)

    print(f"Total number of feature sets loaded: {len(features)}")
    return np.array(features), song_names

def pad_or_truncate(feature_array, target_length):
    """Pad or truncate a feature array to a fixed target length."""
    print(f"Padding or truncating array of shape {feature_array.shape} to {target_length}")
    if len(feature_array) > target_length:
        # Truncate if the feature array is longer than the target length
        return feature_array[:target_length]
    elif len(feature_array) < target_length:
        # Zero-pad if the feature array is shorter than the target length
        pad_width = target_length - len(feature_array)
        return np.pad(feature_array, (0, pad_width), mode='constant')
    else:
        return feature_array

def perform_clustering(features, n_clusters=5):
    """Perform K-means clustering on the extracted features."""
    print(f"Performing PCA to reduce dimensionality to 2 components...")
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    reduced_features = pca.fit_transform(features)
    print(f"PCA complete. Reduced features shape: {reduced_features.shape}")

    print(f"Running K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reduced_features)
    print(f"K-means clustering complete. Cluster labels: {kmeans.labels_}")

    return reduced_features, kmeans.labels_

def print_cluster_assignments(labels, song_names):
    """Print the song names and the cluster they are assigned to."""
    cluster_dict = {}
    for i, label in enumerate(labels):
        cluster = label + 1  # Make cluster numbers start from 1
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(song_names[i])

    print("\nCluster Assignments:")
    for cluster, songs in cluster_dict.items():
        print(f"\nCluster {cluster}:")
        for song in songs:
            print(f"  - {song}")

def visualize_clusters(reduced_features, labels):
    """Visualize the clustered data with cluster numbers (1, 2, 3, etc.)."""
    print("Visualizing the clusters...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of the clusters
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.title("Song Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Add a colorbar for the clusters
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')

    plt.show()

if __name__ == "__main__":
    # Load all feature files in the 'features' directory
    features, song_names = load_features('features/')
    
    if len(features) == 0:
        print("No features found or selected.")
    else:
        print(f"Features loaded. Shape: {features.shape}")
        reduced_features, labels = perform_clustering(features, n_clusters=5)

        # Print cluster assignments before displaying the plot
        print_cluster_assignments(labels, song_names)
        
        # Show the scatter plot
        visualize_clusters(reduced_features, labels)

    print("Clustering process complete.")
