# Music Cluster Project

### Overview

The **Music Cluster Project** allows you to analyze songs from a Spotify playlist by extracting their core audio features using **Fourier Transform (FT)**, **Short-Time Fourier Transform (STFT)**, **Mel-Frequency Cepstral Coefficients (MFCC)**, and **Wavelet Transform**. By converting each song or stem into a multidimensional data point using these features, the project enables you to cluster songs based on their audio characteristics. The goal is to discover hidden similarities between tracks and create seamless DJ sets where stems from different songs can be mixed with ease.

### Project Structure

```
/MusicClusterProject/
    ├── /data/                # Folder to store .mp4 files
    ├── /features/            # Folder to store extracted features
    ├── /output/              # Folder for clustering results and visualizations
    ├── extract_features.py   # Python script to extract features
    ├── clustering.py         # Python script to perform clustering and visualization
    └── requirements.txt      # Python dependencies
```

### Features

- **Fourier Transform (FT)**: Captures the overall frequency content of the song.
- **Short-Time Fourier Transform (STFT)**: Adds a time component, capturing how frequencies evolve over time.
- **Mel-Frequency Cepstral Coefficients (MFCC)**: Mimics human auditory perception, emphasizing frequencies important for human hearing.
- **Wavelet Transform**: Captures time and frequency information at different scales.

By combining these features, the project provides a rich representation of the songs for clustering.

### Getting Started

#### Prerequisites

Before running the project, ensure that you have the required Python packages installed. You can install all dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

#### How to Use

1. **Step 1**: Place your `.mp4` files in the `/data/` directory.
   
2. **Step 2**: Extract features from the songs by running the following command:
   ```bash
   python extract_features.py
   ```
   This will process all `.mp4` files in the `data` directory and save the extracted features as JSON files in the `/features/` directory.

3. **Step 3**: Perform K-means clustering on the extracted features and visualize the clusters:
   ```bash
   python clustering.py
   ```
   This will create a scatter plot of the clustered songs, reducing their dimensionality using PCA for easy visualization.

### Output

The project outputs the following:
- **Feature Files**: JSON files containing Fourier, STFT, MFCC, and Wavelet coefficients for each song.
- **Clusters**: A visual graph showing the song clusters based on their extracted features. Songs in the same cluster have similar frequency and time-based characteristics.

### Future Prospects

Here are some future directions for improving and expanding the functionality of the Music Cluster Project:

1. **Real-Time DJ Integration**: 
   - Automate the process of mixing songs from the same cluster, allowing for seamless transitions between stems during live DJ sets.
   
2. **Dynamic Feature Selection**:
   - Add flexibility to choose which features to use for clustering (e.g., focus on rhythm-heavy clustering by prioritizing STFT or harmonic clustering using Fourier).
   
3. **Enhanced Clustering Algorithms**:
   - Explore other clustering methods such as **DBSCAN** for non-linear clusters or hierarchical clustering for dynamic cluster formation.
   
4. **Genre Prediction**:
   - Use the extracted features to predict or suggest genres for each song based on the learned clusters.
   
5. **Song Recommendation Engine**:
   - Create a recommendation system that suggests new tracks based on the similarity of their feature vectors, clustering new songs with similar ones from existing playlists.
   
6. **Expanded Audio Format Support**:
   - Extend support to other file formats (e.g., `.wav`, `.flac`) for better audio quality and more feature extraction options.
   
7. **Deep Learning Integration**:
   - Explore neural network-based feature extraction to automate complex feature identification for even richer clustering results.

### Contributing

Feel free to fork this project and submit pull requests to improve functionality or add new features!

### License

This project is licensed under the MIT License.