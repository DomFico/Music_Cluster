import os
import numpy as np
import librosa
import pywt
import json
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler

# Function to compute the Fourier Transform
def compute_fourier(y, sr):
    return np.abs(fft(y))

# Function to compute STFT
def compute_stft(y, sr):
    stft = np.abs(librosa.stft(y))
    return np.mean(stft, axis=1)

# Function to compute MFCC
def compute_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Function to compute Wavelet transform
def compute_wavelet(y):
    coeffs = pywt.wavedec(y, 'db1')
    return np.concatenate([np.mean(c) for c in coeffs])

# Function to extract features from each .mp4 file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load audio with original sampling rate
    features = {
        "fourier": compute_fourier(y, sr).tolist(),
        "stft": compute_stft(y, sr).tolist(),
        "mfcc": compute_mfcc(y, sr).tolist(),
        "wavelet": compute_wavelet(y).tolist()
    }
    return features

# Directory to store extracted features
output_dir = './features/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all files in the data directory
data_dir = './data/'
for file_name in os.listdir(data_dir):
    if file_name.endswith('.mp4'):
        file_path = os.path.join(data_dir, file_name)
        print(f"Extracting features from {file_name}...")
        features = extract_features(file_path)
        # Save features as JSON
        with open(os.path.join(output_dir, f"{file_name.split('.')[0]}_features.json"), 'w') as f:
            json.dump(features, f)
        print(f"Saved features for {file_name}.")

