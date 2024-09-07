import librosa
import numpy as np
import json
import os

def extract_fourier_transform(y, sr):
    """Extract Fourier Transform from an audio signal."""
    ft = np.fft.fft(y)
    return np.abs(ft[:len(ft)//2])  # Use half of the spectrum

def extract_stft(y, sr):
    """Extract Short-Time Fourier Transform from an audio signal."""
    stft = librosa.stft(y)
    return np.abs(stft)

def extract_mfcc(y, sr):
    """Extract Mel-Frequency Cepstral Coefficients (MFCC) from an audio signal."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs

def extract_wavelet(y):
    """Extract Wavelet Transform coefficients."""
    import pywt
    coeffs, _ = pywt.cwt(y, scales=np.arange(1, 128), wavelet='gaus1')
    return np.abs(coeffs)

def save_features(features, filename):
    """Save features to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(features, f)

def process_audio(file_path, output_dir):
    """Extract features from a single audio file and save them."""
    y, sr = librosa.load(file_path, sr=None)  # Load audio file
    
    features = {
        'fourier': extract_fourier_transform(y, sr).tolist(),
        'stft': extract_stft(y, sr).tolist(),
        'mfcc': extract_mfcc(y, sr).tolist(),
        'wavelet': extract_wavelet(y).tolist(),
    }
    
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_features(features, os.path.join(output_dir, f'{file_name}_features.json'))

def extract_all_features(data_dir, output_dir):
    """Extract features from all .mp4 files in the data directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(data_dir):
        if file.endswith(".mp4"):
            process_audio(os.path.join(data_dir, file), output_dir)
            print(f"Processed {file}")

if __name__ == "__main__":
    extract_all_features('data/', 'features/')
