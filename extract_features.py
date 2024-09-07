import librosa
import numpy as np
import json
import os
import random
import shutil  # For clearing the directory

# Global variables for feature selection, cutoff time, and random file selection
EXTRACT_FOURIER = True
EXTRACT_STFT = False
EXTRACT_MFCC = False
EXTRACT_WAVELET = False  # Set to False if you want faster execution
CUTOFF_TIME = 5  # Number of seconds of audio to analyze
NUM_RANDOM_FILES = 100  # Number of random files to process from the data directory
RANDOM_SEED = 42  # Random seed for reproducibility

def set_random_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to: {seed}")

def extract_fourier_transform(y, sr):
    """Extract Fourier Transform from an audio signal."""
    print("Extracting Fourier Transform...")
    ft = np.fft.fft(y)
    print("Fourier Transform extraction complete.")
    return np.abs(ft[:len(ft)//2])  # Use half of the spectrum

def extract_stft(y, sr):
    """Extract Short-Time Fourier Transform from an audio signal."""
    print("Extracting Short-Time Fourier Transform (STFT)...")
    stft = librosa.stft(y)
    print("STFT extraction complete.")
    return np.abs(stft)

def extract_mfcc(y, sr):
    """Extract Mel-Frequency Cepstral Coefficients (MFCC) from an audio signal."""
    print("Extracting Mel-Frequency Cepstral Coefficients (MFCC)...")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print("MFCC extraction complete.")
    return mfccs

def extract_wavelet(y):
    """Extract Wavelet Transform coefficients."""
    import pywt
    print("Extracting Wavelet Transform...")
    coeffs, _ = pywt.cwt(y, scales=np.arange(1, 128), wavelet='gaus1')
    print("Wavelet Transform extraction complete.")
    return np.abs(coeffs)

def save_features(features, filename):
    """Save features to a JSON file."""
    print(f"Saving features to {filename}...")
    with open(filename, 'w') as f:
        json.dump(features, f)
    print(f"Features saved to {filename}.")

def process_audio(file_path, output_dir):
    """Extract features from a single audio file and save them."""
    print(f"\nProcessing file: {file_path}")
    
    try:
        # Load only up to CUTOFF_TIME seconds of audio
        y, sr = librosa.load(file_path, sr=None, duration=CUTOFF_TIME)
        print(f"Loaded {file_path} (Sample rate: {sr}, Duration: {min(len(y)/sr, CUTOFF_TIME):.2f} seconds)")

        # Prepare features dictionary based on global settings
        features = {}

        if EXTRACT_FOURIER:
            features['fourier'] = extract_fourier_transform(y, sr).tolist()

        if EXTRACT_STFT:
            features['stft'] = extract_stft(y, sr).tolist()

        if EXTRACT_MFCC:
            features['mfcc'] = extract_mfcc(y, sr).tolist()

        if EXTRACT_WAVELET:
            features['wavelet'] = extract_wavelet(y).tolist()

        # Generate output filename based on the input file
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        save_features(features, os.path.join(output_dir, f'{file_name}_features.json'))
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def clear_features_directory(output_dir):
    """Clear the features directory before extracting new features."""
    if os.path.exists(output_dir):
        print(f"Clearing the '{output_dir}' directory...")
        shutil.rmtree(output_dir)  # Remove all contents of the directory
        os.makedirs(output_dir)  # Recreate the directory
        print(f"'{output_dir}' directory cleared.")
    else:
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
        print(f"Created '{output_dir}' directory.")

def extract_all_features(data_dir, output_dir, seed=None):
    """Extract features from a random subset of .mp3 or .mp4 files in the data directory."""
    if seed is not None:
        set_random_seed(seed)
    
    clear_features_directory(output_dir)  # Clear the features directory before running
    
    # Find all files and randomly select the desired number
    files = [f for f in os.listdir(data_dir) if f.endswith(".mp3") or f.endswith(".mp4")]
    if NUM_RANDOM_FILES < len(files):
        files = random.sample(files, NUM_RANDOM_FILES)
    
    print(f"Randomly selected {len(files)} files from {data_dir}: {files}")

    total_files = len(files)

    print(f"Starting feature extraction for {total_files} file(s) in '{data_dir}'...\n")
    
    files_processed = 0
    for file in files:
        process_audio(os.path.join(data_dir, file), output_dir)
        files_processed += 1
        print(f"Progress: {files_processed}/{total_files} files processed.")

    print(f"\nFeature extraction complete. {files_processed}/{total_files} files successfully processed.")
    print(f"Extracted features saved in '{output_dir}'.")

if __name__ == "__main__":
    extract_all_features('data/', 'features/', seed=RANDOM_SEED)
