# Music Genre Classification Project - Feature Extraction

# Imports
import os
import warnings
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import pywt
warnings.filterwarnings("ignore", category=UserWarning)


# Feature Functions
def calculate_energy_entropy(y, frame_length=2048, hop_length=512, n_bands=10):
    """Calculates short-term Energy Entropy"""

    # Calculate the Root Mean Square (RMS) energy
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0] ** 2
    # Divide energy into n_bands
    frame_energy = np.sum(librosa.util.frame(energy, frame_length=n_bands, hop_length=n_bands), axis=0)

    total_energy = np.sum(frame_energy)
    if total_energy == 0:
        # Handle silence or near-zero energy to prevent log error
        return np.zeros(1)

    # Calculate probability distribution of energy
    prob_dist = frame_energy / total_energy
    # Apply the Shannon Entropy formula
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
    return np.array([entropy])

def calculate_spectral_entropy(y, sr, n_fft=2048, hop_length=512):
    """Calculates short-term Spectral Entropy"""

    # Compute the magnitude spectrum
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # Normalize the magnitude spectrum
    S_norm = S / np.sum(S, axis=0, keepdims=True)
    # Apply the Shannon Entropy formula across the frequency bins
    spectral_entropy = -np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)
    return spectral_entropy

def calculate_pwp(y, wavelet='db4', level=4):
    """Calculates the normalized energy coefficients for Perceptual Wavelet Packets"""
    try:
        # Perform Wavelet Packet Decomposition
        wp = pywt.WaveletPacket(data=y, wavelet=wavelet, maxlevel=level)
        # Extract the signal components from the terminal nodes
        terminal_nodes = [node.data for node in wp.get_level(level, 'natural')]
        # Calculate the energy for each node/band
        pwp_energy = np.array([np.sum(node ** 2) for node in terminal_nodes])
        # Normalize the energy to obtain the PWP coefficients
        pwp_norm = pwp_energy / np.sum(pwp_energy)
        return pwp_norm
    except Exception:
        # Return zeros if decomposition fails
        return np.zeros(2 ** level)

# Audio Preprocessing
def highpass_filter(y, sr=22050, cutoff=30, order=3):
    """Applies a Butterworth high-pass filter to remove low-frequency noise"""

    # Normalize the cutoff frequency against the Nyquist frequency
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    # Design the Butterworth filter coefficients
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # Apply the filter
    return filtfilt(b, a, y)


# Feature Extraction Main Function
def extract_features(file_path, sr_target=22050, apply_filter=False):
    """Extracts the final feature vector"""
    # Load audio, resample, and limit to 30 seconds
    y, sr = librosa.load(file_path, sr=sr_target, duration=30)
    if y is None or len(y) == 0:
        raise ValueError("Empty audio")

    if apply_filter:
        try:
            # Apply pre-processing to remove low-frequency noise
            y = highpass_filter(y, sr, cutoff=30)
        except Exception:
            # Fail silently if filter application causes an issue
            pass

    # Feature Calculation
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)[0]

    # Energy Entropy
    ee_agg = calculate_energy_entropy(y, frame_length=2048, hop_length=512, n_bands=10)
    # Spectral Entropy
    se = calculate_spectral_entropy(y, sr, n_fft=2048, hop_length=512)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)

    # Perceptual Wavelet Packets (PWP)
    pwp = calculate_pwp(y, level=4)

    # Feature Assembly (Mean and Std)
    fv = [
        zcr.mean(), zcr.std(),
        spec_centroid.mean(), spec_centroid.std(),
    ]

    # Append aggregated entropy features
    fv.extend([ee_agg.mean(), ee_agg.std(), se.mean(), se.std()])

    # Append PWP coefficients
    fv.extend(pwp)

    # Append MFCCs and Chroma
    fv.extend(mfcc.mean(axis=1))
    fv.extend(mfcc.std(axis=1))
    fv.extend(chroma.mean(axis=1))

    return fv

# Dataset Assembly Functions
def get_feature_names():
    """Generates column headers corresponding to the output order of extract_features()"""
    names = [
        "zcr_mean", "zcr_std",
        "centroid_mean", "centroid_std",
    ]

    # Entropy Features
    names.extend(["energy_entropy_mean", "energy_entropy_std",
                 "spectral_entropy_mean", "spectral_entropy_std"])

    # PWP Features
    names.extend([f"pwp{i}" for i in range(16)])

    # MFCCs
    for i in range(1, 14):
        names.append(f"mfcc{i}_mean")
    for i in range(1, 14):
        names.append(f"mfcc{i}_std")

    # Chroma
    for i in range(1, 13):
        names.append(f"chroma{i}_mean")

    return names

def build_feature_csv(root='GTZAN', out_csv=os.path.join("results", "features_gtzan.csv"), overwrite=False):
    """Loads features from CSV or extracts features from audio files"""

    # Attempt to load existing feature CSV
    if os.path.exists(out_csv) and not overwrite:
        print(f"{out_csv} already exists. Loading it.")

        df = pd.read_csv(out_csv)

        # Data Cleaning
        feature_cols = df.columns.drop('genre', errors='ignore')

        # Clean the strings: remove brackets, quotes and whitespaces
        for col in feature_cols:
            df[col] = df[col].astype(str).str.strip('[]\'\" ').replace('nan', np.nan)

        # Convert features to numeric
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

        # Drop rows where feature extraction failed or resulted in NaN
        df.dropna(subset=feature_cols, inplace=True)

        if df.shape[0] == 0:
            print(
                "WARNING: Loaded CSV resulted in 0 rows after cleaning. Please set overwrite=True or check the CSV file.")

        return df

    # Proceed with extraction if file is missing or overwrite is True
    feature_names = get_feature_names()
    rows, labels = [], []
    # Identify genre subdirectories within the root path
    genres = sorted([g for g in os.listdir(root) if os.path.isdir(os.path.join(root, g))])
    print("Found genres:", genres)

    # Main extraction loop
    for genre in genres:
        # Use tqdm for progress bar visualization
        for f in tqdm(sorted([x for x in os.listdir(os.path.join(root, genre)) if x.lower().endswith(".wav")]),
                      desc=f"Processing {genre}"):
            fp = os.path.join(root, genre, f)
            try:
                # Call the main extraction function
                fv = extract_features(fp)
                rows.append(fv)
                labels.append(genre)
            except Exception as e:
                print(f"Error processing {fp}: {e}")

    # Finalize and Save
    df = pd.DataFrame(rows, columns=feature_names)
    df["genre"] = labels
    df.to_csv(out_csv, index=False)

    print(f"Saved features to {out_csv} shape: {df.shape}")
    return build_feature_csv(root='GTZAN', out_csv=os.path.join("results", "features_gtzan.csv"), overwrite=False)

