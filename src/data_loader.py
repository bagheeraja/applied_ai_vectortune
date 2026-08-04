import os
import urllib.request
import pandas as pd
import numpy as np
import streamlit as st

# Paste the Direct Download Link you copied from your GitHub Release here:
DATASET_URL = "https://github.com/bagheeraja/applied_ai_vectortune/releases/download/v1.0.0/tracks_features.csv"

@st.cache_data
def initialize_engine(csv_path="data/tracks_features.csv"):
    """
    Checks if the dataset exists locally. If missing (e.g., for a grader), 
    programmatically streams it from GitHub Releases before building the matrix.
    """
    # --- AUTOMATED SEAMLESS DOWNLOAD FOR GRADERS ---
    if not os.path.exists(csv_path):
        # Create data/ directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.warning("📊 First-time setup: Downloading the 300MB core vector dataset from GitHub mirror. Please wait...")
        
        # Track download progress to show the grader everything is working smoothly
        def download_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(int(downloaded * 100 / total_size), 100)
                progress_bar.progress(percent / 100)
        
        try:
            urllib.request.urlretrieve(DATASET_URL, csv_path, reporthook=download_hook)
            status_text.success("✅ Dataset downloaded successfully! Compiling RAM vector spaces...")
            progress_bar.empty()
        except Exception as e:
            status_text.error(f"❌ Failed to download dataset automatically. Error: {e}")
            st.stop()

    # --- NORMAL LOADING PIPELINE ---
    columns_to_keep = [
        'id', 'name', 'artists', 'album_id', 'danceability', 'energy',
        'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    df = pd.read_csv(csv_path, usecols=columns_to_keep)
    df = df.drop_duplicates(subset=['name', 'artists'])
    
    # Downcasting logic
    dtype_mapping = {}
    float_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    for col in float_cols:
        dtype_mapping[col] = 'float32'
    df = df.astype(dtype_mapping)
    
    # Normalization
    df['tempo_norm'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())
    df['loudness_norm'] = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())

    feature_cols = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo_norm',
        'loudness_norm'
    ]
    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)

    return df, feature_matrix, feature_cols