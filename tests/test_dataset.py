# tests/test_dataset.py

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from ai4health.dataset import DatasetBuilder

# Dummy paths (replace or mock for actual tests)
TEST_RAW_DIR = Path("tests/data/raw/COUGHVID_mock")
TEST_PROCESSED_METADATA = Path("../tests/data/processed/metadata_clean.csv")
TEST_PROCESSED_MFCC = Path("../tests/data/processed/mfccs")

@pytest.fixture
def builder():
    return DatasetBuilder(
        input_dir=TEST_RAW_DIR,
        output_metadata_path=TEST_PROCESSED_METADATA,
        mfcc_output_dir=TEST_PROCESSED_MFCC,
        sr=16000,
        duration=4,
        n_mfcc=13,
        min_confidence=0.1,
        allowed_labels=("COVID-19", "healthy")
    )

def test_pairing_and_json_existence(builder):
    pairs, unpaired = builder.find_wav_json_pairs()
    assert isinstance(pairs, list)
    assert all(len(p) == 2 for p in pairs)

def test_extract_metadata_format(builder):
    pairs, _ = builder.find_wav_json_pairs()
    df = builder.extract_metadata(pairs)
    assert isinstance(df, pd.DataFrame)
    assert "status" in df.columns
    assert "cough_detected" in df.columns
    assert df["cough_detected"].dtype in ["float64", "float32"]

def test_filtering_logic(builder):
    pairs, _ = builder.find_wav_json_pairs()
    df = builder.extract_metadata(pairs)
    filtered_df = builder.filter_metadata(df)
    assert not filtered_df.empty
    assert all(filtered_df["status"].isin(builder.allowed_labels))
    assert all(filtered_df["cough_detected"] >= builder.min_confidence)

def test_mfcc_extraction_shape(builder):
    # Mocked example only: requires valid .wav file in tests/data/raw/
    df = pd.DataFrame({
        "filename": ["example.wav"],
        "id": ["example"],
        "status": ["healthy"],
        "cough_detected": [1.0],
        "has_valid_wav": [True]
    })
    builder.extract_mfccs(df)
    mfcc_path = TEST_PROCESSED_MFCC / "example.npy"
    assert mfcc_path.exists()
    mfcc = np.load(mfcc_path)
    assert mfcc.shape[0] == builder.n_mfcc
