# ai4health/features.py

from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
import typer
import yaml
import json

from ai4health.config import INTERIM_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# -------------------------------
# Defaults
# -------------------------------
MFCC_DIR = PROCESSED_DATA_DIR / "mfcc"
MFCC_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MFCC = 40
DEFAULT_DURATION = 5.0  # seconds
FIXED_LENGTH = int(DEFAULT_SAMPLE_RATE * DEFAULT_DURATION)


# -------------------------------
# MFCC Extraction Logic
# -------------------------------
def extract_mfcc(audio_path: Path, sr: int, n_mfcc: int, fixed_length: int) -> np.ndarray:
    """
    Load a .wav file and extract MFCC features.

    Args:
        audio_path (Path): Path to .wav file
        sr (int): Sampling rate
        n_mfcc (int): Number of MFCC coefficients
        fixed_length (int): Target audio length in samples

    Returns:
        np.ndarray: MFCC array (shape: [n_mfcc, T]) or None if failed
    """
    try:
        signal, _ = librosa.load(audio_path, sr=sr)

        if len(signal) < fixed_length:
            signal = np.pad(signal, (0, fixed_length - len(signal)))
        else:
            signal = signal[:fixed_length]

        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        return mfcc

    except Exception as e:
        logger.error(f"Failed to process {audio_path.name}: {e}")
        return None


# -------------------------------
# Batch Extraction Entry
# -------------------------------
@app.command()
def main(
        metadata_path: Path = INTERIM_DATA_DIR / "metadata_filtered.csv",
        audio_dir: Path = RAW_DATA_DIR / "COUGHVID",
        output_dir: Path = MFCC_DIR,
        sr: int = DEFAULT_SAMPLE_RATE,
        n_mfcc: int = DEFAULT_N_MFCC,
        duration: float = DEFAULT_DURATION,
        save_config: bool = True,
):
    """
    Extract MFCC features from .wav files listed in metadata CSV.
    Saves .npy files to processed directory.

    Args:
        metadata_path (Path): Filtered metadata CSV.
        audio_dir (Path): Directory with .wav files.
        output_dir (Path): Output dir for .npy MFCC files.
        sr (int): Sampling rate.
        n_mfcc (int): Number of MFCCs.
        duration (float): Target length in seconds.
        save_config (bool): Whether to save extraction config as .yaml.
    """
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} entries from metadata.")

    total_samples = int(sr * duration)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        wav_filename = row["filename"]
        wav_path = audio_dir / wav_filename
        output_path = output_dir / (wav_filename.replace(".wav", ".npy"))

        if output_path.exists():
            continue

        mfcc = extract_mfcc(wav_path, sr=sr, n_mfcc=n_mfcc, fixed_length=total_samples)
        if mfcc is not None:
            np.save(output_path, mfcc)
            logger.debug(f"Saved: {output_path.name}")

    logger.success(f"MFCC extraction complete. Saved to {output_dir.resolve()}")

    # Save config for reproducibility
    if save_config:
        config_path = output_dir / "mfcc_config.yaml"
        config = {
            "sr": sr,
            "n_mfcc": n_mfcc,
            "duration": duration,
            "fixed_length_samples": total_samples
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        logger.info(f"Saved extraction config to {config_path}")


if __name__ == "__main__":
    app()
