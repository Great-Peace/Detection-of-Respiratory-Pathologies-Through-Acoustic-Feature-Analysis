from pathlib import Path
import json
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from loguru import logger
import typer

from ai4health.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


class DatasetBuilder:
    def __init__(
            self,
            input_dir: Path,
            output_metadata_path: Path,
            mfcc_output_dir: Path,
            sr: int = 16000,
            duration: int = 4,
            n_mfcc: int = 40,
            min_confidence: float = 0.3,
            allowed_labels: tuple = ("COVID-19", "healthy")
    ):
        self.input_dir = input_dir
        self.output_metadata_path = output_metadata_path
        self.mfcc_output_dir = mfcc_output_dir
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.min_confidence = min_confidence
        self.allowed_labels = allowed_labels

    def find_wav_json_pairs(self):
        wav_files = sorted(self.input_dir.glob("*.wav"))
        logger.info(f"Scanning {self.input_dir} for .wav/.json pairs...")

        paired = []
        unpaired = []

        for wav in tqdm(wav_files, desc="Matching .wav with .json"):
            json_file = wav.with_suffix(".json")
            if json_file.exists():
                paired.append((wav, json_file))
            else:
                unpaired.append(wav)

        return paired, unpaired

    def extract_metadata(self, pairs: list[tuple[Path, Path]]) -> pd.DataFrame:
        metadata_records = []

        logger.info("Extracting metadata from JSON files...")

        for wav_path, json_path in tqdm(pairs, desc="Parsing metadata"):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON: {json_path.name}")
                continue

            record = {
                "filename": wav_path.name,
                "json_file": json_path.name,
                "id": wav_path.stem,
                "status": data.get("status"),
                "age": data.get("age"),
                "gender": data.get("gender"),
                "cough_detected": data.get("cough_detected", -1),
                "respiratory_condition": data.get("respiratory_condition", "unspecified"),
                "fever_muscle_pain": data.get("fever_muscle_pain", "unspecified"),
                "difficulty_in_breathing": data.get("difficulty_in_breathing", "unspecified"),
            }

            metadata_records.append(record)

        df = pd.DataFrame(metadata_records)
        df["cough_detected"] = pd.to_numeric(df["cough_detected"], errors="coerce")
        df["has_valid_wav"] = df["filename"].apply(lambda x: (self.input_dir / x).exists())
        return df

    def filter_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering metadata based on confidence and labels...")
        df = df[
            (df["has_valid_wav"]) &
            (df["status"].isin(self.allowed_labels)) &
            (df["cough_detected"] >= self.min_confidence)
            ].copy()

        df = df.reset_index(drop=True).sort_values("id")
        return df

    def extract_mfccs(self, df: pd.DataFrame):
        logger.info("Extracting MFCCs for cleaned metadata entries...")
        self.mfcc_output_dir.mkdir(parents=True, exist_ok=True)

        for _, row in tqdm(df.iterrows(), total=len(df)):
            wav_path = self.input_dir / row["filename"]
            out_path = self.mfcc_output_dir / f"{row['id']}.npy"

            try:
                y, _ = librosa.load(wav_path, sr=self.sr)
                y = librosa.util.fix_length(y, size=self.sr * self.duration)
                mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
                np.save(out_path, mfcc)
            except Exception as e:
                logger.warning(f"Failed MFCC extraction: {row['filename']} — {e}")

    def run(self):
        paired, unpaired = self.find_wav_json_pairs()
        logger.info(f"Total .wav files found: {len(paired) + len(unpaired)}")
        logger.success(f"Matched pairs (.wav + .json): {len(paired)}")

        if unpaired:
            logger.warning(f"Unpaired .wav files: {len(unpaired)}")
            for wav in unpaired[:5]:
                logger.warning(f"Unpaired example: {wav.name}")
            if len(unpaired) > 5:
                logger.warning("...more unpaired files not shown.")

        df = self.extract_metadata(paired)
        df_filtered = self.filter_metadata(df)

        self.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(self.output_metadata_path, index=False)
        logger.success(f"Clean metadata saved to: {self.output_metadata_path}")

        self.extract_mfccs(df_filtered)
        logger.success(f"MFCCs saved to: {self.mfcc_output_dir}")


@app.command()
def main(
        input_dir: Path = RAW_DATA_DIR / "COUGHVID",
        output_metadata_path: Path = PROCESSED_DATA_DIR / "metadata_clean.csv",
        mfcc_output_dir: Path = PROCESSED_DATA_DIR / "mfccs",
        sr: int = 16000,
        duration: int = 4,
        n_mfcc: int = 40,
        min_confidence: float = 0.3,
):
    """
    Extracts structured metadata and MFCCs from raw .wav/.json files.

    Args:
        input_dir (Path): Directory containing .wav and .json files.
        output_metadata_path (Path): Path to save cleaned metadata CSV.
        mfcc_output_dir (Path): Output directory for saved MFCC .npy files.
        sr (int): Sampling rate.
        duration (int): Audio duration in seconds.
        n_mfcc (int): Number of MFCC coefficients.
        min_confidence (float): Minimum cough detection score.
    """
    builder = DatasetBuilder(
        input_dir=input_dir,
        output_metadata_path=output_metadata_path,
        mfcc_output_dir=mfcc_output_dir,
        sr=sr,
        duration=duration,
        n_mfcc=n_mfcc,
        min_confidence=min_confidence,
    )
    builder.run()


if __name__ == "__main__":
    app()
