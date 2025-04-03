# ai4health/modeling/predict.py

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from tqdm import tqdm
import typer

from ai4health.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
from ai4health.modeling.registry import load_registry

app = typer.Typer()

# -------------------------------
# Dataset
# -------------------------------

class MFCCDataset(Dataset):
    def __init__(self, metadata_csv: Path, mfcc_dir: Path):
        self.df = pd.read_csv(metadata_csv)
        self.mfcc_dir = mfcc_dir

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mfcc_path = self.mfcc_dir / row["filename"].replace(".wav", ".npy")
        mfcc = np.load(mfcc_path).astype(np.float32)
        mfcc = np.expand_dims(mfcc, axis=0)
        label = np.float32(row["label"])
        meta = {col: row[col] for col in self.df.columns}
        return torch.tensor(mfcc), torch.tensor(label), meta

# -------------------------------
# Model
# -------------------------------

class CNNClassifier(nn.Module):
    def __init__(self, input_shape=(1, 40, 157)):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = self.conv_block(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------------
# Plots
# -------------------------------

def save_confusion_matrix(y_true, y_pred, path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "COVID"], yticklabels=["Healthy", "COVID"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

def save_roc_curve(y_true, y_prob, path: Path):
    if len(np.unique(y_true)) < 2:
        logger.warning("Cannot plot ROC curve: only one class present.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

def save_pr_curve(y_true, y_prob, path: Path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

# -------------------------------
# Predict Logic
# -------------------------------

def predict(
        model_name: str,
        version: int,
        metadata_path: Path,
        mfcc_dir: Path,
        output_csv: Path,
        figures_dir: Path,
        batch_size: int = 64,
):
    registry = load_registry()
    if model_name not in registry:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    # Support both legacy (dict) and current (list of dicts) formats
    if isinstance(registry[model_name], dict):
        versions = {int(k.lstrip("v")): v for k, v in registry[model_name].items()}
    else:
        versions = {entry["version"]: entry for entry in registry[model_name]}

    if version not in versions:
        raise ValueError(f"Version {version} not found for model '{model_name}'")

    entry = versions[version]
    version_dir = MODELS_DIR / entry["path"]
    model_path = version_dir / entry["model_file"]

    # Load model
    logger.info(f"Loading {model_name} v{version} from registry...")
    model = CNNClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Load dataset
    dataset = MFCCDataset(metadata_path, mfcc_dir)
    loader = DataLoader(dataset, batch_size=batch_size)

    y_true, y_pred, y_prob = [], [], []
    all_meta = []

    logger.info("Running inference...")
    with torch.no_grad():
        for X, y, metas in tqdm(loader):
            outputs = model(X)
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs)
            y_pred.extend(preds)

            all_meta.extend([dict(zip(metas.keys(), v)) for v in zip(*metas.values())])

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    logger.success(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # Save plots
    save_confusion_matrix(y_true, y_pred, figures_dir / "confusion_matrix.png")
    save_roc_curve(y_true, y_prob, figures_dir / "roc_curve.png")
    save_pr_curve(y_true, y_prob, figures_dir / "pr_curve.png")

    # Save predictions
    df = pd.DataFrame(all_meta)
    df["predicted_prob"] = y_prob
    df["predicted_label"] = y_pred
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.success(f"Saved predictions to: {output_csv}")
    logger.success(f"Saved plots to: {figures_dir}")

# -------------------------------
# CLI Entrypoint
# -------------------------------

@app.command()
def main(
        model_name: str = "CNN",
        version: int = 1,
        metadata_path: Path = INTERIM_DATA_DIR / "metadata_filtered.csv",
        mfcc_dir: Path = PROCESSED_DATA_DIR / "mfcc",
        output_csv: Path = INTERIM_DATA_DIR / "predictions.csv",
        figures_dir: Path = FIGURES_DIR,
        batch_size: int = 64,
):
    """
    Run inference using a registered model version.

    Args:
        model_name (str): Registered model name (e.g., CNN).
        version (int): Version number to load from registry.
        metadata_path (Path): Input metadata CSV.
        mfcc_dir (Path): Directory containing MFCC .npy files.
        output_csv (Path): Output CSV with predictions.
        figures_dir (Path): Where to save confusion matrix and ROC/PR curves.
    """
    predict(model_name, version, metadata_path, mfcc_dir, output_csv, figures_dir, batch_size)


if __name__ == "__main__":
    app()
