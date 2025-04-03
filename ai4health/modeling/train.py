# ai4health/modeling/train.py

from pathlib import Path
from datetime import datetime
import json
import yaml
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
from loguru import logger
import typer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

from ai4health.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from ai4health.modeling.registry import register_model

app = typer.Typer()

# -------------------------------
# Dataset Loader
# -------------------------------

class MFCCDataset(Dataset):
    def __init__(self, metadata_csv: Path, mfcc_dir: Path):
        self.df = pd.read_csv(metadata_csv)
        self.mfcc_dir = mfcc_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mfcc_path = self.mfcc_dir / row["filename"].replace(".wav", ".npy")
        mfcc = np.load(mfcc_path).astype(np.float32)
        mfcc = np.expand_dims(mfcc, axis=0)
        label = np.float32(row["label"])
        return torch.tensor(mfcc), torch.tensor(label)

# -------------------------------
# CNN Model Definition
# -------------------------------

class CNNClassifier(nn.Module):
    def __init__(self, input_shape=(1, 40, 157)):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_dim = self.conv_block(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------------
# Training Function
# -------------------------------

def train(
        metadata_path: Path,
        mfcc_dir: Path,
        model_name: str,
        version: int,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 20,
) -> dict:
    logger.info("Initializing dataset and dataloader...")
    dataset = MFCCDataset(metadata_path, mfcc_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNNClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    df = pd.read_csv(metadata_path)
    pos_weight = torch.tensor([len(df[df.label == 0]) / len(df[df.label == 1])]).to(device)
    logger.info(f"Using pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), y.to(device).unsqueeze(1)
            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logger.success(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    # Save model
    version_dir = MODELS_DIR / model_name / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path.resolve()}")

    # Evaluation
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            probs = model(X)
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    metrics = {
        "final_loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "samples": len(dataset),
        "timestamp": datetime.now().isoformat()
    }

    metrics_path = version_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "model_name": model_name,
        "version": version,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "architecture": "CNNClassifier"
    }
    config_path = version_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    register_model(
        model_name=model_name,
        version=version,
        config=config,
        metrics=metrics,
        model_path=model_path
    )

    return {"version": version, **metrics}

# -------------------------------
# Parameter Sweep Function
# -------------------------------

@app.command()
def sweep(
        metadata_path: Path = INTERIM_DATA_DIR / "metadata_filtered.csv",
        mfcc_dir: Path = PROCESSED_DATA_DIR / "mfcc",
        model_name: str = "CNN",
        base_version: int = 100
):
    """
    Run multiple training experiments with different hyperparameter combinations.
    Automatically picks and logs the best model based on F1-score.
    """
    batch_sizes = [32, 64]
    lrs = [1e-3, 5e-4]
    epoch_counts = [10, 20]

    all_results = []
    for i, (batch_size, lr, epochs) in enumerate(product(batch_sizes, lrs, epoch_counts)):
        version = base_version + i
        logger.info(f"🔁 Running v{version}: batch={batch_size}, lr={lr}, epochs={epochs}")
        result = train(metadata_path, mfcc_dir, model_name, version, batch_size, lr, epochs)
        all_results.append(result)

    best = sorted(all_results, key=lambda x: x["f1"], reverse=True)[0]
    logger.success(f"🏆 Best Model: v{best['version']} | F1: {best['f1']} | AUC: {best['auc']}")

# -------------------------------
# CLI Entrypoint for Single Run
# -------------------------------

@app.command()
def main(
        metadata_path: Path = INTERIM_DATA_DIR / "metadata_filtered.csv",
        mfcc_dir: Path = PROCESSED_DATA_DIR / "mfcc",
        model_name: str = "CNN",
        version: int = 1,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 20,
):
    """
    Train a single CNN model version and register it.
    """
    train(metadata_path, mfcc_dir, model_name, version, batch_size, lr, epochs)


if __name__ == "__main__":
    app()
