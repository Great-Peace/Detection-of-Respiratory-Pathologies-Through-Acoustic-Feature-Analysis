"""
ai4health.modeling.dataloaders
==============================

Reusable and configurable PyTorch dataloaders for MFCC-based modeling.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class MFCCDataset(Dataset):
    """
    PyTorch Dataset for loading MFCC features and labels from metadata CSV.

    Args:
        metadata_path (Path): Path to cleaned metadata CSV file.
        mfcc_dir (Path): Directory where MFCC .npy files are stored.
        label_column (str): Name of column to use as label.
        transform (callable, optional): Optional transform to apply to each MFCC sample.
    """

    def __init__(
            self,
            metadata_path: Path,
            mfcc_dir: Path,
            label_column: str = "status",
            transform: Optional[callable] = None,
    ):
        self.metadata = pd.read_csv(metadata_path)
        self.mfcc_dir = mfcc_dir
        self.label_column = label_column
        self.transform = transform

        # Filter out rows without MFCC file
        self.metadata = self.metadata[self.metadata["has_valid_wav"] == True]
        self.metadata["mfcc_path"] = self.metadata["id"].apply(
            lambda x: self.mfcc_dir / f"{x}.npy"
        )

        self.metadata = self.metadata[self.metadata["mfcc_path"].apply(Path.exists)]

        self.label_map = {
            "COVID-19": 1,
            "healthy": 0,
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        mfcc = np.load(row["mfcc_path"]).astype(np.float32)
        label = self.label_map.get(row[self.label_column], -1)

        if self.transform:
            mfcc = self.transform(mfcc)

        return torch.tensor(mfcc), torch.tensor(label)


def create_dataloader(
        metadata_path: Path,
        mfcc_dir: Path,
        label_column: str = "status",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for the MFCC dataset.

    Args:
        metadata_path (Path): Path to cleaned metadata CSV.
        mfcc_dir (Path): Path to directory with .npy MFCC files.
        label_column (str): Column to use for labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle data.
        num_workers (int): Number of parallel workers for loading data.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    dataset = MFCCDataset(
        metadata_path=metadata_path,
        mfcc_dir=mfcc_dir,
        label_column=label_column,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
