# tests/test_model_dataloaders.py

from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import torch

from ai4health.modeling.dataloaders import MFCCDataset, create_dataloader

TEST_MFCC_DIR = Path("tests/data/processed/mfccs")
TEST_META_PATH = Path("tests/data/processed/metadata_clean.csv")

@pytest.fixture(scope="module")
def mock_metadata():
    TEST_MFCC_DIR.mkdir(parents=True, exist_ok=True)

    # Create fake MFCC
    mfcc = np.random.rand(40, 157).astype(np.float32)
    np.save(TEST_MFCC_DIR / "mock_1.npy", mfcc)

    # Create metadata
    df = pd.DataFrame([
        {
            "id": "mock_1",
            "status": "COVID-19",
            "has_valid_wav": True
        }
    ])
    TEST_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEST_META_PATH, index=False)
    return df

def test_dataset_loading(mock_metadata):
    dataset = MFCCDataset(
        metadata_path=TEST_META_PATH,
        mfcc_dir=TEST_MFCC_DIR,
        label_column="status"
    )
    assert len(dataset) == 1

    X, y = dataset[0]
    assert isinstance(X, torch.Tensor)
    assert X.shape == (40, 157)
    assert y.item() == 1  # COVID-19 => 1

def test_dataloader(mock_metadata):
    loader = create_dataloader(
        metadata_path=TEST_META_PATH,
        mfcc_dir=TEST_MFCC_DIR,
        batch_size=2,
        shuffle=False
    )
    for batch_x, batch_y in loader:
        assert batch_x.shape[1:] == torch.Size([40, 157])
        assert batch_y.shape[0] <= 2
        break
