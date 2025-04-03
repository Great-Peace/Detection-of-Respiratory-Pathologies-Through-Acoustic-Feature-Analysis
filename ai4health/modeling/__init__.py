"""
ai4health.modeling
==================

Modeling utilities including dataloaders, training, and inference routines.
"""

from .dataloaders import MFCCDataset, create_dataloader
from .train import CNNClassifier  # Optional: expose CNN directly
