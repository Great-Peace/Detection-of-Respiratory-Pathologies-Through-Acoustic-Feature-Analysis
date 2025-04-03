# ai4health/modeling/registry.py

"""
Model Registry Utility
======================

Handles saving, loading, and managing model metadata in a central registry.

Registry structure (models/registry.json):

{
    "CNN": {
        "v1": {
            "path": "CNN/v1",
            "config": "config.yaml",
            "metrics": "metrics.json",
            "model_file": "model.pt",
            "created": "2025-04-03T01:23:45",
            "notes": "Initial CNN baseline"
        },
        ...
    },
    ...
}
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger

from ai4health.config import MODELS_DIR

# CLI app
app = typer.Typer()

# Path to registry JSON
REGISTRY_PATH = MODELS_DIR / "registry.json"


def load_registry() -> dict:
    """Load the model registry JSON file. Create an empty one if it doesn't exist."""
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps({}, indent=2))
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def get_next_version(model_name: str) -> int:
    registry = load_registry()
    if model_name not in registry:
        return 1
    versions = registry[model_name].keys()
    latest = max([int(v.strip("v")) for v in versions])
    return latest + 1


def save_registry(registry: dict):
    """Write the updated registry dictionary to the registry file."""
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(
        model_name: str,
        version: int,
        config: dict,
        metrics: dict,
        model_path: Optional[Path] = None,
        notes: Optional[str] = None,
):
    """
    Register a new model version with its configuration, evaluation metrics, and optional weights.

    Args:
        model_name (str): Architecture name (e.g., 'CNN', 'attention-CNN-LSTM').
        version (int): Version number (e.g., 1).
        config (dict): Model configuration or hyperparameters.
        metrics (dict): Evaluation metrics.
        model_path (Optional[Path]): Optional path to trained model (.pt file).
        notes (Optional[str]): Optional notes or comments about the model.
    """
    version_str = f"v{version}"
    version_dir = MODELS_DIR / model_name / version_str
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = version_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    # Save metrics
    metrics_path = version_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Copy model file
    if model_path and model_path.exists():
        model_file = "model.pt"
        dest = version_dir / model_file
        if model_path.resolve() != dest.resolve():
            shutil.copy2(model_path, dest)


    # Load existing registry and update entry
    registry = load_registry()
    if model_name not in registry:
        registry[model_name] = {}

    registry[model_name][version_str] = {
        "path": str(version_dir.relative_to(MODELS_DIR)),
        "config": config_path.name,
        "metrics": metrics_path.name,
        "model_file": model_file,
        "created": datetime.utcnow().isoformat(),
        "notes": notes or "",
    }

    save_registry(registry)
    logger.success(f"✅ Model '{model_name}' v{version} registered successfully.")


@app.command("register-model")
def register_model_cli(
        model_name: str,
        version: int,
        config_path: Path,
        metrics_path: Path,
        model_path: Optional[Path] = None,
        notes: Optional[str] = typer.Option(None, help="Optional notes about the model version."),
):
    """
    CLI tool to register a model version and track its config, metrics, and weights.

    Example:
        python -m ai4health.modeling.registry register-model \\
            --model-name CNN \\
            --version 2 \\
            --config-path models/CNN/v2/config.yaml \\
            --metrics-path models/CNN/v2/metrics.json \\
            --model-path models/CNN/v2/model.pt
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(metrics_path) as f:
        metrics = json.load(f)

    register_model(
        model_name=model_name,
        version=version,
        config=config,
        metrics=metrics,
        model_path=model_path,
        notes=notes,
    )

@app.command("list-models")
def list_models():
    """
    List all registered model architectures and their versions.
    """
    registry = load_registry()
    if not registry:
        logger.info("No models registered yet.")
        return

    logger.info("📚 Registered Models:")
    for model_name, versions in registry.items():
        version_list = ', '.join(versions.keys())
        logger.info(f"  • {model_name}: {version_list}")


@app.command("describe-model")
def describe_model(
        model_name: str,
        version: Optional[int] = None,
):
    """
    Print detailed information about a specific model and version.

    Args:
        model_name (str): Name of the model (e.g., "CNN").
        version (int, optional): Version to describe. If omitted, shows all versions.
    """
    registry = load_registry()

    if model_name not in registry:
        logger.error(f"Model '{model_name}' not found in registry.")
        raise typer.Exit(1)

    versions = registry[model_name]

    if version is not None:
        version_key = f"v{version}"
        if version_key not in versions:
            logger.error(f"Version {version} of model '{model_name}' not found.")
            raise typer.Exit(1)
        versions = {version_key: versions[version_key]}  # limit to one

    logger.info(f"📄 Details for model: {model_name}")
    for ver, meta in versions.items():
        logger.info(f"\n🔹 Version: {ver}")
        for key, value in meta.items():
            logger.info(f"   {key}: {value}")


@app.command("get-latest-version")
def get_latest_version(
        model_name: str,
):
    """
    Returns the latest version number of a model.

    Args:
        model_name (str): Name of the model (e.g., 'CNN')
    """
    registry = load_registry()

    if model_name not in registry:
        logger.error(f"Model '{model_name}' not found in registry.")
        raise typer.Exit(1)

    versions = registry[model_name]
    version_nums = [
        int(v.lstrip("v")) for v in versions.keys() if v.startswith("v")
    ]
    latest = max(version_nums)
    logger.success(f"Latest version of '{model_name}' is: v{latest}")


if __name__ == "__main__":
    app()
