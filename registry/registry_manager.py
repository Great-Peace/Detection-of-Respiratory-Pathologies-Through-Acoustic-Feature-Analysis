import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

REGISTRY_PATH = Path("../models/registry.json")


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)


def register_model(
        model_name: str,
        version: str,
        config_path: Path,
        metrics_path: Path,
        weights_path: Path,
        extra_metadata: Dict[str, Any] = None
):
    registry = load_registry()

    entry = {
        "version": version,
        "registered_at": datetime.utcnow().isoformat(),
        "config": str(config_path),
        "metrics": str(metrics_path),
        "weights": str(weights_path),
    }

    if extra_metadata:
        entry.update(extra_metadata)

    if model_name not in registry:
        registry[model_name] = []

    registry[model_name].append(entry)
    save_registry(registry)
    print(f"✅ Registered {model_name} v{version} successfully.")


if __name__ == "__main__":
    # Example usage
    register_model(
        model_name="CNN",
        version="v1",
        config_path=Path("../models/CNN/v1/config.yaml"),
        metrics_path=Path("../models/CNN/v1/metrics.json"),
        weights_path=Path("../models/CNN/v1/model.pt"),
        extra_metadata={"description": "First baseline CNN model"}
    )
