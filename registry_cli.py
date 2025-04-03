import typer
from pathlib import Path
from ai4health.registry_manager import load_registry, save_registry, register_model

app = typer.Typer()

REGISTRY_PATH = Path("models/registry.json")


@app.command()
def add(
        model_name: str = typer.Argument(..., help="Name of the model (e.g., cnn, cnn_lstm)"),
        version: str = typer.Argument(..., help="Version tag (e.g., v1.0, exp-cnn-0423)"),
        config_path: Path = typer.Option(..., help="Path to training config file"),
        metrics_path: Path = typer.Option(..., help="Path to metrics .json or .csv"),
        weights_path: Path = typer.Option(..., help="Path to model weights .pt or .pth file"),
        notes: str = typer.Option("", help="Optional notes or description")
):
    """Register a new model version with metadata."""
    registry = load_registry(REGISTRY_PATH)
    metadata = {
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "weights_path": str(weights_path),
        "notes": notes
    }
    register_model(registry, model_name, version, metadata)
    save_registry(registry, REGISTRY_PATH)
    typer.echo(f"✅ Registered {model_name} version {version}.")


@app.command()
def list():
    """List all registered models and versions."""
    registry = load_registry(REGISTRY_PATH)
    for model_name, versions in registry.items():
        typer.echo(f"\n📦 {model_name}:")
        for version, info in versions.items():
            typer.echo(f"  - {version} | {info.get('timestamp')} | {info.get('notes', '')}")


if __name__ == "__main__":
    app()
