
---

# 🧠 ai4health

A modular, reproducible AI pipeline for classifying COVID-19 coughs using deep learning (CNN, LSTM, Attention). Built with PyTorch, Librosa, Typer, and the Cookiecutter Data Science v2 structure.

---

## 📁 Project Structure

```
├── LICENSE
├── Makefile                  # 💻 Run `make help` to view CLI options
├── README.md
│
├── data/
│   ├── raw/                  # Original COUGHVID data (.wav/.json)
│   ├── interim/              # Cleaned metadata (e.g., filtered CSV)
│   ├── processed/            # Extracted MFCCs, labeled files, ready for modeling
│   └── external/             # Optional third-party resources
│
├── models/
│   ├── CNN/v1/               # Model version directory with config, metrics, and .pt
│   ├── registry.json         # 🔧 Central model registry
│
├── notebooks/
│   ├── 1.1-extract-metadata.ipynb   # EDA & preprocessing exploration
│
├── reports/
│   └── figures/              # 📊 Visualizations: confusion matrix, ROC, PR, etc.
│
├── ai4health/                # Main project module
│   ├── config.py             # 🔧 Central paths and environment variables
│   ├── dataset.py            # 📦 Metadata parsing, MFCC extraction, CLI
│   ├── features.py           # Optional legacy feature code
│   ├── plots.py              # Visualization utilities
│   ├── modeling/
│   │   ├── dataloaders.py    # PyTorch Datasets & DataLoaders
│   │   ├── train.py          # CNN training logic and CLI
│   │   ├── predict.py        # Inference + evaluation
│   │   └── registry.py       # 🚀 Model registration logic
│   └── __init__.py
│
├── pyproject.toml
├── requirements.txt
└── setup.cfg
```

---

## ⚙️ Setup Instructions

1. **Create Conda environment:**
   ```bash
   make create_environment
   ```

2. **Install dependencies:**
   ```bash
   make requirements
   ```

3. **Verify everything works:**
   ```bash
   make test
   ```

---

## 🛠️ Key Commands (via `make`)

| Action                | Command                       |
|----------------------|-------------------------------|
| Extract metadata      | `make data`                   |
| Custom data build     | `make data-custom SR=22050 DURATION=3 MFCC=30 MIN_CONF=0.5` |
| Run tests             | `make test`                   |
| Format code           | `make format`                 |
| Lint code             | `make lint`                   |

---

## 🧪 Training a Model

Train a CNN baseline and register it automatically:

```bash
python -m ai4health.modeling.train \
    --metadata-path data/processed/metadata_clean.csv \
    --mfcc-dir data/processed/mfccs \
    --model-out-path models/CNN/v1/model.pt \
    --batch-size 64 --epochs 10 --lr 1e-3
```

✅ Upon training, the model is:
- Saved to `models/CNN/v1/model.pt`
- Registered in `models/registry.json`
- Includes `config.yaml` and `metrics.json`

---

## 🔍 Running Inference

```bash
python -m ai4health.modeling.predict \
    --model-path models/CNN/v1/model.pt \
    --metadata-path data/processed/metadata_clean.csv \
    --mfcc-dir data/processed/mfccs \
    --output-csv reports/predictions.csv \
    --figures-dir reports/figures
```

---

## 📚 Model Registry

Use `ai4health.modeling.registry` to:

### Register a model manually:

```bash
python -m ai4health.modeling.registry register-model \
    --model-name CNN \
    --version 2 \
    --config-path models/CNN/v2/config.yaml \
    --metrics-path models/CNN/v2/metrics.json \
    --model-path models/CNN/v2/model.pt
```

### Check the registry:

```bash
cat models/registry.json | jq
```

---

## 🧠 Supported Architectures

- ✅ CNN (Baseline)
- 🧪 CNN-LSTM
- 🧪 Attention-based CNN-LSTM
- 🧪 XGBoost (with statistical features)
- 🧪 Vision Transformers (future)

---

## 📈 Evaluation Outputs

- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`
- `reports/figures/pr_curve.png`

---

