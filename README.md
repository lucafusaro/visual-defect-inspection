# Visual Defect Inspection MVP

An end-to-end visual anomaly detection project for industrial images.
The MVP uses the `bottle` category from MVTec AD and classifies each image as
`normal` or `defective`.

The focus is ML engineering: a reproducible data split, leakage-free threshold
calibration, saved inference artifacts, a CLI, a minimal FastAPI service, tests,
and a Docker image.

## Approach

```text
image -> ResNet18 preprocessing -> 512-d embedding -> k-NN anomaly score -> threshold -> prediction
```

- Backbone: pretrained `torchvision.models.resnet18` with
  `ResNet18_Weights.DEFAULT`.
- Feature extractor: the final classification layer is replaced with `Identity`.
- Reference set: normal training embeddings only.
- Score: mean Euclidean distance to the `k=5` nearest normal embeddings.
- Threshold: 95th percentile of normal validation scores.

The test set is kept separate while selecting the scoring method and calibrating
the threshold.

## Result

On the MVTec AD `bottle` test split, the selected k-NN method achieved:

| Metric | Result |
| --- | --- |
| Accuracy | 96% |
| Defective images detected | 61 / 63 |
| Normal images correctly accepted | 19 / 20 |

The detailed exploration and comparison with centroid distance are available in
`notebooks/03_anomaly_scoring.ipynb`.

## Project Structure

```text
src/visual_inspection/
  data.py        Dataset index and train/validation split
  features.py    ResNet18 feature extraction
  scoring.py     Centroid, k-NN, threshold, and prediction helpers
  artifacts.py   Save and load inference artifacts
  inference.py   Image and feature-level inference
  cli.py         Command-line prediction entry point
  api.py         FastAPI service
tests/           Unit and API tests
notebooks/       Exploration and evaluation
```

## Setup

Use Python 3.11 or newer.

```powershell
pip install -e ".[dev]"
```

Download MVTec AD and place the `bottle` category under `data/bottle`:

```text
data/bottle/
  train/good/*.png
  test/good/*.png
  test/<defect_type>/*.png
```

Run `notebooks/03_anomaly_scoring.ipynb` from top to bottom to create the
inference artifact at `artifacts/bottle_knn_model.pt`.

## CLI

Run a prediction from the repository root:

```powershell
python -m visual_inspection.cli predict `
  --image data/bottle/test/good/000.png `
  --artifact artifacts/bottle_knn_model.pt
```

Example output:

```json
{
  "prediction": "normal",
  "anomaly_score": 3.91,
  "threshold": 4.56
}
```

## API

Start the local service from the repository root:

```powershell
uvicorn visual_inspection.api:app --reload
```

- `GET /health` checks that the service started correctly.
- `POST /predict` accepts a multipart image upload and returns the prediction.

Interactive documentation is available at `http://127.0.0.1:8000/docs`.

## Docker

The Docker image uses CPU-only PyTorch to avoid unnecessary CUDA dependencies.
The artifact must exist locally before building the image.

```powershell
docker build -t visual-defect-inspection-api .
docker run --rm -p 8000:8000 visual-defect-inspection-api
```

Then open `http://127.0.0.1:8000/docs`.

## Quality Checks

```powershell
pytest -v
ruff check .
```

## Continuous Integration

GitHub Actions runs the test suite and Ruff on every push and pull request.

## Current Scope

- One MVTec AD category: `bottle`.
- Image-level anomaly classification only.
- No model fine-tuning, segmentation, localization, heatmaps, database, or UI.
