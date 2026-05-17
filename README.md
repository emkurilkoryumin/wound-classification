# wound-classification

Baseline repository for a research prototype that classifies wound states from clinical images and supports Grad-CAM-based interpretation.

This repository was assembled from the current NIR materials and is intended to provide a public code reference for the project mentioned in the reports. The dataset itself is not committed to GitHub.

## Repository layout

```text
wound-classification/
├── configs/
├── data/
├── docs/reports/
├── notebooks/
├── outputs/
└── src/
```

## Implemented baseline

- Folder-based image dataset loading with stratified train/validation/test split.
- ResNet18 baseline for multi-class image classification.
- Training loop with checkpointing, CSV history, and learning-curve plots.
- Evaluation script with accuracy and F1 metrics.
- Grad-CAM script for visual interpretation of predictions.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare the dataset in the following layout:

```text
data/raw/
├── class_1/
├── class_2/
└── class_3/
```

4. Adjust `configs/baseline_resnet.yaml` if needed.
5. Train the baseline model:

```bash
python -m src.train --config configs/baseline_resnet.yaml
```

6. Evaluate the saved checkpoint:

```bash
python -m src.evaluate \
  --config configs/baseline_resnet.yaml \
  --checkpoint outputs/checkpoints/baseline_resnet18.pt
```

7. Build a Grad-CAM overlay for a single image:

```bash
python -m src.interpret \
  --config configs/baseline_resnet.yaml \
  --checkpoint outputs/checkpoints/baseline_resnet18.pt \
  --image path/to/example.jpg
```

## Notes

- `docs/reports/` contains the current PDF materials for the NIR stages.
- `outputs/` is kept in the repository as a target location for metrics, figures, and checkpoints.
- If the final report uses a GitHub URL, update it to the public URL of this repository.
