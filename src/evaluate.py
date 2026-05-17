from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from .data_loader import build_dataloaders
from .model import build_model
from .utils import load_config, resolve_device, save_json, save_rows


def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    criterion: nn.Module | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[int], list[int]]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            if criterion is not None:
                losses.append(float(criterion(logits, labels).item()))

            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    metrics = {
        "loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "accuracy": accuracy_score(targets, predictions) if targets else 0.0,
        "macro_precision": precision_score(targets, predictions, average="macro", zero_division=0)
        if targets
        else 0.0,
        "macro_recall": recall_score(targets, predictions, average="macro", zero_division=0)
        if targets
        else 0.0,
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0)
        if targets
        else 0.0,
        "weighted_f1": f1_score(targets, predictions, average="weighted", zero_division=0)
        if targets
        else 0.0,
    }

    rows: list[dict[str, Any]] = []
    if targets:
        report = classification_report(
            targets,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        for label, values in report.items():
            if isinstance(values, dict):
                row = {"label": label}
                row.update(values)
                rows.append(row)

    return metrics, rows, predictions, targets


def save_confusion_outputs(
    output_dir: str | Path,
    split_name: str,
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
) -> None:
    target_dir = Path(output_dir)
    if not targets:
        return

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    csv_rows: list[dict[str, Any]] = []
    for row_index, row_name in enumerate(class_names):
        row_payload: dict[str, Any] = {"true_label": row_name}
        for col_index, col_name in enumerate(class_names):
            row_payload[col_name] = int(matrix[row_index, col_index])
        csv_rows.append(row_payload)
    save_rows(target_dir / f"{split_name}_confusion_matrix.csv", csv_rows)

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=15, ha="right")
    axis.set_yticks(np.arange(len(class_names)), labels=class_names)
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    axis.set_title("Confusion matrix")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            axis.text(
                col_index,
                row_index,
                str(int(matrix[row_index, col_index])),
                ha="center",
                va="center",
                color="white" if matrix[row_index, col_index] > matrix.max() / 2 else "black",
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(target_dir / f"{split_name}_confusion_matrix.png", dpi=150)
    plt.close(figure)


def save_evaluation_outputs(
    output_dir: str | Path,
    split_name: str,
    metrics: dict[str, float],
    rows: list[dict[str, Any]],
) -> None:
    target_dir = Path(output_dir)
    save_json(target_dir / f"{split_name}_metrics.json", metrics)
    save_rows(target_dir / f"{split_name}_classification_report.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a wound classification checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["device"])
    bundle = build_dataloaders(config)

    class_names = [name for name, _ in sorted(bundle.class_to_idx.items(), key=lambda item: item[1])]
    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = nn.CrossEntropyLoss()
    metrics, rows, predictions, targets = evaluate_loader(
        model=model,
        loader=bundle.test_loader,
        device=device,
        class_names=class_names,
        criterion=criterion,
    )

    metrics_dir = Path(config["output_dir"]) / "metrics"
    save_evaluation_outputs(metrics_dir, "test", metrics, rows)
    save_confusion_outputs(metrics_dir, "test", targets, predictions, class_names)
    print(metrics)


if __name__ == "__main__":
    main()
