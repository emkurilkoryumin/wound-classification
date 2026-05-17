from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW

from .data_loader import build_dataloaders
from .evaluate import evaluate_loader, save_evaluation_outputs
from .model import build_model
from .utils import ensure_dirs, load_config, resolve_device, save_json, save_rows, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the wound classification baseline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_examples += int(images.size(0))

    average_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def plot_history(history_rows: list[dict], target_path: str | Path) -> None:
    epochs = [row["epoch"] for row in history_rows]
    train_loss = [row["train_loss"] for row in history_rows]
    val_loss = [row["val_loss"] for row in history_rows]
    val_f1 = [row["val_macro_f1"] for row in history_rows]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="val_macro_f1")
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    ensure_dirs(
        [
            output_dir / "checkpoints",
            output_dir / "figures",
            output_dir / "gradcam",
            output_dir / "metrics",
        ]
    )

    device = resolve_device(config["device"])
    bundle = build_dataloaders(config)
    class_names = [name for name, _ in sorted(bundle.class_to_idx.items(), key=lambda item: item[1])]

    model = build_model(
        num_classes=len(class_names),
        pretrained=config["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    checkpoint_path = output_dir / "checkpoints" / config["model"]["checkpoint_name"]
    history: list[dict] = []
    best_macro_f1 = -1.0

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics, _, _, _ = evaluate_loader(
            model=model,
            loader=bundle.val_loader,
            device=device,
            class_names=class_names,
            criterion=criterion,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(history_row)

        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": bundle.class_to_idx,
                    "config": config,
                },
                checkpoint_path,
            )

    save_rows(output_dir / "metrics" / "history.csv", history)
    plot_history(history, output_dir / "figures" / "learning_curves.png")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, test_rows, _, _ = evaluate_loader(
        model=model,
        loader=bundle.test_loader,
        device=device,
        class_names=class_names,
        criterion=criterion,
    )

    save_evaluation_outputs(output_dir / "metrics", "test", test_metrics, test_rows)
    save_json(
        output_dir / "metrics" / "run_summary.json",
        {
            "best_val_macro_f1": best_macro_f1,
            "class_names": class_names,
            "checkpoint": str(checkpoint_path),
        },
    )
    print(test_metrics)


if __name__ == "__main__":
    main()
