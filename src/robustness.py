from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data_loader import build_dataloaders
from .evaluate import evaluate_loader
from .model import build_model
from .utils import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robustness on perturbed images")
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

    metrics, _, _, _ = evaluate_loader(
        model=model,
        loader=bundle.test_loader,
        device=device,
        class_names=class_names,
        criterion=None,
    )
    print(metrics)


if __name__ == "__main__":
    main()
