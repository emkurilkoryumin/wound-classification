from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import build_model
from .transforms import build_eval_transform
from .utils import ensure_dirs, load_config, resolve_device, resolve_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Grad-CAM visualization")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to saved checkpoint")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--target-layer", default=None, help="Override target layer name")
    parser.add_argument("--target-class", type=int, default=None, help="Optional class index")
    return parser.parse_args()


def main() -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise SystemExit("Install the grad-cam package before running interpretation.") from exc

    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["device"])

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)

    model = build_model(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    rgb_image = np.asarray(image).astype(np.float32) / 255.0
    transform = build_eval_transform(config["data"]["image_size"])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = int(torch.argmax(logits, dim=1).item())

    target_class = args.target_class if args.target_class is not None else predicted_class
    target_layer_name = args.target_layer or config["interpretation"]["target_layer"]
    target_layer = resolve_module(model, target_layer_name)

    targets = [ClassifierOutputTarget(target_class)]
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    output_dir = Path(config["output_dir"]) / "gradcam"
    ensure_dirs([output_dir])
    output_path = output_dir / f"{image_path.stem}_gradcam.png"
    Image.fromarray(cam_image).save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
