from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .transforms import build_eval_transform, build_train_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_to_idx: dict[str, int]


class ImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def scan_image_folder(root_dir: str | Path) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found in: {root}")

    class_to_idx = {path.name: index for index, path in enumerate(class_dirs)}
    samples: list[tuple[Path, int]] = []

    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))

    if not samples:
        raise FileNotFoundError(f"No image files found in: {root}")

    return samples, class_to_idx


def split_samples(
    samples: list[tuple[Path, int]],
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_split + val_split + test_split must equal 1.0")

    labels = [label for _, label in samples]
    train_samples, temp_samples = _split_with_fallback(
        samples,
        test_size=val_split + test_split,
        labels=labels,
        seed=seed,
    )

    if not temp_samples:
        return train_samples, [], []

    if val_split == 0:
        return train_samples, [], temp_samples

    if test_split == 0:
        return train_samples, temp_samples, []

    temp_labels = [label for _, label in temp_samples]
    val_ratio = val_split / (val_split + test_split)
    val_samples, test_samples = _split_with_fallback(
        temp_samples,
        test_size=1 - val_ratio,
        labels=temp_labels,
        seed=seed,
    )
    return train_samples, val_samples, test_samples


def _split_with_fallback(samples, test_size: float, labels, seed: int):
    try:
        return train_test_split(
            samples,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        return train_test_split(
            samples,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def build_dataloaders(config: dict) -> DatasetBundle:
    data_config = config["data"]
    training_config = config["training"]

    samples, class_to_idx = scan_image_folder(data_config["root_dir"])
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        train_split=data_config["train_split"],
        val_split=data_config["val_split"],
        test_split=data_config["test_split"],
        seed=config["seed"],
    )

    image_size = data_config["image_size"]
    train_dataset = ImageDataset(train_samples, build_train_transform(image_size))
    val_dataset = ImageDataset(val_samples, build_eval_transform(image_size))
    test_dataset = ImageDataset(test_samples, build_eval_transform(image_size))

    batch_size = training_config["batch_size"]
    num_workers = config["num_workers"]
    pin_memory = False

    return DatasetBundle(
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        val_loader=DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        class_to_idx=class_to_idx,
    )


def describe_class_balance(samples: list[tuple[Path, int]]) -> dict[int, int]:
    return dict(Counter(label for _, label in samples))
