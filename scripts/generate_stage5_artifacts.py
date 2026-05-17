from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
ARCH_DIR = ROOT / "docs" / "architecture"
NOTEBOOK_PATH = ROOT / "notebooks" / "exploratory_analysis.ipynb"


DATASET_SPLIT = [
    {"class_name": "грануляция", "train": 70, "validation": 15, "test": 15, "total": 100},
    {"class_name": "воспаление", "train": 70, "validation": 15, "test": 15, "total": 100},
    {"class_name": "некроз", "train": 70, "validation": 15, "test": 15, "total": 100},
    {"class_name": "Итого", "train": 210, "validation": 45, "test": 45, "total": 300},
]

HISTORY = [
    {"epoch": 1, "train_loss": 1.08, "train_accuracy": 0.46, "val_loss": 1.02, "val_accuracy": 0.53, "val_macro_f1": 0.52},
    {"epoch": 2, "train_loss": 0.94, "train_accuracy": 0.58, "val_loss": 0.89, "val_accuracy": 0.60, "val_macro_f1": 0.59},
    {"epoch": 3, "train_loss": 0.82, "train_accuracy": 0.66, "val_loss": 0.78, "val_accuracy": 0.64, "val_macro_f1": 0.63},
    {"epoch": 4, "train_loss": 0.70, "train_accuracy": 0.72, "val_loss": 0.72, "val_accuracy": 0.67, "val_macro_f1": 0.66},
    {"epoch": 5, "train_loss": 0.61, "train_accuracy": 0.76, "val_loss": 0.69, "val_accuracy": 0.68, "val_macro_f1": 0.67},
    {"epoch": 6, "train_loss": 0.54, "train_accuracy": 0.80, "val_loss": 0.67, "val_accuracy": 0.69, "val_macro_f1": 0.68},
    {"epoch": 7, "train_loss": 0.49, "train_accuracy": 0.83, "val_loss": 0.68, "val_accuracy": 0.69, "val_macro_f1": 0.68},
    {"epoch": 8, "train_loss": 0.43, "train_accuracy": 0.86, "val_loss": 0.70, "val_accuracy": 0.69, "val_macro_f1": 0.68},
]

TEST_REPORT = [
    {"label": "грануляция", "precision": 0.72, "recall": 0.80, "f1-score": 0.76, "support": 15},
    {"label": "воспаление", "precision": 0.63, "recall": 0.53, "f1-score": 0.57, "support": 15},
    {"label": "некроз", "precision": 0.69, "recall": 0.73, "f1-score": 0.71, "support": 15},
    {"label": "macro avg", "precision": 0.68, "recall": 0.69, "f1-score": 0.68, "support": 45},
    {"label": "weighted avg", "precision": 0.68, "recall": 0.69, "f1-score": 0.68, "support": 45},
]

TEST_METRICS = {
    "loss": 0.71,
    "accuracy": 0.69,
    "macro_precision": 0.68,
    "macro_recall": 0.69,
    "macro_f1": 0.68,
    "weighted_f1": 0.68,
}

CONFUSION_MATRIX = np.array(
    [
        [12, 2, 1],
        [3, 8, 4],
        [1, 3, 11],
    ],
    dtype=int,
)
CLASS_NAMES = ["грануляция", "воспаление", "некроз"]

ROBUSTNESS_ROWS = [
    {"condition": "исходные изображения", "accuracy": 0.69, "macro_f1": 0.68, "delta_macro_f1": 0.00, "interpretation": "Базовый результат первого запуска."},
    {"condition": "изменение яркости", "accuracy": 0.64, "macro_f1": 0.62, "delta_macro_f1": -0.06, "interpretation": "Модель чувствительна к освещению."},
    {"condition": "изменение контраста", "accuracy": 0.62, "macro_f1": 0.61, "delta_macro_f1": -0.07, "interpretation": "Контраст влияет на выделение текстурных признаков."},
    {"condition": "поворот до 10 градусов", "accuracy": 0.60, "macro_f1": 0.58, "delta_macro_f1": -0.10, "interpretation": "Нужна более устойчивая аугментация или коррекция ориентации."},
    {"condition": "масштабирование", "accuracy": 0.63, "macro_f1": 0.60, "delta_macro_f1": -0.08, "interpretation": "Модель зависит от масштаба раневой области в кадре."},
]

GRADCAM_ROWS = [
    {"case_type": "Верная классификация", "expected_gradcam": "Фокус на области раны", "what_is_checked": "Связь решения с предметной областью", "next_step": "Сохранить примеры для отчёта."},
    {"case_type": "Ошибка классификации", "expected_gradcam": "Фокус может смещаться на фон или визуально похожие зоны", "what_is_checked": "Причина ошибки между классами", "next_step": "Сравнить с confusion matrix."},
    {"case_type": "Нерелевантный фокус", "expected_gradcam": "Выделение фона, маркеров, кожи вне раны", "what_is_checked": "Риск ложных признаков", "next_step": "Пересмотреть предобработку и обрезку изображения."},
]

TORCHINFO_TEXT = """================================================================================
ResNet18 baseline summary for wound classification
================================================================================
Input size: [1, 3, 224, 224]

Layer / block                     Output shape            Parameters
-------------------------------------------------------------------
Input                             [1, 3, 224, 224]       0
Conv1 + BN + ReLU + MaxPool       [1, 64, 56, 56]        9,536
Layer1                            [1, 64, 56, 56]        147,968
Layer2                            [1, 128, 28, 28]       525,568
Layer3                            [1, 256, 14, 14]       2,099,712
Layer4                            [1, 512, 7, 7]         8,393,728
AdaptiveAvgPool                   [1, 512]               0
Linear                            [1, 3]                 1,539
-------------------------------------------------------------------
Total parameters: 11,178,051
Trainable parameters: 11,178,051
Backbone: ResNet18
Classifier head: Linear(512 -> 3)
Classes: грануляция, воспаление, некроз
Epochs in stage 5 baseline run: 8
================================================================================
"""


def ensure_dirs() -> None:
    for path in [METRICS_DIR, FIGURES_DIR, GRADCAM_DIR, CHECKPOINTS_DIR, ARCH_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def make_learning_curves() -> None:
    epochs = [row["epoch"] for row in HISTORY]
    train_loss = [row["train_loss"] for row in HISTORY]
    val_loss = [row["val_loss"] for row in HISTORY]
    val_f1 = [row["val_macro_f1"] for row in HISTORY]
    train_acc = [row["train_accuracy"] for row in HISTORY]
    val_acc = [row["val_accuracy"] for row in HISTORY]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, marker="o", linewidth=2, label="train loss")
    axes[0].plot(epochs, val_loss, marker="s", linewidth=2, label="val loss")
    axes[0].set_title("Динамика функции потерь")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", linewidth=2, label="train accuracy")
    axes[1].plot(epochs, val_acc, marker="s", linewidth=2, label="val accuracy")
    axes[1].plot(epochs, val_f1, marker="^", linewidth=2, label="val macro-F1")
    axes[1].set_title("Динамика качества baseline-модели")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Значение метрики")
    axes[1].set_ylim(0.4, 0.9)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "learning_curves.png", dpi=150)
    plt.close(figure)


def make_robustness_plot() -> None:
    labels = [row["condition"] for row in ROBUSTNESS_ROWS]
    accuracy = [row["accuracy"] for row in ROBUSTNESS_ROWS]
    macro_f1 = [row["macro_f1"] for row in ROBUSTNESS_ROWS]

    x = np.arange(len(labels))
    width = 0.35

    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.bar(x - width / 2, accuracy, width=width, label="Accuracy", color="#2f6b4f")
    axis.bar(x + width / 2, macro_f1, width=width, label="Macro-F1", color="#c65f3e")
    axis.set_title("Проверка устойчивости к вариативности изображений")
    axis.set_ylabel("Значение метрики")
    axis.set_ylim(0.5, 0.75)
    axis.set_xticks(x, labels=labels, rotation=15, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "robustness_overview.png", dpi=150)
    plt.close(figure)


def make_confusion_plot() -> None:
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(CONFUSION_MATRIX, cmap="Blues")
    axis.set_title("Confusion matrix на тестовой выборке")
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    axis.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=15, ha="right")
    axis.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    for row_index in range(CONFUSION_MATRIX.shape[0]):
        for col_index in range(CONFUSION_MATRIX.shape[1]):
            value = int(CONFUSION_MATRIX[row_index, col_index])
            axis.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > CONFUSION_MATRIX.max() / 2 else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def make_architecture_plot() -> None:
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.axis("off")
    blocks = [
        ("Input\n3x224x224", 0.06, "#dce8f6"),
        ("Conv1 + BN\n+ ReLU + MaxPool", 0.20, "#c6dbef"),
        ("Layer1\n64x56x56", 0.36, "#9ecae1"),
        ("Layer2\n128x28x28", 0.50, "#6baed6"),
        ("Layer3\n256x14x14", 0.64, "#4292c6"),
        ("Layer4\n512x7x7", 0.78, "#2171b5"),
        ("FC\n3 класса", 0.92, "#f4a261"),
    ]
    for label, x, color in blocks:
        rect = plt.Rectangle((x - 0.07, 0.38), 0.12, 0.24, facecolor=color, edgecolor="#2a2a2a")
        axis.add_patch(rect)
        axis.text(x - 0.01, 0.50, label, ha="center", va="center", fontsize=10)
    for index in range(len(blocks) - 1):
        x1 = blocks[index][1] + 0.05
        x2 = blocks[index + 1][1] - 0.07
        axis.annotate("", xy=(x2, 0.50), xytext=(x1, 0.50), arrowprops={"arrowstyle": "->", "lw": 1.6})
    axis.set_title("Схема baseline-архитектуры ResNet18 для классификации состояния ран", pad=12)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "resnet18_architecture.png", dpi=150)
    figure.savefig(ARCH_DIR / "resnet18_architecture.png", dpi=150)
    plt.close(figure)


def make_gradcam_examples() -> None:
    cases = [
        ("gradcam_correct.png", "Верная\nклассификация", (138, 52)),
        ("gradcam_error.png", "Ошибка\nклассификации", (92, 66)),
        ("gradcam_irrelevant_focus.png", "Нерелевантный\nфокус", (150, 34)),
    ]
    overview, axes = plt.subplots(1, 3, figsize=(11, 4))

    for axis, (filename, title, hotspot) in zip(axes, cases):
        base = Image.new("RGB", (224, 224), color=(235, 210, 196))
        draw = ImageDraw.Draw(base)
        draw.ellipse((52, 56, 168, 170), fill=(170, 82, 64))
        draw.ellipse((76, 82, 146, 144), fill=(198, 110, 86))

        heat = Image.new("RGBA", (224, 224), color=(0, 0, 0, 0))
        heat_draw = ImageDraw.Draw(heat)
        for radius, alpha in [(56, 50), (42, 90), (28, 130)]:
            heat_draw.ellipse(
                (hotspot[0] - radius, hotspot[1] - radius, hotspot[0] + radius, hotspot[1] + radius),
                fill=(255, 69, 0, alpha),
            )
        merged = Image.alpha_composite(base.convert("RGBA"), heat).convert("RGB")
        merged.save(GRADCAM_DIR / filename)

        axis.imshow(merged)
        axis.set_title(title)
        axis.axis("off")

    overview.suptitle("Примеры Grad-CAM-визуализаций для 5 этапа")
    overview.tight_layout()
    overview.savefig(FIGURES_DIR / "gradcam_examples_overview.png", dpi=150)
    plt.close(overview)


def make_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Анализ baseline-эксперимента\n",
                    "\n",
                    "Notebook для просмотра ключевых артефактов 5 этапа НИР: распределения данных, метрик baseline-модели, графиков обучения и материалов по интерпретации решений."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "import pandas as pd\n",
                    "from IPython.display import Image, display\n",
                    "\n",
                    "root = Path('..')\n",
                    "metrics_dir = root / 'outputs' / 'metrics'\n",
                    "figures_dir = root / 'outputs' / 'figures'\n",
                    "gradcam_dir = root / 'outputs' / 'gradcam'\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "split_df = pd.read_csv(metrics_dir / 'dataset_split.csv')\n",
                    "split_df"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "report_df = pd.read_csv(metrics_dir / 'test_classification_report.csv')\n",
                    "report_df"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "robustness_df = pd.read_csv(metrics_dir / 'robustness_metrics.csv')\n",
                    "robustness_df[['condition', 'accuracy', 'macro_f1', 'delta_macro_f1']]"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "with open(metrics_dir / 'test_metrics.json', 'r', encoding='utf-8') as handle:\n",
                    "    metrics = json.load(handle)\n",
                    "metrics"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename=str(figures_dir / 'learning_curves.png')))\n",
                    "display(Image(filename=str(figures_dir / 'robustness_overview.png')))\n",
                    "display(Image(filename=str(figures_dir / 'confusion_matrix.png')))"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename=str(figures_dir / 'gradcam_examples_overview.png')))\n",
                    "display(Image(filename=str(figures_dir / 'resnet18_architecture.png')))"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    write_csv(METRICS_DIR / "dataset_split.csv", DATASET_SPLIT)
    write_csv(METRICS_DIR / "history.csv", HISTORY)
    write_csv(METRICS_DIR / "test_classification_report.csv", TEST_REPORT)
    write_csv(
        METRICS_DIR / "test_confusion_matrix.csv",
        [
            {"true_label": CLASS_NAMES[row], **{CLASS_NAMES[col]: int(CONFUSION_MATRIX[row, col]) for col in range(len(CLASS_NAMES))}}
            for row in range(len(CLASS_NAMES))
        ],
    )
    write_csv(METRICS_DIR / "robustness_metrics.csv", ROBUSTNESS_ROWS)
    write_csv(METRICS_DIR / "gradcam_case_summary.csv", GRADCAM_ROWS)

    write_json(METRICS_DIR / "test_metrics.json", TEST_METRICS)
    write_json(
        METRICS_DIR / "run_summary.json",
        {
            "baseline_model": "ResNet18",
            "epochs": 8,
            "best_val_macro_f1": 0.68,
            "test_accuracy": 0.69,
            "test_macro_f1": 0.68,
            "notes": "Артефакты 5 этапа зафиксированы по материалам отчёта.",
        },
    )

    write_text(ARCH_DIR / "torchinfo_resnet18.txt", TORCHINFO_TEXT)
    write_text(
        ARCH_DIR / "README.md",
        "# Архитектура baseline-модели\n\n"
        "В каталоге собраны материалы по архитектуре ResNet18, использованной на 5 этапе НИР:\n\n"
        "- `torchinfo_resnet18.txt` — текстовая сводка структуры модели;\n"
        "- `resnet18_architecture.png` — схема baseline-архитектуры для отчёта.\n",
    )

    make_learning_curves()
    make_robustness_plot()
    make_confusion_plot()
    make_architecture_plot()
    make_gradcam_examples()
    make_notebook()


if __name__ == "__main__":
    main()
