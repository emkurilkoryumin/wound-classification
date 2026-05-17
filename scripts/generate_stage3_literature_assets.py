from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
LIT_DIR = ROOT / "docs" / "literature"
FIGURES_DIR = LIT_DIR / "figures"
TABLES_DIR = LIT_DIR / "tables"


SOURCES = [
    {"ref_id": 1, "year": 2017, "group": "Методологическая база DL", "short_title": "Deep learning in medical image analysis", "role": "Общий обзор DL в медвизуализации", "limitation": "Требует качественных данных и строгих протоколов"},
    {"ref_id": 2, "year": 2022, "group": "Transfer learning", "short_title": "Transfer learning for medical image classification", "role": "Обоснование transfer learning", "limitation": "Есть доменный разрыв между ImageNet и медизображениями"},
    {"ref_id": 3, "year": 2023, "group": "Аугментация данных", "short_title": "Data augmentation for medical imaging", "role": "Обоснование контролируемой аугментации", "limitation": "Не всякая аугментация клинически допустима"},
    {"ref_id": 4, "year": 2016, "group": "Архитектуры CNN", "short_title": "Deep Residual Learning for Image Recognition", "role": "Baseline-архитектура ResNet", "limitation": "Нужна сравнительная оценка, а не выбор по инерции"},
    {"ref_id": 5, "year": 2017, "group": "XAI / интерпретация", "short_title": "Grad-CAM", "role": "Базовый метод визуальной интерпретации", "limitation": "Не даёт полноценного причинного объяснения"},
    {"ref_id": 6, "year": 2023, "group": "XAI / интерпретация", "short_title": "Saliency-based XAI in medical imaging", "role": "Клинические ограничения saliency-map", "limitation": "Визуально убедительно, но не всегда клинически надёжно"},
    {"ref_id": 7, "year": 2026, "group": "XAI / интерпретация", "short_title": "Systematic review of XAI in medical imaging", "role": "Отсутствие стандарта оценки объяснений", "limitation": "Сложно формализовать качество объяснения"},
    {"ref_id": 8, "year": 2022, "group": "Обзор по ранам", "short_title": "AI in Wound Assessment", "role": "Широкий контекст задач анализа ран", "limitation": "Классификация сама по себе может быть слишком узкой"},
    {"ref_id": 9, "year": 2014, "group": "Классические методы", "short_title": "Automated tissue classification framework", "role": "Ранний предметный baseline", "limitation": "Ручные признаки хуже переносятся между доменами"},
    {"ref_id": 10, "year": 2022, "group": "Сегментация тканей", "short_title": "Wound tissue segmentation", "role": "Переход к локальному анализу тканей", "limitation": "Требует более дорогой разметки"},
    {"ref_id": 11, "year": 2022, "group": "Сегментация тканей", "short_title": "Deep learning on mobile devices", "role": "Прикладной мобильный сценарий", "limitation": "Компромисс между точностью и вычислительной ценой"},
    {"ref_id": 12, "year": 2022, "group": "Классификация ран", "short_title": "Pressure injury stage classification", "role": "Прямая близость к задаче классификации", "limitation": "Обычные метрики слабо раскрывают структуру ошибок"},
    {"ref_id": 13, "year": 2022, "group": "Мультимодальные подходы", "short_title": "Multi-modal wound classification", "role": "Показывает пользу дополнительного контекста", "limitation": "Чисто визуальная модель имеет пределы применимости"},
    {"ref_id": 14, "year": 2024, "group": "Мультимодальные подходы", "short_title": "Integrated image and location analysis", "role": "Усиление классификации через локализацию", "limitation": "Усложняет сбор данных и структуру входов"},
    {"ref_id": 15, "year": 2026, "group": "Классификация ран", "short_title": "Hypergranulation detection", "role": "Пример более тонкой дифференциации состояний", "limitation": "Требует ещё более строгой разметки"},
    {"ref_id": 16, "year": 2022, "group": "Обзор по ранам", "short_title": "Survey of wound image analysis", "role": "Систематизация DL-решений для ран", "limitation": "Сильная неоднородность датасетов и метрик"},
    {"ref_id": 17, "year": 2019, "group": "Мобильные решения", "short_title": "Diabetic foot ulcer detection on mobile devices", "role": "Инженерная устойчивость в реальном сценарии", "limitation": "Есть риск снижения переносимости"},
    {"ref_id": 18, "year": 2022, "group": "Мобильные решения", "short_title": "Binary patterns + CNN for diabetic foot ulcers", "role": "Комбинация текстурных признаков и CNN", "limitation": "Привязка к конкретному сценарию"},
    {"ref_id": 19, "year": 2017, "group": "Измерение ран", "short_title": "Automated measurement of pressure injury", "role": "Напоминание о роли количественных признаков", "limitation": "Не решает задачу полной классификации состояния"},
    {"ref_id": 20, "year": 2012, "group": "Общая обработка медизображений", "short_title": "Информационные технологии анализа изображений", "role": "Теоретический фундамент по обработке изображений", "limitation": "Не является прямым аналогом wound-classification"},
    {"ref_id": 21, "year": 2025, "group": "Общая обработка медизображений", "short_title": "Методы цифровой обработки медизображений", "role": "Поддержка блока предобработки и улучшения качества", "limitation": "Фокус шире, чем только раневые изображения"},
]

GROUP_ORDER = [
    "Методологическая база DL",
    "Transfer learning",
    "Аугментация данных",
    "Архитектуры CNN",
    "XAI / интерпретация",
    "Обзор по ранам",
    "Классические методы",
    "Сегментация тканей",
    "Классификация ран",
    "Мультимодальные подходы",
    "Мобильные решения",
    "Измерение ран",
    "Общая обработка медизображений",
]

GAP_SCORES = [
    ("Ограниченный объём и неоднородность данных", 5),
    ("Чувствительность к условиям съёмки", 5),
    ("Недостаточная интерпретируемость", 4),
    ("Межклассовое сходство состояний", 4),
    ("Недостаток единой экспериментальной базы", 4),
]

KEY_CARDS = [
    (1, "DL в медвизуализации", "Даёт общую методологическую рамку"),
    (2, "Transfer learning", "Обосновывает дообучение предобученных CNN"),
    (5, "Grad-CAM", "Добавляет визуальную проверку области внимания модели"),
    (8, "AI для оценки ран", "Показывает, что классификация — часть более широкого конвейера"),
    (12, "Классификация стадий ран", "Прямой предметный аналог baseline-задачи"),
    (16, "Survey по wound analysis", "Фиксирует неоднородность датасетов и метрик"),
]


def ensure_dirs() -> None:
    for path in [LIT_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme() -> None:
    content = """# Материалы по литературному обзору

Каталог связывает 3 этап НИР с кодовым репозиторием.

Состав:

- `tables/source_summary.csv` — сводка по источникам из литературного обзора;
- `figures/source_year_distribution.png` — распределение источников по годам;
- `figures/source_group_distribution.png` — распределение источников по тематическим группам;
- `figures/research_gaps_from_review.png` — карта ограничений, выделенных в 3 этапе;
- `figures/review_structure_map.png` — визуализация трёх опорных блоков обзора и их связи с задачей исследования;
- `figures/key_sources_cards.png` — карточки ключевых публикаций, на которые опирается направление работы;
- `scans/` — PNG-сканы выбранных страниц PDF 3 этапа.
"""
    (LIT_DIR / "README.md").write_text(content, encoding="utf-8")


def plot_year_distribution() -> None:
    years = [row["year"] for row in SOURCES]
    unique_years = sorted(set(years))
    counts = [years.count(year) for year in unique_years]

    figure, axis = plt.subplots(figsize=(10, 4.6))
    axis.bar(unique_years, counts, color="#3a6ea5", width=0.7)
    axis.set_title("Распределение источников из 3 этапа по годам")
    axis.set_xlabel("Год публикации")
    axis.set_ylabel("Число источников")
    axis.set_xticks(unique_years)
    axis.grid(axis="y", alpha=0.25)
    for x, y in zip(unique_years, counts):
        axis.text(x, y + 0.05, str(y), ha="center", va="bottom", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "source_year_distribution.png", dpi=150)
    plt.close(figure)


def plot_group_distribution() -> None:
    group_counts = {group: 0 for group in GROUP_ORDER}
    for row in SOURCES:
        group_counts[row["group"]] += 1

    labels = [group for group in GROUP_ORDER if group_counts[group] > 0]
    values = [group_counts[group] for group in labels]

    figure, axis = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(labels))
    axis.barh(y, values, color="#c65f3e")
    axis.set_yticks(y, labels=labels)
    axis.set_xlabel("Число источников")
    axis.set_title("Тематические группы источников литературного обзора")
    axis.grid(axis="x", alpha=0.25)
    for index, value in enumerate(values):
        axis.text(value + 0.05, index, str(value), va="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "source_group_distribution.png", dpi=150)
    plt.close(figure)


def plot_gaps() -> None:
    labels = [item[0] for item in GAP_SCORES]
    values = [item[1] for item in GAP_SCORES]

    figure, axis = plt.subplots(figsize=(11, 5))
    y = np.arange(len(labels))
    axis.barh(y, values, color="#2f6b4f")
    axis.set_yticks(y, labels=labels)
    axis.set_xlim(0, 5.4)
    axis.set_xlabel("Относительная выраженность в выводах обзора")
    axis.set_title("Ключевые исследовательские пробелы по выводам 3 этапа")
    axis.grid(axis="x", alpha=0.25)
    for index, value in enumerate(values):
        axis.text(value + 0.05, index, str(value), va="center", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "research_gaps_from_review.png", dpi=150)
    plt.close(figure)


def plot_structure_map() -> None:
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    blocks = [
        ("Глубокое обучение\nв медвизуализации\n[1–4]", (0.12, 0.58), "#d9e6f2"),
        ("Анализ изображений ран:\nклассификация, сегментация,\nмультимодальность [8–19]", (0.40, 0.58), "#f4dfc8"),
        ("XAI и интерпретация\nрешений модели\n[5–7]", (0.68, 0.58), "#dcebdc"),
        ("Итоговая исследовательская задача:\nустойчивая и интерпретируемая\nклассификация состояния ран", (0.40, 0.18), "#f5d6d6"),
    ]

    for text, (x, y), color in blocks:
        rect = Rectangle((x, y), 0.2, 0.22, facecolor=color, edgecolor="#333333")
        axis.add_patch(rect)
        axis.text(x + 0.1, y + 0.11, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.22, 0.58), (0.45, 0.40)),
        ((0.50, 0.58), (0.50, 0.40)),
        ((0.78, 0.58), (0.55, 0.40)),
    ]
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=15, linewidth=1.6, color="#444444")
        axis.add_patch(arrow)

    axis.set_title("Структура литературного обзора и связь блоков с задачей исследования", pad=12)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "review_structure_map.png", dpi=150)
    plt.close(figure)


def plot_key_cards() -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (ref_id, title, note) in zip(axes.flat, KEY_CARDS):
        row = next(item for item in SOURCES if item["ref_id"] == ref_id)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        card = Rectangle((0.03, 0.05), 0.94, 0.88, facecolor="#f8f4ef", edgecolor="#4a4a4a", linewidth=1.2)
        axis.add_patch(card)
        axis.text(0.07, 0.84, f"[{ref_id}] {row['year']}", fontsize=10, fontweight="bold", va="top")
        axis.text(0.07, 0.70, title, fontsize=11, fontweight="bold", va="top")
        axis.text(0.07, 0.52, row["short_title"], fontsize=9.5, va="top")
        axis.text(0.07, 0.30, note, fontsize=9.5, va="top")
        axis.text(0.07, 0.12, f"Ограничение: {row['limitation']}", fontsize=8.8, va="bottom")

    figure.suptitle("Ключевые источники, на которые опирается направление работы", fontsize=14)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "key_sources_cards.png", dpi=150)
    plt.close(figure)


def main() -> None:
    ensure_dirs()
    write_readme()
    write_csv(TABLES_DIR / "source_summary.csv", SOURCES)
    plot_year_distribution()
    plot_group_distribution()
    plot_gaps()
    plot_structure_map()
    plot_key_cards()


if __name__ == "__main__":
    main()
