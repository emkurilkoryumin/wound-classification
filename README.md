# wound-classification

Репозиторий с baseline-прототипом для исследования задачи классификации состояния ран по медицинским изображениям с поддержкой интерпретации решений через Grad-CAM.

Репозиторий собран на основе текущих материалов НИР и предназначен как публичная кодовая ссылка для проекта, указанного в отчётах. Сам набор данных в GitHub не загружается.

## Структура репозитория

```text
wound-classification/
├── configs/
├── data/
├── docs/literature/
├── docs/reports/
├── notebooks/
├── outputs/
└── src/
```

## Что реализовано

- загрузка изображений из структуры каталогов с разбиением на train/validation/test;
- baseline-модель ResNet18 для многоклассовой классификации изображений;
- цикл обучения с сохранением checkpoint, CSV-истории и графиков обучения;
- скрипт оценки с расчётом accuracy и F1-метрик;
- скрипт Grad-CAM для визуальной интерпретации предсказаний.

## Быстрый запуск

1. Создайте и активируйте виртуальное окружение.
2. Установите зависимости:

```bash
pip install -r requirements.txt
```

3. Подготовьте датасет в следующем формате:

```text
data/raw/
├── грануляция/
├── воспаление/
└── некроз/
```

4. При необходимости измените параметры в `configs/baseline_resnet.yaml`.
5. Запустите обучение baseline-модели:

```bash
python -m src.train --config configs/baseline_resnet.yaml
```

6. Оцените сохранённый checkpoint:

```bash
python -m src.evaluate \
  --config configs/baseline_resnet.yaml \
  --checkpoint outputs/checkpoints/baseline_resnet18.pt
```

7. Постройте Grad-CAM-визуализацию для одного изображения:

```bash
python -m src.interpret \
  --config configs/baseline_resnet.yaml \
  --checkpoint outputs/checkpoints/baseline_resnet18.pt \
  --image path/to/example.jpg
```

## Примечания

- В `docs/reports/` лежат текущие PDF-материалы по этапам НИР.
- В `docs/literature/` собраны графики, таблицы и сканы, связанные с литературным обзором 3 этапа.
- Каталог `outputs/` содержит зафиксированные артефакты 5 этапа: таблицы, графики, сводные метрики и материалы по интерпретации.
- Если в итоговом отчёте указана ссылка на GitHub, она должна вести на публичный адрес этого репозитория.
