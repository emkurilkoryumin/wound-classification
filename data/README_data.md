# Data layout

The dataset is expected to be stored locally and is not tracked by Git.

Recommended folder layout:

```text
data/raw/
├── class_1/
│   ├── image_001.jpg
│   └── ...
├── class_2/
└── class_3/
```

Each subdirectory name is treated as a class label. The training pipeline scans all image files recursively inside each class directory.
