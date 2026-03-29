# Data

Raw datasets are **not committed** to this repository. Download from the sources below.

## CelebA — Gender Classification

- **Source:** [Kaggle — CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- **Task:** Binary classification of the `Male` attribute (attribute index 20)
- **Splits used:**
  - Train: 10,000 images
  - Prune reference: 6,000 images (used for BN recalibration during pruning)
  - Evaluation: 19,867 images (natural class proportions, ~57% vs. ~43%)
- **Preprocessing:** Resized to 224×224, normalized with ImageNet statistics

```bash
# Via Kaggle API
kaggle datasets download -d jessicali9530/celeba-dataset -p ./data/celeba --unzip
```

Place at: `data/celeba/`

## CIFAR-10 — Binary (Truck vs. Ship)

- **Source:** [torchvision.datasets.CIFAR10](https://pytorch.org/vision/stable/datasets.html#cifar)
- **Task:** Binary classification: Class 9 (truck) vs. Class 8 (ship)
- **Splits used:**
  - Train: 10,000 images
  - Prune reference: 6,000 images
  - Evaluation: ~2,000 images (balanced by construction)
- **Preprocessing:** Resized to 224×224 to match architecture input, ImageNet normalization

```python
from torchvision.datasets import CIFAR10
ds = CIFAR10(root='./data/cifar10', train=True, download=True)
```

Place at: `data/cifar10/`

## Expected Layout

```
data/
├── celeba/
│   ├── list_attr_celeba.csv
│   ├── list_eval_partition.csv
│   └── img_align_celeba/
│       ├── 000001.jpg
│       └── ...
└── cifar10/
    └── cifar-10-batches-py/
        └── ...
```

## Notes

- All datasets are fully anonymized and cleared for academic use.
- The CelebA dataset requires a Kaggle account for download.
- The CIFAR-10 binary task restricts to classes 8 and 9 only — see
  `experiments/global_cifar10_s2.py` for the exact filtering logic (`CIFAR_CLASSES = (9, 8)`).
