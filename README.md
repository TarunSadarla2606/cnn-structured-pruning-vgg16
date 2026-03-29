# CNN Filter Pruning — Hybrid Crossover Merging vs. Standard Drop

> *Can synthesizing a new filter from two redundant ones outperform simply discarding the weaker one?*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Course](https://img.shields.io/badge/Course-CSCE%205934%20Directed%20Study-blueviolet)]()
[![University](https://img.shields.io/badge/UNT-Computer%20Science-00853E)]()

---

## Overview

This repository presents research from a directed study (CSCE 5934) at the University of North Texas, conducted under the supervision of **Prof. Russel Pears**. The project originated from an observation made by a prior student in Prof. Pears' research group, and was extended into a structured comparative study of CNN filter pruning strategies.

The core research question: when two convolutional filters are found to be redundant, is it better to **drop the weaker one** (standard approach), or to **synthesize a new filter** that captures the salient features of both (the hybrid crossover approach proposed here)?

The study compares two structured pruning paradigms — **Local Hierarchical Clustering (Local HC)** and **Global Scoring** — applied to two CNN architectures and two datasets, all constrained by hard accuracy guard rails. Within the Global framework, this repo specifically introduces and evaluates the **Advanced Tier 2 Hybrid Crossover** merging strategy as a novel contribution.

---

## The Novel Contribution: Hybrid Crossover Merging

Standard structured pruning ("Tier 3 / Standard Drop") identifies a pair of similar, redundant filters and discards the weaker one. Information encoded uniquely by the discarded filter is permanently lost.

The **Hybrid Crossover** method (Tarun Sadarla) proposes a fundamentally different decision: instead of discarding, **synthesize a new filter** whose weights are solved via regression to approximate a hybrid target feature map — the element-wise union of the most salient activations from both parent filters.

### How It Works

Given two candidate filters A and B identified as redundant:

1. **Feature Map Generation** — Run both filters on calibration data to produce activation maps `FM_A` and `FM_B`.
2. **Hybrid Target Synthesis** — Construct `Target_Hybrid` preserving the peak activations from both parents:
   ```
   Target_Hybrid[i,j] = max(FM_A[i,j], FM_B[i,j])   # spatially union of salient regions
   ```
3. **Weight Regression** — Solve for a new filter weight set `F_new` via gradient descent:
   ```
   F_new = argmin_W || Conv(X, W) - Target_Hybrid ||²
   ```
4. **Replacement** — Replace both filters A and B with the single synthesized filter `F_new`.

Two channels compress into one — but the synthesized filter retains representational capacity from both parents, rather than losing whatever the dropped filter uniquely encoded.

### Head-to-Head Results (VGG16-BN / CelebA, identical guard rails)

| Metric | Standard Drop (Teammate) | Hybrid Crossover (Tarun) | Delta |
|--------|--------------------------|--------------------------|-------|
| Final Top-1 Accuracy | 92.42% | **92.79%** | +0.37% |
| Total Channels Pruned | 196 | **372** | +176 (~2×) |
| Optimization Runtime | ~4,060 s | **~2,154 s** | ~2× faster |
| Layer 16 Compression | 35.5% pruned | **68.75%** pruned | +33.2 pp |
| Accuracy Drop at L16 | −0.98% | **−0.87%** | Less drop, more pruning |

The hybrid method pruned **nearly twice as many channels** while achieving **higher final accuracy** and running **twice as fast**. The speed improvement comes from a smoother loss landscape — regression-synthesized filters satisfy accuracy constraints on the first or second attempt, avoiding the expensive rollback loops that plague standard drop when aggressive compression is attempted.

---

## Pruning Paradigms

### Local Hierarchical Clustering (Local HC)

The Local HC approach (Sai Naga Chaithanya Aavula) operates independently within each convolutional layer. For every layer, it computes pairwise cosine similarity between all filter activation maps, then greedily merges the most similar pairs — subject to accuracy guard rails that are checked after each proposed merge. Key properties:

- Layer-local decisions; no cross-layer ranking
- Cosine similarity + correction probability as merging criterion
- Conservative by design; rarely saturates the global accuracy budget
- Simpler implementation; shorter optimization runs

### Global Scoring + ILR

The Global approach (Sai Naga Chaithanya Aavula) maintains a single ranking pool across all prunable layers. Filters are scored using an **ILR (Inside-Layer Ranking)** signal — a weighted fusion of three forward-only statistics:

| Signal | Weight | Description |
|--------|--------|-------------|
| Activation RMS | 0.6 | Post-ReLU activation magnitude (streaming) |
| BN-γ magnitude | 0.4 | Batch normalization scale parameter |
| HRank / Frobenius energy | 0.2 | Feature map energy |

Each signal is per-layer min-max normalized to [0,1], then fused into a single importance score `I_{ℓ,i}`. Filters with smaller `I_{ℓ,i}` are ranked as less important globally. The Global loop then prunes in chunks, checking per-step and cumulative guard rails after each structural change.

The **Hybrid Crossover** merging strategy is integrated into the Global loop's merge decision step, replacing the standard "drop the weaker filter" action with the regression-based synthesis described above.

---

## Experiments

### Architectures

| Architecture | Description |
|---|---|
| 5-Layer CNN (CNN_V1) | Custom 5-block CNN with BatchNorm; 29.6M params baseline |
| VGG16-BN | VGG16 with BatchNorm, ImageNet-pretrained head replaced for binary task |

### Datasets

| Dataset | Task | Evaluation Set | Guard Rails |
|---|---|---|---|
| CelebA | Gender (Male/Female) classification | 19,867 images (natural proportions) | ≤2 pp overall, ≤6 pp per-class |
| CIFAR-10 (binary) | Truck vs. Ship classification | ~2,000 images (balanced) | ≤3 pp overall, ≤9 pp per-class |

### Hard Floor Settings (S2 Configuration)

Minimum surviving channels per block under the canonical S2 floor setting:

| Block | Original Channels | S2 Min Survivors | Max Prunable |
|-------|---|---|---|
| B5 (features.16) | 512 | 160 | 352 (68.8%) |
| B4 (features.12) | 512 | 128 | 384 (75.0%) |
| B3 (features.8)  | 256 | 192 | 64 (25.0%) |
| B2 (features.4)  | 128 | 112 | 16 (12.5%) |
| B1 (features.0)  |  64 |  56 | 8 (12.5%) |

See [`experiments/hard_floor_settings.csv`](experiments/hard_floor_settings.csv) for full S1–S6 floor configurations.

---

## Canonical Results

| Approach | Architecture | Dataset | Acc. Before | Acc. After | Δ overall | Params Pruned | Conv MAC Ratio | Latency Speedup |
|---|---|---|---|---|---|---|---|---|
| Local HC | 5-CNN | CelebA | 94.15% | 92.41% | 1.74 pp | 46.0% | 0.85 | 1.04× |
| Local HC | VGG16 | CelebA | 97.93% | 96.45% | 1.48 pp | 30.5% | 0.75 | 1.03× |
| Global | 5-CNN | CelebA | 94.15% | 92.15% | 1.99 pp | **70.1%** | 0.61 | **1.28×** |
| Global | VGG16 | CelebA | 97.93% | 95.94% | 1.99 pp | 45.6% | 0.72 | 1.21× |
| Global | 5-CNN | CIFAR-10 | 91.45% | 90.25% | 1.20 pp | 7.9% | 0.70 | **2.05×** |
| Global | VGG16 | CIFAR-10 | 97.95% | 95.35% | 2.60 pp | 5.2% | 0.68 | 1.52× |

All runs satisfy their dataset-specific accuracy guard rails.

### Key Findings

**Global pruning consistently achieves higher compression and speedup than Local HC** under the same guard rails. On the 5-CNN with CelebA, Global removes 70% of parameters (vs. 46% for Local HC) and achieves 1.28× speedup (vs. 1.04×). On CIFAR-10, the 5-CNN Global run achieves the study's strongest result: **2.05× latency speedup** with only 1.2 pp accuracy drop — by concentrating pruning on compute-intensive blocks (especially B4, where 75% of channels are removed) while leaving early blocks intact.

**The Hybrid Crossover merging strategy outperforms standard drop** on the same architecture and dataset, achieving ~2× the channel compression with higher final accuracy, by synthesizing filters that preserve information from both parents rather than discarding it.

---

## Result Figures

| Figure | Description |
|--------|-------------|
| [`fig_overall_acc.png`](results/figures/fig_overall_acc.png) | Overall accuracy before vs. after pruning across all six experiments |
| [`fig_per_class_delta_all.png`](results/figures/fig_per_class_delta_all.png) | Per-class accuracy delta — guard-rail budget usage breakdown |
| [`fig_params_vs_speedup.png`](results/figures/fig_params_vs_speedup.png) | Parameter compression vs. latency speedup scatter |
| [`fig_mac_ratio_vs_speedup.png`](results/figures/fig_mac_ratio_vs_speedup.png) | Convolutional MAC ratio vs. latency speedup |

---

## Repository Structure

```
cnn-pruning-filter-merging/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   ├── hybrid_crossover_advanced_merging.ipynb   ← Tarun's Hybrid Crossover implementation
│   └── baseline_pipeline.ipynb                   ← Shared Phase 1–2 baseline pipeline
│
├── experiments/
│   ├── global_celeba_s2.py          ← Full Global scoring pipeline, CelebA (8,694 lines)
│   ├── global_cifar10_s2.py         ← Full Global scoring pipeline, CIFAR-10
│   ├── local_celeba_s2.py           ← Full Local HC pipeline, CelebA (2,966 lines)
│   └── hard_floor_settings.csv      ← S1–S6 floor configurations for all blocks
│
├── src/
│   ├── hybrid_crossover.py          ← Standalone Hybrid Crossover merging module
│   ├── ilr_scoring.py               ← ILR (Inside-Layer Ranking) scoring functions
│   └── guard_rails.py               ← Accuracy guard-rail enforcement utilities
│
├── results/
│   ├── experiment_summary.csv       ← Full metrics for all 6 canonical experiments
│   └── figures/
│       ├── fig_overall_acc.png
│       ├── fig_per_class_delta_all.png
│       ├── fig_params_vs_speedup.png
│       └── fig_mac_ratio_vs_speedup.png
│
├── docs/
│   ├── final_report_ieee.pdf         ← IEEE-format final report (CSCE 5934)
│   ├── weekly-reports/
│   │   ├── directed_study_weekly_log.pdf      ← Week-by-week research log (Weeks 1–8+)
│   │   └── cnn_training_time_reduction_notes.pdf ← Background on pruning alternatives
│   └── strategy-docs/
│       ├── advanced_merging_initial_report.pdf    ← Tarun's first results: Hybrid vs. Standard
│       ├── merge_strategies_analysis.pdf          ← 30-page analysis of 6 merge strategies
│       ├── threshold_design_strategies.pdf        ← 35-page analysis of 7 threshold methods
│       ├── threshold_design_strategies_v2.pdf     ← Revised threshold strategy document
│       └── phase4_configuration_analysis.pdf      ← Phase 4 toggle configuration reference
│
└── data/
    └── README.md                    ← Dataset download instructions
```

---

## Quickstart

```bash
git clone https://github.com/TarunSadarla2606/cnn-pruning-filter-merging.git
cd cnn-pruning-filter-merging
pip install -r requirements.txt
```

**To run the Hybrid Crossover notebook** (Google Colab recommended):
```
Open notebooks/hybrid_crossover_advanced_merging.ipynb
```

**To run a canonical Global experiment** (requires GPU, ~2–12 hours):
```bash
# Set backbone and dataset in CONFIG at top of script
python experiments/global_celeba_s2.py
```

---

## Design Decisions and Engineering Notes

### Why the Hybrid Crossover is Faster

Standard drop, when applied aggressively, frequently violates guard rails and triggers rollback — the algorithm undoes the structural change, soft-locks the affected channels, and searches for an alternative. This is expensive. The Hybrid Crossover avoids this by synthesizing a filter that already satisfies the constraints by construction: the regression target is the spatial union of both parents' strongest activations, which closely approximates the pre-pruning representation. Fewer rollbacks → faster overall runtime.

### Why ILR Outperforms Taylor Saliency in Production

Taylor-based saliency requires gradient computation through the model, which is sensitive to training mode, mixed-precision settings, and `torch.compile` behavior. ILR uses only forward-pass statistics (activation RMS, BN-γ, HRank), making it cheaper, reproducible, and stable across long runs with many architectural changes. In practice, ILR as the sole ranking signal delivered better compression at equivalent accuracy cost compared to Taylor-dominant configurations.

### Guard Rail Architecture

Both approaches share a common guard-rail framework:

```
After each structural change:
  If (baseline_acc - current_acc) > Δ_max_overall  →  ROLLBACK
  If (baseline_class_acc[c] - current_class_acc[c]) > Δ_max_class  →  ROLLBACK
  
CelebA limits: Δ_max_overall = 2 pp, Δ_max_class = 6 pp
CIFAR-10 limits: Δ_max_overall = 3 pp, Δ_max_class = 9 pp
```

The Global approach adds per-step limits (1 pp overall, 3 pp per-class on CelebA) in addition to cumulative limits, and uses soft locks to temporarily exclude channels that repeatedly trigger rollbacks.

### Hard Floors

Each convolutional block has a hard floor — a minimum number of surviving channels below which the block is permanently excluded from further pruning. This prevents catastrophic collapse of any single block and ensures stable inference. The S2 floor setting (used in all canonical runs) leaves 160 channels surviving in B5, 128 in B4, 192 in B3, 112 in B2, and 56 in B1 for the 5-CNN.

---

## Contributions

| Component | Author |
|-----------|--------|
| **Hybrid Crossover (Tier 2) merging strategy** — core novel idea | **Tarun Sadarla** |
| **Merge Strategies analysis** (6 strategies, 30 pages) | **Tarun Sadarla** |
| **Threshold Design Strategies** (7 methods, 35 pages) | **Tarun Sadarla** |
| Global scoring pipeline (`global_*.py`) — ILR, guard rails, benefit-aware pruning | Sai Naga Chaithanya Aavula |
| Local HC pipeline (`local_celeba_s2.py`) — hierarchical clustering pruner | Sai Naga Chaithanya Aavula |
| Shared baseline (Phase 1–2): architectures, data pipeline, training | Both |
| Hard floor configuration design | Both |
| Final IEEE report | Both |

**Supervised by:** Prof. Russel Pears, Department of Computer Science and Engineering, University of North Texas

---

## Background: The Research Origin

### Ashwini Sharma's Prior Work — The Founding Observation

This directed study was directly motivated by findings from **Ashwini Sharma**, a former MS thesis student of Prof. Russel Pears at UNT. In his thesis research (code and thesis available [here](https://drive.google.com/drive/folders/1s6zxkvT8qTnh8kvPToem1MN7Os8ACOEF?usp=sharing)), Ashwini studied a gender classification task using the CelebA dataset on a custom 5-layer CNN. His key finding was striking: **not all feature maps in the final convolutional layer were necessary for the classification task**. Under the specific conditions of his experiment, only a small subset of feature maps contributed meaningfully to the model's predictions — the rest were largely redundant.

This observation opened a natural question: if redundant feature maps exist, can they be identified and removed in a principled, data-driven way — and how much efficiency can be recovered without sacrificing accuracy?

### The Initial Framework — Prof. Pears' Research Overview

The founding document Prof. Pears shared at the start of this directed study (see `docs/weekly-reports/cnn_training_time_reduction_notes.pdf`) laid out the motivation and initial methodology. It identified the key problem with existing approaches to CNN efficiency:

- **Transfer learning** reduces training time but is limited to a handful of standard architectures and requires tuning a hyperparameter (which layer to freeze) that directly trades away its benefits.
- **1×1 convolutional filters** reduce computation but only work in architectures that already use them.
- **Ad-hoc channel/layer restriction** can reduce cost but is entirely arbitrary — there is no principled way to choose how many channels to remove, and no control over the accuracy impact.

The proposed alternative: **remove only those filters that can be statistically proven to be redundant** based on the data itself. The initial method used IoU (Intersection over Union) across pairs of feature maps — after thresholding each map to retain only its top 5% brightest pixels — and a statistical test to determine whether two feature maps were encoding essentially the same information. If so, only one needed to remain.

This IoU-based similarity approach became the intellectual seed from which the entire project grew. Over the course of the directed study (documented week by week in `docs/weekly-reports/directed_study_weekly_log.pdf`), it evolved through: cosine similarity replacing IoU (Week 1), a correction-probability stopping criterion (Week 3), the differentiation of Local HC vs. Global paradigms (Week 4), the introduction of multi-signal ILR scoring, accuracy guard rails, and ultimately the Hybrid Crossover merging strategy introduced in this work.

---

## Future Work

- [ ] Phase 4 fine-tuning: short retraining after pruning to recover accuracy in middle layers (B3, B2) that currently resist compression due to low redundancy
- [ ] Relax guard rails to 4 pp overall to test middle-layer compression feasibility
- [ ] Multi-class extension beyond binary CelebA and CIFAR-10 tasks
- [ ] Hybrid Crossover applied within the Local HC framework (currently only tested in Global)
- [ ] Quantile-adaptive threshold selection (replacing static cosine similarity cutoffs)
- [ ] Free-living validation on real-world data distributions
- [ ] INT8 quantization of pruned models for embedded deployment

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
scipy>=1.11.0
tqdm>=4.65.0
```

---

## Citation

```bibtex
@misc{sadarla2025cnnpruning,
  author       = {Sadarla, Tarun and Aavula, Sai Naga Chaithanya},
  title        = {CNN Filter Pruning: Hybrid Crossover Merging vs. Standard Drop under Accuracy Guard Rails},
  year         = {2025},
  institution  = {University of North Texas, Department of Computer Science and Engineering},
  supervisor   = {Pears, Russel},
  url          = {https://github.com/TarunSadarla2606/cnn-pruning-filter-merging}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*CSCE 5934 Directed Study — University of North Texas, Fall 2025*
*Supervised by Prof. Russel Pears*
