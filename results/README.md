# Results

## experiment_summary.csv

Full metrics for all 6 canonical pruning experiments. Key columns:

| Column | Description |
|--------|-------------|
| `exp_id` | Experiment identifier |
| `approach` | `local` or `global` |
| `backbone` | `cnn` (5-layer CNN_V1) or `vgg16_bn` |
| `dataset` | `celeba` or `cifar10` |
| `acc_before` / `acc_after` | Top-1 accuracy before and after pruning |
| `overall_drop_pp` | Overall accuracy drop in percentage points |
| `max_class_drop_pp` | Worst per-class drop in percentage points |
| `channels_removed_pct` | Percentage of convolutional channels removed |
| `param_ratio_total` | Ratio of total parameters after/before |
| `mac_ratio_conv` | Ratio of convolutional MACs after/before |
| `latency_speedup` | End-to-end latency improvement factor |
| `latency_ms_before` / `latency_ms_after` | Per-sample inference time (ms) |
| `optimization_overhead_s` | Total wall-clock time of pruning run |

## figures/

| File | Description |
|------|-------------|
| `fig_overall_acc.png` | Accuracy before vs. after for all 6 experiments |
| `fig_per_class_delta_all.png` | Per-class accuracy delta; guard-rail budget usage |
| `fig_params_vs_speedup.png` | Parameter compression vs. latency speedup scatter |
| `fig_mac_ratio_vs_speedup.png` | MAC ratio vs. latency speedup — shows that targeting compute-intensive layers yields disproportionate speedup |

## Key Numbers at a Glance

| Experiment | Params Pruned | MAC Ratio | Speedup |
|---|---|---|---|
| Local HC, 5-CNN, CelebA | 46.0% | 0.85 | 1.04× |
| Local HC, VGG16, CelebA | 30.5% | 0.75 | 1.03× |
| Global, 5-CNN, CelebA | **70.1%** | 0.61 | 1.28× |
| Global, VGG16, CelebA | 45.6% | 0.72 | 1.21× |
| Global, 5-CNN, CIFAR-10 | 7.9% | 0.70 | **2.05×** |
| Global, VGG16, CIFAR-10 | 5.2% | 0.68 | 1.52× |

The CIFAR-10 5-CNN result (2.05× speedup at only 7.9% parameter reduction) demonstrates that
pruning concentrated in compute-intensive blocks (B4: 75% channels removed) can yield
disproportionate latency gains even with modest overall compression.
