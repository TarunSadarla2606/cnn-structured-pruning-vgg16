"""
hybrid_crossover.py
--------------------
Standalone implementation of the Hybrid Crossover (Tier 2) filter merging
strategy for structured CNN pruning.

Core idea (Tarun Sadarla):
    Standard pruning ("Tier 3 / Standard Drop") identifies two redundant
    filters A and B, then discards the weaker one — permanently losing any
    unique information it encoded.

    Hybrid Crossover instead synthesizes a NEW filter F_new whose weights
    are solved via gradient descent to approximate a hybrid target feature
    map — the spatial union of the most salient activations from both
    parent filters A and B.

    Two channels compress into one, but the synthesized filter retains
    representational capacity from both parents.

Algorithm:
    1. Generate feature maps FM_A and FM_B using calibration data.
    2. Construct Hybrid Target = element-wise max(FM_A, FM_B).
    3. Solve for F_new = argmin_W || Conv(X, W) - Target_Hybrid ||^2
       via a small regression (gradient descent on W alone).
    4. Replace both filters A and B in the weight tensor with F_new.
    5. Update the subsequent layer's input channel count accordingly.

Empirical results vs. Standard Drop (VGG16-BN / CelebA):
    - Hybrid:   92.79% final acc | 372 channels pruned | ~2154s runtime
    - Standard: 92.42% final acc | 196 channels pruned | ~4060s runtime
    → ~2x more compression, higher accuracy, ~2x faster optimization.

Usage:
    merger = HybridCrossoverMerger(model, calibration_loader, device)
    success = merger.merge_pair(layer_path, keep_idx, drop_idx)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
import numpy as np


class HybridCrossoverMerger:
    """
    Implements Hybrid Crossover merging for a pair of convolutional filters.

    For a given layer and a candidate pair (keep_idx, drop_idx):
      - Collects activation maps for both filters on calibration data
      - Synthesizes a hybrid target (spatial union of peak activations)
      - Regresses a new filter weight to match the hybrid target
      - Surgically replaces both filters with the single synthesized filter
      - Updates the downstream layer's input channels accordingly

    Args:
        model: The CNN model (with ChannelGate modules or raw conv layers)
        calib_loader: DataLoader for calibration data (no labels needed)
        device: torch.device
        regression_lr: Learning rate for the regression step
        regression_steps: Number of gradient steps for weight synthesis
        fusion_mode: How to combine parent maps. 'max' (default) takes
                     element-wise maximum. 'weighted' uses activation-
                     magnitude-weighted blend.
    """

    def __init__(
        self,
        model: nn.Module,
        calib_loader,
        device: torch.device,
        regression_lr: float = 1e-3,
        regression_steps: int = 50,
        fusion_mode: str = "max",
        max_calib_batches: int = 8,
    ):
        self.model = model
        self.calib_loader = calib_loader
        self.device = device
        self.regression_lr = regression_lr
        self.regression_steps = regression_steps
        self.fusion_mode = fusion_mode
        self.max_calib_batches = max_calib_batches

        self._hooks = {}
        self._activations = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def merge_pair(
        self,
        conv_path: str,
        keep_idx: int,
        drop_idx: int,
        next_conv_path: Optional[str] = None,
        verbose: bool = True,
    ) -> bool:
        """
        Merge two filters at conv_path by synthesizing a hybrid replacement.

        Args:
            conv_path: Dot-separated path to the Conv2d layer (e.g. 'features.16')
            keep_idx: Index of the filter to keep (higher activation RMS)
            drop_idx: Index of the filter to drop (lower activation RMS)
            next_conv_path: Path to the downstream Conv2d whose in-channels
                            must be updated. If None, auto-detected.
            verbose: Print progress.

        Returns:
            True if merge succeeded, False if regression diverged.
        """
        conv = self._get_module(conv_path)
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"{conv_path} is not a Conv2d layer")

        # 1. Collect activation maps for both parent filters
        FM_keep, FM_drop = self._collect_activation_pair(conv_path, keep_idx, drop_idx)

        # 2. Synthesize hybrid target
        target = self._build_hybrid_target(FM_keep, FM_drop)

        # 3. Regress new filter weights
        F_new = self._regress_filter(conv_path, keep_idx, target, verbose=verbose)
        if F_new is None:
            return False  # regression diverged

        # 4. Apply surgery: replace both filters with F_new
        self._apply_filter_surgery(conv, keep_idx, drop_idx, F_new)

        # 5. Update downstream layer input channels
        next_conv = self._find_next_conv(conv_path, next_conv_path)
        if next_conv is not None:
            self._update_downstream_input(next_conv, keep_idx, drop_idx)

        # 6. Clean up any BN layer
        self._update_bn_after_merge(conv_path, keep_idx, drop_idx)

        if verbose:
            n_before = conv.out_channels + 1  # before surgery
            print(
                f"  [HybridCrossover] {conv_path}: merged ch{drop_idx}→ch{keep_idx} "
                f"| channels: {n_before} → {conv.out_channels}"
            )
        return True

    # ------------------------------------------------------------------ #
    #  Step 1: Collect activation maps                                     #
    # ------------------------------------------------------------------ #

    def _collect_activation_pair(
        self, conv_path: str, keep_idx: int, drop_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run calibration batches through the model, collect post-conv
        activation maps for keep_idx and drop_idx channels.

        Returns:
            FM_keep, FM_drop — tensors of shape (N, H, W)
        """
        maps_keep, maps_drop = [], []

        handle = self._register_hook(conv_path)
        self.model.eval()
        with torch.no_grad():
            for i, (x, *_) in enumerate(self.calib_loader):
                if i >= self.max_calib_batches:
                    break
                x = x.to(self.device)
                self.model(x)
                act = self._activations.get(conv_path)  # (B, C, H, W)
                if act is None:
                    continue
                maps_keep.append(act[:, keep_idx, :, :].cpu())
                maps_drop.append(act[:, drop_idx, :, :].cpu())

        handle.remove()
        del self._activations[conv_path]

        FM_keep = torch.cat(maps_keep, dim=0)  # (N, H, W)
        FM_drop = torch.cat(maps_drop, dim=0)
        return FM_keep, FM_drop

    def _register_hook(self, conv_path: str):
        module = self._get_module(conv_path)

        def hook(m, inp, out):
            self._activations[conv_path] = out.detach()

        return module.register_forward_hook(hook)

    # ------------------------------------------------------------------ #
    #  Step 2: Build hybrid target                                         #
    # ------------------------------------------------------------------ #

    def _build_hybrid_target(
        self, FM_keep: torch.Tensor, FM_drop: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct target feature map preserving peak activations from both parents.

        'max' mode:   Target[n,i,j] = max(FM_keep[n,i,j], FM_drop[n,i,j])
        'weighted':   Target = alpha*FM_keep + (1-alpha)*FM_drop
                      where alpha = |FM_keep| / (|FM_keep| + |FM_drop| + eps)
        """
        if self.fusion_mode == "max":
            return torch.maximum(FM_keep, FM_drop)

        elif self.fusion_mode == "weighted":
            sal_k = FM_keep.abs().mean(dim=[1, 2], keepdim=True)
            sal_d = FM_drop.abs().mean(dim=[1, 2], keepdim=True)
            alpha = sal_k / (sal_k + sal_d + 1e-8)
            return alpha * FM_keep + (1 - alpha) * FM_drop

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    # ------------------------------------------------------------------ #
    #  Step 3: Regress new filter weights                                  #
    # ------------------------------------------------------------------ #

    def _regress_filter(
        self,
        conv_path: str,
        keep_idx: int,
        target: torch.Tensor,
        verbose: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Solve for F_new = argmin_W || Conv(X, W) - Target_Hybrid ||^2

        Freezes all model weights; only optimizes the single filter W.
        Returns F_new weight tensor, or None if regression diverged.
        """
        conv = self._get_module(conv_path)

        # Initialize from the keep filter weights
        F_new = nn.Parameter(conv.weight[keep_idx].clone().to(self.device))
        optimizer = optim.Adam([F_new], lr=self.regression_lr)
        loss_fn = nn.MSELoss()

        self.model.eval()
        prev_loss = float("inf")

        for step in range(self.regression_steps):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            n_batches = 0

            for i, (x, *_) in enumerate(self.calib_loader):
                if i >= self.max_calib_batches:
                    break
                x = x.to(self.device)

                # Forward through all layers up to (but not including) this conv
                with torch.no_grad():
                    feat_in = self._get_input_to_layer(x, conv_path)

                # Compute output of F_new filter on this input
                pred = nn.functional.conv2d(
                    feat_in,
                    F_new.unsqueeze(0),  # (1, C_in, kH, kW)
                    bias=None,
                    stride=conv.stride,
                    padding=conv.padding,
                    dilation=conv.dilation,
                )  # (B, 1, H, W)

                tgt_batch = target[i * x.size(0) : (i + 1) * x.size(0)].to(self.device)
                tgt_batch = tgt_batch.unsqueeze(1)  # (B, 1, H, W)
                min_len = min(pred.size(0), tgt_batch.size(0))
                total_loss = total_loss + loss_fn(pred[:min_len], tgt_batch[:min_len])
                n_batches += 1

            if n_batches == 0:
                break

            avg_loss = total_loss / n_batches
            avg_loss.backward()
            optimizer.step()

            # Divergence check
            if torch.isnan(avg_loss) or avg_loss.item() > prev_loss * 10:
                if verbose:
                    print(f"    [HybridCrossover] Regression diverged at step {step}")
                return None
            prev_loss = avg_loss.item()

        return F_new.detach().cpu()

    # ------------------------------------------------------------------ #
    #  Step 4 & 5: Surgery                                                 #
    # ------------------------------------------------------------------ #

    def _apply_filter_surgery(
        self,
        conv: nn.Conv2d,
        keep_idx: int,
        drop_idx: int,
        F_new: torch.Tensor,
    ):
        """
        Replace the keep_idx filter with F_new and remove the drop_idx filter.
        Updates conv.weight and conv.out_channels in-place.
        """
        W = conv.weight.data  # (C_out, C_in, kH, kW)
        W[keep_idx] = F_new.to(W.device)

        # Remove drop_idx row
        indices = [i for i in range(W.size(0)) if i != drop_idx]
        new_W = W[indices]

        conv.weight = nn.Parameter(new_W)
        conv.out_channels = new_W.size(0)

        if conv.bias is not None:
            b = conv.bias.data
            new_b = b[indices]
            conv.bias = nn.Parameter(new_b)

    def _update_downstream_input(
        self, next_conv: nn.Conv2d, keep_idx: int, drop_idx: int
    ):
        """Remove the drop_idx input channel from the downstream conv."""
        W = next_conv.weight.data  # (C_out, C_in, kH, kW)
        indices = [i for i in range(W.size(1)) if i != drop_idx]
        next_conv.weight = nn.Parameter(W[:, indices, :, :])
        next_conv.in_channels = len(indices)

    def _update_bn_after_merge(
        self, conv_path: str, keep_idx: int, drop_idx: int
    ):
        """
        If a BatchNorm layer follows this conv, remove the drop_idx statistics.
        Assumes BN path is conv_path with last integer incremented by 1.
        """
        # Attempt to find BN by path convention (e.g. features.16 → features.17)
        parts = conv_path.split(".")
        try:
            last = int(parts[-1])
            bn_path = ".".join(parts[:-1] + [str(last + 1)])
            bn = self._get_module(bn_path)
            if not isinstance(bn, nn.BatchNorm2d):
                return
        except (ValueError, AttributeError):
            return

        indices = [i for i in range(bn.num_features) if i != drop_idx]
        for attr in ["weight", "bias", "running_mean", "running_var"]:
            tensor = getattr(bn, attr).data
            setattr(bn, attr, nn.Parameter(tensor[indices])
                    if attr in ["weight", "bias"] else tensor[indices])
        bn.num_features = len(indices)

    # ------------------------------------------------------------------ #
    #  Utilities                                                            #
    # ------------------------------------------------------------------ #

    def _get_module(self, path: str) -> nn.Module:
        m = self.model
        for part in path.split("."):
            m = getattr(m, part)
        return m

    def _find_next_conv(
        self, conv_path: str, next_conv_path: Optional[str]
    ) -> Optional[nn.Conv2d]:
        if next_conv_path is not None:
            return self._get_module(next_conv_path)
        # Auto-detect: increment last index
        parts = conv_path.split(".")
        try:
            last = int(parts[-1])
            for offset in range(1, 5):
                candidate = ".".join(parts[:-1] + [str(last + offset)])
                try:
                    m = self._get_module(candidate)
                    if isinstance(m, nn.Conv2d):
                        return m
                except AttributeError:
                    continue
        except ValueError:
            pass
        return None

    def _get_input_to_layer(self, x: torch.Tensor, conv_path: str) -> torch.Tensor:
        """
        Run forward pass up to (but not including) conv_path and return input tensor.
        Uses a hook on the target layer to capture its input.
        """
        captured = {}

        def pre_hook(m, inp):
            captured["input"] = inp[0].detach()

        module = self._get_module(conv_path)
        handle = module.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            self.model(x)
        handle.remove()
        return captured["input"]
