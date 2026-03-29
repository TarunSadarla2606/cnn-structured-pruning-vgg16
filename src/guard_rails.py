"""
guard_rails.py
--------------
Accuracy guard-rail enforcement for structured CNN pruning.

Both Local HC and Global pruning share this constraint framework.
After each proposed structural change (filter removal or merge),
the pruner evaluates the model and checks two types of constraints:

  1. Overall accuracy drop  ≤ Δ_max_overall
  2. Per-class accuracy drop ≤ Δ_max_class  (for each class c)

If either constraint is violated, the change is rolled back.

The Global approach additionally enforces:
  - Per-step limits (how much can drop in a single pruning step)
  - Cumulative limits (total drop since baseline, tracked running)
  - Soft locks: channels/layers that repeatedly trigger rollbacks are
    temporarily excluded from future shortlists

Dataset-specific limits used in canonical experiments:
  CelebA:  Δ_max_overall = 2 pp,  Δ_max_class = 6 pp
  CIFAR10: Δ_max_overall = 3 pp,  Δ_max_class = 9 pp
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy


@dataclass
class GuardRailConfig:
    """
    Configuration for accuracy guard rails.

    Attributes:
        overall_max_pp: Maximum allowed cumulative overall accuracy drop
                        in percentage points.
        class_max_pp:   Maximum allowed cumulative per-class accuracy drop
                        in percentage points (applied to each class).
        step_overall_max_pp: Maximum allowed per-step overall drop (Global only).
        step_class_max_pp:   Maximum allowed per-step per-class drop (Global only).
        dataset: 'celeba' or 'cifar10' — used for default limit lookup.
    """
    overall_max_pp: float = 2.0
    class_max_pp: float = 6.0
    step_overall_max_pp: float = 1.0
    step_class_max_pp: float = 3.0
    dataset: str = "celeba"

    @classmethod
    def for_celeba(cls) -> "GuardRailConfig":
        return cls(
            overall_max_pp=2.0,
            class_max_pp=6.0,
            step_overall_max_pp=1.0,
            step_class_max_pp=3.0,
            dataset="celeba",
        )

    @classmethod
    def for_cifar10(cls) -> "GuardRailConfig":
        return cls(
            overall_max_pp=3.0,
            class_max_pp=9.0,
            step_overall_max_pp=1.5,
            step_class_max_pp=4.5,
            dataset="cifar10",
        )


@dataclass
class AccuracySnapshot:
    """
    A snapshot of model accuracy at a given pruning step.
    """
    overall: float
    per_class: Dict[int, float]

    def overall_drop_pp(self, baseline: "AccuracySnapshot") -> float:
        return (baseline.overall - self.overall) * 100.0

    def max_class_drop_pp(self, baseline: "AccuracySnapshot") -> float:
        drops = [
            (baseline.per_class.get(c, 0.0) - self.per_class.get(c, 0.0)) * 100.0
            for c in baseline.per_class
        ]
        return max(drops) if drops else 0.0

    def budget_usage_overall(
        self, baseline: "AccuracySnapshot", config: GuardRailConfig
    ) -> float:
        """Fraction of overall accuracy budget consumed (0–1)."""
        return self.overall_drop_pp(baseline) / max(1e-6, config.overall_max_pp)

    def budget_usage_class(
        self, baseline: "AccuracySnapshot", config: GuardRailConfig
    ) -> float:
        """Fraction of per-class budget consumed by the worst class (0–1)."""
        return self.max_class_drop_pp(baseline) / max(1e-6, config.class_max_pp)


class GuardRailChecker:
    """
    Evaluates and enforces accuracy guard rails during pruning.

    Usage:
        checker = GuardRailChecker(model, eval_loader, device, config)
        checker.set_baseline()

        # After a structural change:
        result = checker.check(last_accepted)
        if result.violated:
            # rollback the change
            ...
    """

    def __init__(
        self,
        model: nn.Module,
        eval_loader,
        device: torch.device,
        config: GuardRailConfig,
        num_classes: int = 2,
    ):
        self.model = model
        self.eval_loader = eval_loader
        self.device = device
        self.config = config
        self.num_classes = num_classes
        self.baseline: Optional[AccuracySnapshot] = None

    def set_baseline(self) -> AccuracySnapshot:
        """Evaluate and store the baseline (pre-pruning) accuracy."""
        snap = self.evaluate()
        self.baseline = snap
        print(
            f"[GuardRail] Baseline — overall: {snap.overall:.4f} | "
            f"per-class: { {c: f'{v:.4f}' for c, v in snap.per_class.items()} }"
        )
        return snap

    def evaluate(self) -> AccuracySnapshot:
        """Run full evaluation and return an AccuracySnapshot."""
        self.model.eval()
        correct_total = 0
        total = 0
        correct_per_class: Dict[int, int] = {c: 0 for c in range(self.num_classes)}
        count_per_class: Dict[int, int] = {c: 0 for c in range(self.num_classes)}

        with torch.no_grad():
            for x, y, *_ in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct_total += (preds == y).sum().item()
                total += len(y)
                for c in range(self.num_classes):
                    mask = y == c
                    correct_per_class[c] += (preds[mask] == y[mask]).sum().item()
                    count_per_class[c] += mask.sum().item()

        overall = correct_total / max(1, total)
        per_class = {
            c: correct_per_class[c] / max(1, count_per_class[c])
            for c in range(self.num_classes)
        }
        return AccuracySnapshot(overall=overall, per_class=per_class)

    def check(
        self,
        current: AccuracySnapshot,
        last_accepted: AccuracySnapshot,
        enforce_step: bool = True,
    ) -> "GuardRailResult":
        """
        Check whether current accuracy satisfies guard rails.

        Args:
            current: Accuracy snapshot after the proposed change.
            last_accepted: Accuracy at the last accepted step (for step limits).
            enforce_step: If True, also enforce per-step limits.

        Returns:
            GuardRailResult with violated flag and reason.
        """
        if self.baseline is None:
            raise RuntimeError("Call set_baseline() before check()")

        cumulative_overall_drop = current.overall_drop_pp(self.baseline)
        cumulative_class_drop = current.max_class_drop_pp(self.baseline)
        step_overall_drop = current.overall_drop_pp(last_accepted)
        step_class_drop = current.max_class_drop_pp(last_accepted)

        violations = []

        if cumulative_overall_drop > self.config.overall_max_pp:
            violations.append(
                f"cumulative overall drop {cumulative_overall_drop:.3f} pp "
                f"> limit {self.config.overall_max_pp} pp"
            )

        if cumulative_class_drop > self.config.class_max_pp:
            violations.append(
                f"cumulative per-class drop {cumulative_class_drop:.3f} pp "
                f"> limit {self.config.class_max_pp} pp"
            )

        if enforce_step and step_overall_drop > self.config.step_overall_max_pp:
            violations.append(
                f"step overall drop {step_overall_drop:.3f} pp "
                f"> step limit {self.config.step_overall_max_pp} pp"
            )

        if enforce_step and step_class_drop > self.config.step_class_max_pp:
            violations.append(
                f"step per-class drop {step_class_drop:.3f} pp "
                f"> step limit {self.config.step_class_max_pp} pp"
            )

        return GuardRailResult(
            violated=len(violations) > 0,
            reasons=violations,
            cumulative_overall_drop_pp=cumulative_overall_drop,
            cumulative_class_drop_pp=cumulative_class_drop,
            step_overall_drop_pp=step_overall_drop,
            step_class_drop_pp=step_class_drop,
            budget_overall=current.budget_usage_overall(self.baseline, self.config),
            budget_class=current.budget_usage_class(self.baseline, self.config),
        )


@dataclass
class GuardRailResult:
    """Result of a guard-rail check."""
    violated: bool
    reasons: List[str]
    cumulative_overall_drop_pp: float
    cumulative_class_drop_pp: float
    step_overall_drop_pp: float
    step_class_drop_pp: float
    budget_overall: float  # 0–1 fraction of overall budget used
    budget_class: float    # 0–1 fraction of per-class budget used

    def summary(self) -> str:
        status = "VIOLATED" if self.violated else "OK"
        return (
            f"[GuardRail {status}] "
            f"cumulative: {self.cumulative_overall_drop_pp:.3f} pp overall, "
            f"{self.cumulative_class_drop_pp:.3f} pp max-class | "
            f"budget used: {self.budget_overall:.1%} overall, "
            f"{self.budget_class:.1%} per-class"
            + (f" | VIOLATIONS: {'; '.join(self.reasons)}" if self.violated else "")
        )


# ------------------------------------------------------------------ #
#  Snapshot / Rollback utilities                                        #
# ------------------------------------------------------------------ #

def save_model_snapshot(model: nn.Module) -> dict:
    """Save a deep copy of the model's state dict for rollback."""
    return copy.deepcopy(model.state_dict())


def restore_model_snapshot(model: nn.Module, snapshot: dict):
    """Restore model weights from a previously saved snapshot."""
    model.load_state_dict(snapshot)
