"""Training loop for GCNs-Net.

Defaults follow `main-GCN.py`: Adam, batch 1024, L2 on the FC parameters only.
The reference's learning rate of 1e-6 is kept as the documented default even
though the paper text says 0.01 — the two disagree, and the code is what produced
the published numbers.

Model selection is best-by-validation, as CLAUDE.md requires. When no validation
split is given the final epoch is used and the run is marked accordingly, so a
row in the experiment log always says which it was.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import ClassificationResult, aggregate_to_trials, evaluate
from .pipeline import Fold, Samples


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation hyperparameters. Defaults reproduce `main-GCN.py`."""

    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-6  # the code's value; the paper says 0.01
    regularization: float = 1e-3  # L2 coefficient on FC parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_every: int = 1
    early_stopping_patience: int | None = None
    grad_clip: float | None = None
    verbose: bool = True


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_acc: float | None = None
    val_trial_acc: float | None = None
    seconds: float = 0.0


@dataclass
class TrainingHistory:
    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int | None = None
    best_val_acc: float | None = None
    selection: str = "final_epoch"

    def as_dict(self) -> dict:
        return {
            "n_epochs_run": len(self.epochs),
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "selection": self.selection,
            "train_loss": [round(e.train_loss, 6) for e in self.epochs],
            "train_acc": [round(e.train_acc, 6) for e in self.epochs],
            "val_acc": [None if e.val_acc is None else round(e.val_acc, 6)
                        for e in self.epochs],
        }


class Trainer:
    """Trains one model on one fold."""

    def __init__(self, model: nn.Module, cfg: TrainConfig | None = None,
                 n_classes: int = 4) -> None:
        self.cfg = cfg or TrainConfig()
        self.model = model.to(self.cfg.device)
        self.n_classes = n_classes
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _tensors(samples: Samples) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(samples.X), torch.from_numpy(samples.y))

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(logits, target)
        if self.cfg.regularization > 0 and hasattr(self.model, "l2_penalty"):
            loss = loss + self.cfg.regularization * self.model.l2_penalty()
        # Optional sparsity on a learned graph's correction, so the result stays
        # interpretable against the PCC prior instead of drifting to a dense matrix.
        cfg = getattr(self.model, "cfg", None)
        l1 = getattr(cfg, "adjacency_l1", 0.0)
        if l1 > 0 and hasattr(self.model, "adjacency_penalty"):
            loss = loss + l1 * self.model.adjacency_penalty()
        # Optional graph-smoothness prior on an unconstrained spatial operator
        # (models.graph_bandpower). Additive and opt-in: every arm that does not set
        # it is bit-identical to before this hook existed.
        smooth = getattr(cfg, "graph_smoothness", 0.0)
        if smooth > 0 and hasattr(self.model, "graph_penalty"):
            loss = loss + smooth * self.model.graph_penalty()
        return loss

    # --------------------------------------------------------------- training
    def fit(self, fold: Fold, checkpoint_path: Path | str | None = None) -> TrainingHistory:
        X, y = self._tensors(fold.train)
        n = len(X)
        history = TrainingHistory()
        best_state = None
        epochs_since_best = 0
        generator = torch.Generator().manual_seed(0)

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            self.model.train()
            order = torch.randperm(n, generator=generator)
            total_loss = 0.0
            correct = 0

            for start in range(0, n, self.cfg.batch_size):
                idx = order[start:start + self.cfg.batch_size]
                xb = X[idx].to(self.cfg.device, non_blocking=True)
                yb = y[idx].to(self.cfg.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self._loss(logits, yb)
                loss.backward()
                if self.cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

                total_loss += loss.item() * len(idx)
                correct += (logits.argmax(1) == yb).sum().item()

            record = EpochRecord(epoch=epoch, train_loss=total_loss / n,
                                 train_acc=correct / n, seconds=time.time() - t0)

            if fold.val is not None and epoch % self.cfg.eval_every == 0:
                val = self.evaluate_samples(fold.val)
                record.val_acc = val["timepoint"].accuracy
                record.val_trial_acc = val["trial"].accuracy
                if history.best_val_acc is None or record.val_acc > history.best_val_acc:
                    history.best_val_acc = record.val_acc
                    history.best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1

            history.epochs.append(record)
            if self.cfg.verbose:
                msg = (f"epoch {epoch:3d}  loss {record.train_loss:.4f}  "
                       f"train {record.train_acc:.4f}")
                if record.val_acc is not None:
                    msg += f"  val {record.val_acc:.4f}  val-trial {record.val_trial_acc:.4f}"
                msg += f"  {record.seconds:.1f}s"
                print(msg, flush=True)

            if (self.cfg.early_stopping_patience is not None
                    and epochs_since_best >= self.cfg.early_stopping_patience):
                if self.cfg.verbose:
                    print(f"early stop at epoch {epoch} "
                          f"(no val improvement for {epochs_since_best})", flush=True)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            history.selection = "best_val"

        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": self.model.state_dict(),
                        "best_epoch": history.best_epoch,
                        "selection": history.selection}, path)

        return history

    # ------------------------------------------------------------- evaluation
    @torch.no_grad()
    def predict_logits(self, samples: Samples) -> np.ndarray:
        self.model.eval()
        X, _ = self._tensors(samples)
        out = []
        for start in range(0, len(X), self.cfg.batch_size):
            xb = X[start:start + self.cfg.batch_size].to(self.cfg.device)
            out.append(self.model(xb).cpu().numpy())
        return np.concatenate(out, axis=0)

    def evaluate_samples(self, samples: Samples) -> dict[str, ClassificationResult]:
        """Timepoint-level and trial-level metrics for one split."""
        logits = self.predict_logits(samples)
        timepoint = evaluate(samples.y, logits.argmax(axis=1), self.n_classes)

        trials, trial_pred = aggregate_to_trials(logits, samples.trial_id)
        # One true label per trial; every timepoint of a trial shares it, so take
        # the label of each trial's first occurrence in sorted order.
        order = np.argsort(samples.trial_id, kind="stable")
        starts = np.searchsorted(samples.trial_id[order], trials)
        trial_true = samples.y[order][starts]
        trial = evaluate(trial_true, trial_pred, self.n_classes)

        return {"timepoint": timepoint, "trial": trial}
