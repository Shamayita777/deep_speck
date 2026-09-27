"""
Training wrapper for the Gohr adapter.

Exposes shuffle, optimizer, learning-rate schedule, batch size, and
model seed as explicit parameters (never hard-coded), so that
H-EV-SHUFFLE can vary exactly one of them while every other call site
(EV-BASELINE, EV-NOISE, H-EV-REPRESENTATION) uses the frozen baseline
values from gohr.baseline.

Checkpoint monitor/selection rule is fixed at val_loss / save_best_only
(matching the documented baseline) and is never varied per condition -
per the frozen scope, checkpoint-vs-final-evaluation is an
Implementation Integrity question, not an EV factor.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from framework.failures import FailureReason, detect_failure_from_history
from framework.provenance import sha256_file


def set_seed(seed: int) -> None:
    """
    Identical in effect to archive/train_nets.py::set_seed - seeds
    Python's random, NumPy's global RNG, and TensorFlow's global RNG.
    Must be called BEFORE model construction (weight initialization
    depends on the RNG state at construction time).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def cyclic_lr(num_epochs: int, high_lr: float, low_lr: float):
    """Identical to archive/train_nets.py::cyclic_lr."""
    def res(i):
        return low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    return res


def make_checkpoint(path: str) -> ModelCheckpoint:
    """
    Fixed checkpoint rule: monitor val_loss, save_best_only=True.
    Identical to archive/train_nets.py::make_checkpoint. Never varied
    per EV condition - see module docstring.
    """
    return ModelCheckpoint(path, monitor="val_loss", save_best_only=True)


@dataclass
class TrainingResult:
    final_val_acc: float
    max_val_acc: float
    final_val_loss: float
    n_epochs_completed: int
    checkpoint_path: str
    checkpoint_hash: Optional[str]
    failure_reason: Optional[FailureReason]
    history: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_val_acc": self.final_val_acc,
            "max_val_acc": self.max_val_acc,
            "final_val_loss": self.final_val_loss,
            "n_epochs_completed": self.n_epochs_completed,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
        }


def train_model(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    shuffle: bool,
    lr_high: float,
    lr_low: float,
    lr_period: int,
    optimizer: str,
    checkpoint_path: str,
    timeout_seconds: Optional[float] = None,
) -> TrainingResult:
    """
    Train one model. `shuffle` is the explicit, single manipulated
    factor for H-EV-SHUFFLE; every other parameter is held constant
    across conditions within an experiment.

    Uses model.compile(loss='mse', metrics=['acc']) identically to
    archive/train_nets.py. Does not silently change the loss function,
    optimizer construction, or checkpoint rule.
    """
    import signal
    import time

    model.compile(optimizer=optimizer, loss="mse", metrics=["acc"])
    checkpoint_cb = make_checkpoint(checkpoint_path)
    lr_cb = LearningRateScheduler(cyclic_lr(lr_period, lr_high, lr_low))

    start = time.time()
    failure_reason: Optional[FailureReason] = None
    history: dict[str, list[float]] = {}

    try:
        if timeout_seconds is not None:
            def _handler(signum, frame):
                raise TimeoutError("Training exceeded timeout_seconds")
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(int(timeout_seconds))
        h = model.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=(X_val, Y_val),
            callbacks=[lr_cb, checkpoint_cb],
            verbose=0,
        )
        history = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    except TimeoutError:
        failure_reason = FailureReason.TIMEOUT
    except tf.errors.ResourceExhaustedError:
        failure_reason = FailureReason.OOM
    finally:
        if timeout_seconds is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    if failure_reason is None and history:
        detected = detect_failure_from_history(history)
        if detected is not None:
            failure_reason = detected

    checkpoint_hash = None
    if failure_reason is None:
        if not Path(checkpoint_path).exists():
            failure_reason = FailureReason.MISSING_CHECKPOINT
        else:
            checkpoint_hash = sha256_file(checkpoint_path)

    max_val_acc = float(np.max(history["val_acc"])) if history.get("val_acc") else float("nan")
    final_val_acc = float(history["val_acc"][-1]) if history.get("val_acc") else float("nan")
    final_val_loss = float(history["val_loss"][-1]) if history.get("val_loss") else float("nan")

    return TrainingResult(
        final_val_acc=final_val_acc,
        max_val_acc=max_val_acc,
        final_val_loss=final_val_loss,
        n_epochs_completed=len(history.get("val_acc", [])),
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        failure_reason=failure_reason,
        history=history,
    )
