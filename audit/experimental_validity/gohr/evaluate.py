"""
Confirmatory evaluation for the Gohr adapter.

This is deliberately SEPARATE from the training-time validation set
used for checkpoint selection and LR scheduling (gohr.train). The EV
statistical comparisons (H-EV-SHUFFLE, H-EV-REPRESENTATION) use
accuracy on a dedicated, sealed confirmatory test set computed here -
not the "max val_acc over training history" figure, because that
figure is itself derived from data used for model selection within the
same run and therefore cannot also serve as an independent confirmatory
measurement. The historical reporting convention is preserved
separately as metadata (gohr.train.TrainingResult.max_val_acc) for
comparability with the documented baseline, per the frozen scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {"accuracy": self.accuracy, "n": self.n}


def evaluate_model(model, X: np.ndarray, Y: np.ndarray, *, batch_size: int = 10000) -> EvaluationResult:
    """
    Accuracy via the same thresholding convention used throughout the
    project (archive/eval.py): Z > 0.5 -> predicted label 1.
    """
    Z = model.predict(X, batch_size=batch_size, verbose=0).flatten()
    Zbin = Z > 0.5
    acc = float(np.mean(Zbin == Y))
    return EvaluationResult(accuracy=acc, n=len(Y))
