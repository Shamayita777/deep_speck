"""
AutoND CE2 adapter: genuine-pair data, trajectory reconstruction, frozen run data, and the
theory-consistency capability over one run under one condition.

Target-level concerns (pinned upstream, checkpoint binding, inference) live in
``targets.autond``. This module holds only what is specific to CE2:

* genuine-pair generation with the construction of upstream ``main.make_train_data`` (Y = 1);
* trajectory reconstruction with upstream ``expand_key`` / ``enc_one_round`` (gates E and F);
* ``FrozenRunData``: read-only arrays with a content hash fixed at creation;
* ``AutoNDCE2Adapter``: implements ``core.capabilities.TheoryConsistencyCapability`` and nothing
  else; one view per (run, condition), all views sharing one ``RunContext``.

The CE1 adapter (``adapters/autond.py``) is not used or modified.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ....targets.autond import model_binding as MB
from ....targets.autond.upstream import Upstream
from . import theory
from .design import DESIGN


class AutoNDCE2Error(RuntimeError):
    pass


def content_sha256(arrays: dict) -> str:
    """Hash of named arrays: name, dtype, shape and C-order bytes, in sorted name order."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def delta_bits(up: Upstream) -> np.ndarray:
    return up.delta_bits(int(DESIGN["cipher"]["input_difference"], 16))


def delta_words(up: Upstream) -> tuple[int, int]:
    return up.delta_words(int(DESIGN["cipher"]["input_difference"], 16))


# ------------------------------------------------------------------ run data


@dataclass(frozen=True)
class FrozenRunData:
    """Immutable genuine-pair dataset of one run. Arrays are read-only; the hash is fixed at creation."""
    arrays: dict
    sha256: str = field(default="")

    @classmethod
    def create(cls, arrays: dict) -> "FrozenRunData":
        frozen = {}
        for k, v in arrays.items():
            a = np.array(v, copy=True)
            a.setflags(write=False)
            frozen[k] = a
        return cls(arrays=frozen, sha256=content_sha256(frozen))

    def verify(self) -> None:
        if content_sha256(self.arrays) != self.sha256:
            raise AutoNDCE2Error("run dataset content hash changed")

    def __getitem__(self, key):
        return self.arrays[key]

    @property
    def n(self) -> int:
        return int(self.arrays["target"].shape[0])


def model_input(up: Upstream, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Upstream model input: hstack of convert_to_binary([x, y]) for C0 and C1 (64 uint8 bits)."""
    conv = up.cipher.convert_to_binary
    return np.hstack([conv([c0[:, 0], c0[:, 1]]), conv([c1[:, 0], c1[:, 1]])])


def generate_genuine_pairs(up: Upstream, n: int, *, rounds: int | None = None,
                           randbytes: Callable[[int], bytes] = os.urandom) -> dict:
    """Genuine pairs with the construction of upstream make_train_data (Y = 1 class), plus the
    reconstructed trajectory and gate E/F evidence."""
    c = up.cipher
    rounds = rounds or DESIGN["cipher"]["rounds"]
    keys0 = (np.frombuffer(randbytes(n * c.key_bits), dtype=np.uint8) & 1).reshape(n, c.key_bits)
    pt0 = (np.frombuffer(randbytes(n * c.plain_bits), dtype=np.uint8) & 1).reshape(n, c.plain_bits)
    keys1 = keys0 ^ 0
    pt1 = pt0 ^ delta_bits(up)
    C0 = c.encrypt(pt0, keys0, rounds)
    C1 = c.encrypt(pt1, keys1, rounds)
    arrays = {"key_words": c.convert_from_binary(keys0),
              "pt0": c.convert_from_binary(pt0), "pt1": c.convert_from_binary(pt1),
              "c0": c.convert_from_binary(C0), "c1": c.convert_from_binary(C1)}
    traj, gates = trajectory_and_weights(up, arrays, rounds)
    arrays.update(traj)
    same = bool(np.array_equal(model_input(up, arrays["c0"], arrays["c1"]), np.hstack([C0, C1])))
    gates["F"]["checks"]["model_input_equals_upstream_hstack"] = same
    gates["F"]["passed"] = gates["F"]["passed"] and same
    return {"arrays": arrays, "gates": gates}


def trajectory_and_weights(up: Upstream, a: dict, rounds: int) -> tuple[dict, dict]:
    """Reconstruct both trajectories with upstream expand_key / enc_one_round and compute the
    per-round factors by the closed form and the automaton (gates E and F)."""
    c = up.cipher
    n = a["pt0"].shape[0]
    ks = c.expand_key(a["key_words"].T, rounds)
    x0, y0 = a["pt0"][:, 0].copy(), a["pt0"][:, 1].copy()
    x1, y1 = a["pt1"][:, 0].copy(), a["pt1"][:, 1].copy()
    dx = np.zeros((n, rounds + 1), dtype=np.uint16)
    dy = np.zeros((n, rounds + 1), dtype=np.uint16)
    dx[:, 0], dy[:, 0] = x0 ^ x1, y0 ^ y1
    mask = (1 << c.word_size) - 1
    add_ok = np.ones(n, dtype=bool)
    lin_ok = np.ones(n, dtype=bool)
    for r in range(rounds):
        s0 = (c.ror(x0, c.ALPHA()).astype(np.int64) + y0) & mask
        s1 = (c.ror(x1, c.ALPHA()).astype(np.int64) + y1) & mask
        x0, y0 = c.enc_one_round((x0, y0), ks[r])
        x1, y1 = c.enc_one_round((x1, y1), ks[r])
        dx[:, r + 1], dy[:, r + 1] = x0 ^ x1, y0 ^ y1
        add_ok &= (s0 ^ s1) == dx[:, r + 1]
        lin_ok &= (c.rol(dy[:, r], c.BETA()) ^ dx[:, r + 1]) == dy[:, r + 1]
    final_ok = (np.array_equal(np.stack([x0, y0], 1), a["c0"]) and
                np.array_equal(np.stack([x1, y1], 1), a["c1"]))
    dxw, dyw = delta_words(up)
    init_ok = bool(np.all(dx[:, 0] == dxw) and np.all(dy[:, 0] == dyw))

    w = c.word_size
    alpha = c.ror(dx[:, :-1], c.ALPHA()).astype(np.int64)
    beta = dy[:, :-1].astype(np.int64)
    gamma = dx[:, 1:].astype(np.int64)
    lm = np.stack([theory.lm_weight(alpha[:, r], beta[:, r], gamma[:, r], w) for r in range(rounds)], 1)
    au = np.stack([theory.weight_from_count(theory.carry_count(alpha[:, r], beta[:, r], gamma[:, r], w), w)
                   for r in range(rounds)], 1)
    e_equal = bool(np.array_equal(lm, au))
    e_positive = bool(np.all(lm >= 0))
    gates = {
        "E": {"passed": e_equal and e_positive, "lm_equals_automaton": e_equal,
              "all_factors_positive": e_positive, "n_mismatch": int(np.count_nonzero(lm != au)),
              "n_impossible": int(np.count_nonzero(lm < 0))},
        "F": {"passed": final_ok and init_ok and bool(add_ok.all()) and bool(lin_ok.all()),
              "checks": {"final_state_equals_upstream_encrypt": final_ok, "initial_difference": init_ok,
                         "addition_output_difference_is_next_dx": bool(add_ok.all()),
                         "dy_linear_layer": bool(lin_ok.all())}},
    }
    target = theory.log2_target(lm) if e_positive else np.zeros(n, dtype=np.int64)
    return ({"dx": dx, "dy": dy, "weights": lm.astype(np.uint8) if e_positive else lm.astype(np.int16),
             "target": target.astype(np.int16)}, gates)


def random_pairs_input(n: int, randbytes: Callable[[int], bytes] = os.urandom) -> np.ndarray:
    """Upstream negative class: 64 uniform input bits per sample."""
    return (np.frombuffer(randbytes(n * 64), dtype=np.uint8) & 1).reshape(n, 64)


# --------------------------------------------------------- capability adapter


@dataclass
class RunContext:
    """Everything one run's three measurements share."""
    data: FrozenRunData
    published_model: Any
    twin_model: Any
    permutation: np.ndarray
    model_input: np.ndarray
    scores: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def score(self, role: str) -> np.ndarray:
        if role not in self.scores:
            model = self.published_model if role == "published" else self.twin_model
            s = MB.predict(model, self.model_input)
            s.setflags(write=False)
            self.scores[role] = s
        return self.scores[role]


class AutoNDCE2Adapter:
    """Theory-consistency capability for one run under one condition."""

    ROLES = {"primary": "published", "permuted_target": "published", "untrained_twin": "twin"}

    def __init__(self, context: RunContext, condition: str):
        if condition not in self.ROLES:
            raise ValueError(f"unknown CE2 condition {condition!r}")
        self.context = context
        self.condition = condition

    def _log(self, method: str):
        self.context.calls.append({"condition": self.condition, "method": method,
                                   "dataset_sha256": self.context.data.sha256})

    def load(self) -> Any:
        self._log("load")
        return (self.context.published_model if self.ROLES[self.condition] == "published"
                else self.context.twin_model)

    def generate_theory_dataset(self) -> FrozenRunData:
        self.context.data.verify()
        self._log("generate_theory_dataset")
        return self.context.data

    def compute_theoretical_reference(self, dataset: FrozenRunData) -> np.ndarray:
        self._check_dataset(dataset)
        self._log("compute_theoretical_reference")
        t = dataset["target"].astype(np.float64)
        if self.condition == "permuted_target":
            t = t[self.context.permutation]
        t.setflags(write=False)
        return t

    def compute_model_predictions(self, model: Any, dataset: FrozenRunData) -> np.ndarray:
        self._check_dataset(dataset)
        role = self.ROLES[self.condition]
        expected = self.context.published_model if role == "published" else self.context.twin_model
        if model is not expected:
            raise AutoNDCE2Error("compute_model_predictions received a model not bound to this condition")
        self._log("compute_model_predictions")
        return self.context.score(role)

    def _check_dataset(self, dataset):
        if dataset is not self.context.data:
            raise AutoNDCE2Error("a measurement received a dataset other than the run's frozen dataset")
        dataset.verify()

    @property
    def name(self) -> str:
        return f"AutoND-CE2[{self.condition}]"
