
"""
D5 — Training-Data Scaling Audit.

Scientific design
-----------------
D5 estimates how predictive performance changes with training-set size
while holding the dataset-generation mechanism, model architecture,
optimizer/training procedure, epoch budget, and fixed held-out test set
constant.

The strengthened design uses BOTH:

1. Independent replicate datasets:
   each replicate generates one independent maximum-size dataset.

2. Nested prefixes within each replicate:
   the 2.5M, 5M, 7.5M and 10M conditions are the prefixes of that same
   replicate's 10M dataset.

This separates:
    - within-dataset sample-size effects; and
    - between-dataset/model-seed variation.

The model initialization seed is held fixed across sizes within a replicate,
so paired differences within a replicate are directly interpretable as
sample-size effects. Replicates have independent seeds.

A single fixed held-out test set is shared by all conditions.

Persistence
-----------
Each replicate owns one persisted maximum-size dataset. Each replicate/size
condition owns its own resumable Keras checkpoint and JSON state. A run
manifest/configuration hash prevents accidental resumption with changed
experimental parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import hashlib
import json
import math
import os

import numpy as np
from keras.callbacks import Callback
from keras.models import load_model


SCHEMA_VERSION = "2.0"


@dataclass
class D5Observation:
    replicate: int
    training_samples: int
    seed: int
    test_samples: int
    test_accuracy: float
    test_loss: float
    completed_epochs: int
    total_epochs: int


@dataclass
class D5Result:
    experiment: str
    training_sizes: list[int]
    replicates: int
    observations: list[D5Observation]
    summary: dict[str, Any]
    pairwise: dict[str, Any]
    scaling: dict[str, Any]
    manifest: dict[str, Any]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    with tmp.open("wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dataset_paths(root: Path, replicate: int) -> dict[str, Path]:
    d = root / "datasets" / f"replicate_{replicate:02d}"
    return {"train_x": d / "train_x.npy", "train_y": d / "train_y.npy",
            "metadata": d / "metadata.json"}


def test_dataset_paths(root: Path) -> dict[str, Path]:
    d = root / "datasets" / "test"
    return {"test_x": d / "test_x.npy", "test_y": d / "test_y.npy",
            "metadata": d / "metadata.json"}


def condition_paths(root: Path, replicate: int, size: int) -> dict[str, Path]:
    d = root / "runs" / f"replicate_{replicate:02d}" / f"n_{size}"
    return {"checkpoint": d / "latest.keras", "state": d / "state.json",
            "history": d / "history.json"}


def save_test_dataset(root: Path, x: np.ndarray, y: np.ndarray,
                      metadata: Mapping[str, Any]) -> None:
    p = test_dataset_paths(root)
    if p["test_x"].exists() and p["test_y"].exists() and p["metadata"].exists():
        return
    _atomic_npy(p["test_x"], np.asarray(x))
    _atomic_npy(p["test_y"], np.asarray(y))
    meta = dict(metadata)
    meta["test_x_sha256"] = _file_sha256(p["test_x"])
    meta["test_y_sha256"] = _file_sha256(p["test_y"])
    _atomic_json(p["metadata"], meta)


def load_test_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    p = test_dataset_paths(root)
    if not all(v.exists() for v in p.values()):
        raise FileNotFoundError("Incomplete fixed test dataset.")
    return np.load(p["test_x"], mmap_mode="r"), np.load(p["test_y"], mmap_mode="r")


def save_replicate_dataset(root: Path, replicate: int,
                           x: np.ndarray, y: np.ndarray,
                           metadata: Mapping[str, Any]) -> None:
    p = dataset_paths(root, replicate)
    if p["train_x"].exists() and p["train_y"].exists() and p["metadata"].exists():
        return
    if len(x) != len(y):
        raise ValueError("Feature/label counts differ.")
    _atomic_npy(p["train_x"], np.asarray(x))
    _atomic_npy(p["train_y"], np.asarray(y))
    meta = dict(metadata)
    meta.update({
        "replicate": replicate,
        "samples": int(len(x)),
        "train_x_shape": list(x.shape),
        "train_y_shape": list(y.shape),
        "train_x_sha256": _file_sha256(p["train_x"]),
        "train_y_sha256": _file_sha256(p["train_y"]),
    })
    _atomic_json(p["metadata"], meta)


def load_replicate_dataset(root: Path, replicate: int) -> tuple[np.ndarray, np.ndarray]:
    p = dataset_paths(root, replicate)
    if not all(v.exists() for v in p.values()):
        raise FileNotFoundError(f"Incomplete dataset for replicate {replicate}.")
    x = np.load(p["train_x"], mmap_mode="r")
    y = np.load(p["train_y"], mmap_mode="r")
    if len(x) != len(y):
        raise ValueError("Persisted feature/label counts differ.")
    return x, y


def dataset_exists(root: Path, replicate: int) -> bool:
    return all(p.exists() for p in dataset_paths(root, replicate).values())


def test_exists(root: Path) -> bool:
    return all(p.exists() for p in test_dataset_paths(root).values())


class EpochCheckpointCallback(Callback):
    """Persist a complete model and state after every completed epoch."""

    def __init__(self, checkpoint_path: Path, state_path: Path,
                 condition: str, replicate: int, training_samples: int,
                 seed: int, total_epochs: int) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.state_path = state_path
        self.condition = condition
        self.replicate = replicate
        self.training_samples = training_samples
        self.seed = seed
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_path.with_name("checkpoint.tmp.keras")
        self.model.save(tmp)
        os.replace(tmp, self.checkpoint_path)
        state = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete" if epoch + 1 >= self.total_epochs else "in_progress",
            "condition": self.condition,
            "replicate": self.replicate,
            "training_samples": self.training_samples,
            "seed": self.seed,
            "total_epochs": self.total_epochs,
            "completed_epochs": int(epoch + 1),
            "last_epoch_logs": {
                str(k): float(v) for k, v in logs.items()
                if np.isscalar(v) and np.isfinite(float(v))
            },
        }
        _atomic_json(self.state_path, state)


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                       n_boot: int, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo = (1.0 - confidence) / 2.0
    hi = 1.0 - lo
    return float(np.quantile(means, lo)), float(np.quantile(means, hi))


def _paired_t_test(diffs: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t statistic and p-value; scipy is used if available."""
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2:
        return float("nan"), float("nan")
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return (float("inf"), 0.0) if np.mean(diffs) != 0 else (0.0, 1.0)
    t = float(np.mean(diffs) / (sd / math.sqrt(len(diffs))))
    try:
        from scipy.stats import t as t_dist
        p = float(2.0 * t_dist.sf(abs(t), df=len(diffs) - 1))
    except Exception:
        # Conservative normal approximation fallback.
        p = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return t, p


def _holm_adjust(pairs: list[tuple[str, float]], alpha: float) -> dict[str, Any]:
    valid = [(k, p) for k, p in pairs if np.isfinite(p)]
    ordered = sorted(valid, key=lambda z: z[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for i, (k, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        adjusted[k] = running
    return {
        "alpha": alpha,
        "method": "Holm step-down",
        "adjusted_p_values": adjusted,
        "rejections": {k: bool(v < alpha) for k, v in adjusted.items()},
    }


def _aggregate(observations: list[D5Observation], sizes: Sequence[int],
                replicates: int, bootstrap_replicates: int,
                seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_size = {}
    for n in sizes:
        vals = np.array([o.test_accuracy for o in observations if o.training_samples == n], dtype=float)
        losses = np.array([o.test_loss for o in observations if o.training_samples == n], dtype=float)
        lo, hi = _bootstrap_mean_ci(vals, rng, bootstrap_replicates)
        by_size[str(n)] = {
            "n": int(n), "replicates": int(len(vals)),
            "accuracy_mean": float(vals.mean()),
            "accuracy_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "accuracy_ci95_bootstrap": [lo, hi],
            "accuracy_min": float(vals.min()), "accuracy_max": float(vals.max()),
            "loss_mean": float(losses.mean()),
            "loss_sd": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
        }

    # Paired contrasts against the smallest size.
    base_n = int(sizes[0])
    paired = {}
    pvals = []
    lookup = {(o.replicate, o.training_samples): o for o in observations}
    for n in sizes[1:]:
        diffs = np.array([
            lookup[(r, n)].test_accuracy - lookup[(r, base_n)].test_accuracy
            for r in range(1, replicates + 1)
            if (r, n) in lookup and (r, base_n) in lookup
        ], dtype=float)
        t, p = _paired_t_test(diffs)
        lo, hi = _bootstrap_mean_ci(diffs, rng, bootstrap_replicates)
        key = f"{base_n}_vs_{n}"
        paired[key] = {
            "baseline_size": base_n, "comparison_size": int(n),
            "replicates": int(len(diffs)),
            "mean_accuracy_difference": float(diffs.mean()) if len(diffs) else float("nan"),
            "sd_accuracy_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "bootstrap_ci95": [lo, hi],
            "paired_t_statistic": t, "paired_p_value": p,
        }
        pvals.append((key, p))
    paired["multiple_comparison_correction"] = _holm_adjust(pvals, 0.05)

    # Replicate-wise slopes against log10(sample count).
    x = np.log10(np.asarray(sizes, dtype=float))
    slopes = []
    for r in range(1, replicates + 1):
        y = np.array([lookup[(r, n)].test_accuracy for n in sizes if (r, n) in lookup], dtype=float)
        if len(y) == len(sizes):
            slope = float(np.polyfit(x, y, 1)[0])
            slopes.append(slope)
    slopes = np.asarray(slopes, dtype=float)
    slo, shi = _bootstrap_mean_ci(slopes, rng, bootstrap_replicates)
    monotone = []
    for r in range(1, replicates + 1):
        y = [lookup[(r, n)].test_accuracy for n in sizes if (r, n) in lookup]
        if len(y) == len(sizes):
            monotone.append(all(y[i+1] >= y[i] for i in range(len(y)-1)))
    scaling = {
        "model": "ordinary least squares accuracy ~ log10(training_samples), fit separately per replicate",
        "replicate_slopes": slopes.tolist(),
        "mean_slope": float(slopes.mean()) if len(slopes) else float("nan"),
        "mean_slope_bootstrap_ci95": [slo, shi],
        "monotone_non_decreasing_replicates": int(sum(monotone)),
        "replicates_with_complete_curves": int(len(monotone)),
    }
    return by_size, paired, scaling


def run_d5(
    *,
    root: Path,
    training_sizes: Sequence[int],
    test_samples: int,
    replicates: int,
    audit_seed: int,
    total_epochs: int,
    batch_size: int,
    train_dataset_factory: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    test_dataset_factory: Callable[[int, int], tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[int], Any],
    training_callbacks_factory: Callable[[int], list[Any]] | None,
    evaluate_model: Callable[[Any, np.ndarray, np.ndarray], tuple[float, float]],
    experiment_name: str,
    bootstrap_replicates: int = 5000,
    manifest: Mapping[str, Any] | None = None,
) -> D5Result:
    root.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(n) for n in training_sizes})
    if len(sizes) < 2:
        raise ValueError("At least two training sizes are required.")
    if replicates < 2:
        raise ValueError("At least two independent replicates are required.")
    if test_samples < 1 or total_epochs < 1 or batch_size < 1:
        raise ValueError("test_samples, total_epochs and batch_size must be positive.")

    max_size = max(sizes)
    base_manifest = dict(manifest or {})
    base_manifest.update({
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment_name,
        "training_sizes": sizes,
        "replicates": replicates,
        "test_samples": test_samples,
        "total_epochs": total_epochs,
        "batch_size": batch_size,
        "audit_seed": audit_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "design": {
            "independent_replicate_datasets": True,
            "nested_prefixes_within_replicate": True,
            "same_model_seed_across_sizes_within_replicate": True,
            "fixed_test_partition": True,
        },
    })
    config_hash = _sha256_json(base_manifest)
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != config_hash:
            raise RuntimeError(
                "D5 run configuration mismatch. Refusing to resume. "
                "Use a new output directory for a changed experiment."
            )
    else:
        payload = dict(base_manifest)
        payload["config_hash"] = config_hash
        _atomic_json(manifest_path, payload)

    # Fixed test set.
    if not test_exists(root):
        print("Generating fixed held-out test dataset...")
        tx, ty = test_dataset_factory(test_samples, audit_seed)
        save_test_dataset(root, tx, ty, {
            "schema_version": SCHEMA_VERSION,
            "samples": int(test_samples),
            "audit_seed": int(audit_seed),
            "config_hash": config_hash,
        })
        del tx, ty
        print("Fixed test dataset saved.")
    else:
        print("Loading existing fixed test dataset...")
    test_x, test_y = load_test_dataset(root)

    # Independent replicate seeds.
    seq = np.random.SeedSequence(audit_seed)
    child = seq.spawn(replicates)
    replicate_seeds = [
        int(s.generate_state(1, dtype=np.uint32)[0]) for s in child
    ]

    observations: list[D5Observation] = []
    for r in range(1, replicates + 1):
        seed = replicate_seeds[r - 1]
        print()
        print("=" * 72)
        print(f"D5 replicate {r}/{replicates} | model/dataset seed={seed}")
        print("=" * 72)

        if not dataset_exists(root, r):
            print(f"Generating independent {max_size:,}-sample dataset for replicate {r}...")
            rx, ry = train_dataset_factory(max_size, seed)
            save_replicate_dataset(root, r, rx, ry, {
                "schema_version": SCHEMA_VERSION,
                "max_samples": int(max_size),
                "generator_seed": None,
                "audit_seed": int(audit_seed),
                "replicate_seed": int(seed),
                "config_hash": config_hash,
                "nested_prefix_sizes": sizes,
            })
            del rx, ry
            print("Persisted maximum-size dataset.")
        else:
            print("Loading persisted independent replicate dataset.")
        train_x_full, train_y_full = load_replicate_dataset(root, r)

        for n in sizes:
            p = condition_paths(root, r, n)
            condition = f"replicate_{r:02d}/n_{n}"
            state = json.loads(p["state"].read_text(encoding="utf-8")) if p["state"].exists() else None
            if state is not None and state.get("config_hash") not in (None, config_hash):
                raise RuntimeError(f"Configuration mismatch in {condition}.")
            completed = int(state.get("completed_epochs", 0)) if state else 0

            if completed >= total_epochs and p["checkpoint"].exists():
                print(f"Skipping completed {condition}.")
                model = load_model(p["checkpoint"], compile=True)
            else:
                if p["checkpoint"].exists() and completed > 0:
                    print(f"Resuming {condition} from epoch {completed}/{total_epochs}.")
                    model = load_model(p["checkpoint"], compile=True)
                else:
                    print(f"Starting {condition} from epoch 0/{total_epochs}.")
                    model = model_factory(seed)
                    completed = 0

                callbacks = [
                    EpochCheckpointCallback(
                        p["checkpoint"], p["state"], condition, r, n, seed, total_epochs
                    )
                ]
                if training_callbacks_factory is not None:
                    callbacks.extend(training_callbacks_factory(seed))

                # Critical nested-design operation: prefix only.
                x = train_x_full[:n]
                y = train_y_full[:n]
                model.fit(
                    x, y,
                    initial_epoch=completed,
                    epochs=total_epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1,
                )
                del x, y

            loss, acc = evaluate_model(model, test_x, test_y)
            _atomic_json(p["state"], {
                "schema_version": SCHEMA_VERSION,
                "config_hash": config_hash,
                "status": "complete",
                "condition": condition,
                "replicate": r,
                "training_samples": n,
                "seed": seed,
                "total_epochs": total_epochs,
                "completed_epochs": total_epochs,
                "test_loss": float(loss),
                "test_accuracy": float(acc),
            })
            _atomic_json(p["history"], {
                "replicate": r, "training_samples": n,
                "test_loss": float(loss), "test_accuracy": float(acc),
            })
            observations.append(D5Observation(
                replicate=r, training_samples=n, seed=seed,
                test_samples=test_samples, test_accuracy=float(acc),
                test_loss=float(loss), completed_epochs=total_epochs,
                total_epochs=total_epochs,
            ))
            del model

        del train_x_full, train_y_full

    # Reconstruct all completed observations from state files, including resumed sessions.
    observations = []
    for r in range(1, replicates + 1):
        seed = replicate_seeds[r - 1]
        for n in sizes:
            p = condition_paths(root, r, n)
            if not p["state"].exists():
                continue
            st = json.loads(p["state"].read_text(encoding="utf-8"))
            if st.get("status") == "complete":
                observations.append(D5Observation(
                    replicate=r, training_samples=n, seed=seed,
                    test_samples=test_samples,
                    test_accuracy=float(st["test_accuracy"]),
                    test_loss=float(st["test_loss"]),
                    completed_epochs=int(st["completed_epochs"]),
                    total_epochs=total_epochs,
                ))

    if len(observations) != replicates * len(sizes):
        raise RuntimeError("D5 finished without a complete observation for every replicate/size condition.")

    summary, pairwise, scaling = _aggregate(
        observations, sizes, replicates, bootstrap_replicates, audit_seed + 991
    )
    return D5Result(
        experiment=experiment_name,
        training_sizes=sizes,
        replicates=replicates,
        observations=observations,
        summary=summary,
        pairwise=pairwise,
        scaling=scaling,
        manifest={**base_manifest, "config_hash": config_hash},
    )


def generate_certificate(result: D5Result, *, dataset_id: str,
                         dataset_version: str, generation_procedure: str,
                         generation_parameters: Mapping[str, Any],
                         audit_seed: int) -> dict[str, Any]:
    return {
        "audit": {
            "id": "D5",
            "name": "Training-Data Scaling Audit",
            "scope": "sample-size scaling under a fixed Gohr dataset/model/training protocol",
            "claim": (
                "Characterizes how predictive performance changes with training-set size; "
                "it does not by itself establish cryptographic learning or a universal "
                "dataset-size law."
            ),
        },
        "design": result.manifest["design"],
        "configuration": result.manifest,
        "findings": {
            "observations": [asdict(x) for x in result.observations],
            "summary_by_training_size": result.summary,
            "paired_contrasts": result.pairwise,
            "scaling_analysis": result.scaling,
        },
        "provenance": {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "generation_procedure": generation_procedure,
            "generation_parameters": dict(generation_parameters),
            "audit_seed": audit_seed,
        },
        "interpretation": (
            "D5 is a descriptive/estimative scaling experiment. "
            "The nested-prefix component controls the sampled population within each "
            "replicate, while independent replicate maximum datasets quantify "
            "between-dataset and training stochastic variation."
        ),
        "limitations": [
            "The result is conditional on the specified model architecture, optimizer, epoch budget, batch size and generator.",
            "Nested prefixes are not independent datasets; their purpose is paired sample-size comparison within replicate.",
            "Independent replicate datasets are required to quantify dataset-instance variability.",
            "A scaling curve does not establish causality beyond the controlled experimental factors.",
            "Test-set uncertainty is not treated as independent across conditions because the same fixed test set is deliberately reused.",
        ],
    }


def print_report(result: D5Result) -> None:
    print()
    print("=" * 72)
    print("Dataset Integrity Audit")
    print("D5 — Training-Data Scaling Audit")
    print("=" * 72)
    print(f"Replicates                  : {result.replicates}")
    print(f"Training sizes              : {', '.join(f'{n:,}' for n in result.training_sizes)}")
    print("Design                      : independent replicate datasets + nested prefixes")
    print()
    print("Learning curve summary")
    print("-" * 72)
    for n in result.training_sizes:
        s = result.summary[str(n)]
        ci = s["accuracy_ci95_bootstrap"]
        print(
            f"{n:>12,d} | mean accuracy={s['accuracy_mean']:.8f} "
            f"| SD={s['accuracy_sd']:.8f} "
            f"| 95% bootstrap CI=[{ci[0]:.8f}, {ci[1]:.8f}]"
        )
    print()
    print("Paired contrasts vs smallest training size")
    print("-" * 72)
    for k, v in result.pairwise.items():
        if k == "multiple_comparison_correction":
            continue
        ci = v["bootstrap_ci95"]
        print(
            f"{k:>28} | mean Δ={v['mean_accuracy_difference']:+.8f} "
            f"| 95% CI=[{ci[0]:+.8f}, {ci[1]:+.8f}] "
            f"| paired p={v['paired_p_value']:.6g}"
        )
    print()
    print("Scaling summary")
    print("-" * 72)
    s = result.scaling
    ci = s["mean_slope_bootstrap_ci95"]
    print(f"Mean accuracy slope vs log10(N): {s['mean_slope']:+.8f}")
    print(f"95% bootstrap CI              : [{ci[0]:+.8f}, {ci[1]:+.8f}]")
    print(
        f"Monotone non-decreasing curves : "
        f"{s['monotone_non_decreasing_replicates']}/"
        f"{s['replicates_with_complete_curves']}"
    )
    print("=" * 72)


def write_learning_curve_plot(result: D5Result, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = np.array(result.training_sizes, dtype=float)
    means = np.array([result.summary[str(n)]["accuracy_mean"] for n in result.training_sizes])
    lo = np.array([result.summary[str(n)]["accuracy_ci95_bootstrap"][0] for n in result.training_sizes])
    hi = np.array([result.summary[str(n)]["accuracy_ci95_bootstrap"][1] for n in result.training_sizes])
    ax.plot(sizes, means, marker="o", label="Mean test accuracy")
    ax.fill_between(sizes, lo, hi, alpha=0.2, label="95% bootstrap CI")
    for r in range(1, result.replicates + 1):
        ys = []
        for n in result.training_sizes:
            obs = next(o for o in result.observations
                       if o.replicate == r and o.training_samples == n)
            ys.append(obs.test_accuracy)
        ax.plot(sizes, ys, marker=".", alpha=0.35, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Test accuracy")
    ax.set_title("D5 Training-Data Scaling")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
