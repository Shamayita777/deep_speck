"""
AutoND D5 target-level verification: independent checkpoint re-evaluation and certification.

The generic D5 engine (audit.dataset.d5_training_data_scaling, schema 2.1) records per-condition
test metrics in state.json but does not bind checkpoint BYTES to a stored digest.  This module is a
separate, target-level safeguard; it does not change or claim to fix the engine's resume logic.

What it adds, for one completed run directory:

* a target evaluation log, written by the driver at the moment the engine evaluates each condition,
  that records the SHA-256 of that condition's final checkpoint (``latest.keras``);
* an independent pass that re-verifies every persisted dataset through the engine's own loaders,
  re-hashes every checkpoint against the log, reloads it, re-predicts the persisted test set and
  compares accuracy and MSE with the engine's recorded values under the registered tolerance;
* a recomputation of the engine's aggregate statistics from the state files, compared with the
  engine certificate; and the registered target-supplementary statistics (paired-t intervals,
  standardized effect sizes, exact Wilcoxon sensitivity test, normality diagnostic);
* a certificate that keeps the engine-generated result and the target verification in separate
  sections, and is ``VERIFIED`` only when every check passed.

TensorFlow, Keras and the engine are imported lazily, so importing this module loads none of them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from audit.dataset.integrity import REPOSITORY_ROOT, IntegrityError, sha256_file, write_bytes_exclusive

CERTIFICATE_SCHEMA = "autond-d5-target-certificate-1"
VERIFICATION_SCHEMA = "autond-d5-target-verification-1"
LOG_SCHEMA = "autond-d5-evaluation-log-record-1"

ENGINE_DIR = "engine"
RESULT_DIR = "result"
RUN_CLAIM = "run_claim.json"
TARGET_LOG = "target_evaluation_log.jsonl"
ENGINE_CERTIFICATE = "engine_certificate.json"
TARGET_VERIFICATION = "target_verification.json"
TARGET_CERTIFICATE = "d5_autond_certificate.json"

PREDICT_BATCH_SIZE = 5000
THRESHOLD = 0.5          # Keras binary accuracy: prediction > 0.5 is class 1

CONTRAST_CLASSES = (
    "SIGNIFICANT_AND_PRACTICALLY_MEANINGFUL",
    "SIGNIFICANT_BELOW_PRACTICAL_THRESHOLD",
    "NOT_SIGNIFICANT_INCONCLUSIVE",
)


def _engine():
    from audit.dataset import d5_training_data_scaling as engine

    return engine


def write_result_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Exclusive (never overwrite) JSON write that, like the engine's own output, keeps NaN/Infinity tokens:
    the engine reports an infinite paired t when every difference is identical."""
    write_bytes_exclusive(Path(path), json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8") + b"\n")


def canonical(payload: Any) -> str:
    """Canonical JSON text (sorted keys); NaN is kept as the token NaN so equal NaNs compare equal."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# ============================================================
# Target evaluation log (append-only, fsynced)
# ============================================================


def append_log_record(path: Path, record: Mapping[str, Any]) -> None:
    line = json.dumps({"schema": LOG_SCHEMA, **record}, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_log(path: Path) -> dict[str, dict]:
    """Parse the log; refuse malformed lines and any condition recorded twice."""
    if not Path(path).is_file():
        raise IntegrityError(f"Target evaluation log {path} is missing.")
    records: dict[str, dict] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise IntegrityError(f"{path}:{number} is not JSON: {exc}") from None
        if record.get("schema") != LOG_SCHEMA or "condition" not in record:
            raise IntegrityError(f"{path}:{number} is not a {LOG_SCHEMA} record.")
        if record["condition"] in records:
            raise IntegrityError(f"{path}: condition {record['condition']} is recorded twice.")
        records[record["condition"]] = record
    return records


def condition_name(replicate: int, size: int) -> str:
    return f"replicate_{int(replicate):02d}/n_{int(size)}"


def expected_conditions(sizes: Iterable[int], replicates: int) -> list[tuple[int, int]]:
    """The engine's evaluation order: replicates 1..R, sizes ascending within each."""
    ordered = sorted({int(n) for n in sizes})
    return [(r, n) for r in range(1, int(replicates) + 1) for n in ordered]


# ============================================================
# Independent re-evaluation
# ============================================================


def reevaluate_checkpoint(path: Path, test_x, test_y, *, batch_size: int = PREDICT_BATCH_SIZE) -> dict:
    """Reload a saved model and recompute accuracy (threshold 0.5), MSE, and the compiled objective.

    Keras' evaluate() loss is the compiled objective: MSE plus the model's weight-regularization losses
    (DBitNet's layers carry L2 penalties).  The engine records that objective as test_loss, so it is
    recomputed as MSE + sum(model.losses) and both terms are reported.
    """
    import tensorflow as tf

    model = tf.keras.models.load_model(path, compile=False)
    regularization = float(tf.add_n(model.losses).numpy()) if model.losses else 0.0
    y = np.asarray(test_y).reshape(-1).astype(np.float64)
    correct = 0
    squared = 0.0
    for start in range(0, len(y), batch_size):
        batch = np.asarray(test_x[start:start + batch_size])
        p = np.asarray(model.predict(batch, batch_size=batch_size, verbose=0), dtype=np.float64).reshape(-1)
        target = y[start:start + batch_size]
        correct += int(np.sum((p > THRESHOLD).astype(np.float64) == target))
        squared += float(np.sum((p - target) ** 2))
    del model
    n = len(y)
    return {"n": int(n), "correct": int(correct), "accuracy": correct / n, "loss_mse": squared / n,
            "regularization_loss": regularization, "loss_objective": squared / n + regularization}


# ============================================================
# Target-supplementary statistics (registered; the engine's analysis is kept as-is)
# ============================================================


def require_statistics_dependencies() -> dict:
    """SciPy is required: without it the engine's paired p-values fall back to a normal approximation,
    which is anti-conservative for small replicate counts."""
    try:
        import scipy
        from scipy import stats
    except Exception as exc:  # noqa: BLE001 - any import failure is a refusal
        raise IntegrityError(f"SciPy is required for the registered D5 analysis and is not importable: {exc}")
    for name in ("t", "wilcoxon", "shapiro", "binomtest"):
        if not hasattr(stats, name):
            raise IntegrityError(f"scipy.stats.{name} is unavailable; the registered analysis cannot run.")
    # The exact import the engine's _paired_t_test performs; its except-branch is the fallback being excluded.
    try:
        from scipy.stats import t as t_dist

        reference = float(2.0 * t_dist.sf(2.5, df=11))
    except Exception as exc:  # noqa: BLE001
        raise IntegrityError(f"scipy.stats.t cannot be used as the engine uses it: {exc}")
    if not abs(reference - 0.0295064) < 1e-6:
        raise IntegrityError(f"scipy.stats.t gives p={reference} for t=2.5, df=11 (expected 0.0295064).")
    return {"scipy": scipy.__version__}


def paired_supplement(diffs: Iterable[float], *, confidence: float = 0.95) -> dict:
    from scipy import stats

    d = np.asarray(list(diffs), dtype=float)
    k = len(d)
    out: dict[str, Any] = {"replicates": k, "mean": float(d.mean()) if k else float("nan")}
    if k < 2:
        return {**out, "note": "fewer than two paired differences"}
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(k)
    half = float(stats.t.ppf(0.5 + confidence / 2, k - 1)) * se
    out.update({
        "sd": sd,
        "t_interval": {"confidence": confidence, "df": k - 1, "lower": out["mean"] - half, "upper": out["mean"] + half},
        "cohens_dz": (out["mean"] / sd) if sd > 0 else None,
        "relative_to_baseline_mean": None,
    })
    if np.any(d != 0):
        try:
            w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
            out["wilcoxon_signed_rank"] = {"statistic": float(w.statistic), "p_value": float(w.pvalue),
                                           "role": "registered sensitivity analysis; not the primary test"}
        except ValueError as exc:
            out["wilcoxon_signed_rank"] = {"error": str(exc)}
    if k >= 3 and sd > 0:
        s = stats.shapiro(d)
        out["shapiro_wilk"] = {"statistic": float(s.statistic), "p_value": float(s.pvalue),
                               "role": "diagnostic of the paired-t normality assumption; low power at this k"}
    return out


def mean_t_interval(values: Iterable[float], *, confidence: float = 0.95) -> dict:
    from scipy import stats

    v = np.asarray(list(values), dtype=float)
    k = len(v)
    if k < 2:
        return {"replicates": k}
    half = float(stats.t.ppf(0.5 + confidence / 2, k - 1)) * float(v.std(ddof=1)) / math.sqrt(k)
    return {"replicates": k, "mean": float(v.mean()), "lower": float(v.mean()) - half, "upper": float(v.mean()) + half,
            "confidence": confidence}


def classify_contrast(*, holm_p: float, mean_difference: float, t_interval: Mapping, alpha: float,
                      practical_threshold: float, practical_rule: str) -> str:
    """Registered per-contrast classification.  A non-significant contrast is inconclusive, never equivalence."""
    if not (np.isfinite(holm_p) and holm_p < alpha):
        return "NOT_SIGNIFICANT_INCONCLUSIVE"
    if practical_rule == "point_estimate_at_least_threshold":
        practical = abs(mean_difference) >= practical_threshold
    elif practical_rule == "t_interval_excludes_threshold_band":
        practical = t_interval["lower"] >= practical_threshold or t_interval["upper"] <= -practical_threshold
    else:
        raise IntegrityError(f"Unknown practical_effect_rule {practical_rule!r}.")
    return "SIGNIFICANT_AND_PRACTICALLY_MEANINGFUL" if practical else "SIGNIFICANT_BELOW_PRACTICAL_THRESHOLD"


def supplementary_statistics(observations: list[dict], *, sizes: list[int], replicates: int, alpha: float,
                             holm: Mapping[str, float], practical_threshold: float, practical_rule: str) -> dict:
    lookup = {(o["replicate"], o["training_samples"]): o for o in observations}
    base = sizes[0]
    per_size = {str(n): mean_t_interval(lookup[(r, n)]["test_accuracy"] for r in range(1, replicates + 1))
                for n in sizes}
    for n in sizes:
        accs = [lookup[(r, n)]["test_accuracy"] for r in range(1, replicates + 1)]
        n_test = lookup[(1, n)]["test_samples"]
        mean_acc = float(np.mean(accs))
        per_size[str(n)]["test_set_binomial_se_at_mean"] = math.sqrt(mean_acc * (1 - mean_acc) / n_test)
    contrasts = {}
    base_mean = float(np.mean([lookup[(r, base)]["test_accuracy"] for r in range(1, replicates + 1)]))
    for n in sizes[1:]:
        key = f"{base}_vs_{n}"
        diffs = [lookup[(r, n)]["test_accuracy"] - lookup[(r, base)]["test_accuracy"] for r in range(1, replicates + 1)]
        loss_diffs = [lookup[(r, n)]["test_loss"] - lookup[(r, base)]["test_loss"] for r in range(1, replicates + 1)]
        entry = paired_supplement(diffs)
        entry["relative_to_baseline_mean"] = entry["mean"] / base_mean if base_mean else None
        entry["holm_adjusted_p_value_engine"] = holm.get(key)
        entry["classification"] = classify_contrast(
            holm_p=holm.get(key, float("nan")), mean_difference=entry["mean"], t_interval=entry["t_interval"],
            alpha=alpha, practical_threshold=practical_threshold, practical_rule=practical_rule)
        entry["test_loss_difference"] = {**paired_supplement(loss_diffs), "role": "exploratory"}
        contrasts[key] = entry
    return {
        "role": ("Target-supplementary, registered before execution: t-based intervals, standardized effect sizes, "
                 "a sensitivity test and an assumption diagnostic. The engine's paired-t / Holm / bootstrap analysis is "
                 "reported unchanged in the engine section and is the primary test; these do not replace it."),
        "per_training_size_accuracy": per_size,
        "contrasts_vs_baseline": contrasts,
        "classification_rule": {"alpha": alpha, "practical_threshold": practical_threshold,
                                "practical_effect_rule": practical_rule, "classes": list(CONTRAST_CLASSES),
                                "non_significance": "inconclusive; never interpreted as equivalence"},
    }


def _finite(value) -> bool:
    """True only for a real, finite number (None, strings, NaN and +-inf are not)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _probability(value) -> bool:
    return _finite(value) and 0.0 <= value <= 1.0


def _holm(raw: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (the engine's formula), for cross-checking its output."""
    adjusted, running = {}, 0.0
    ordered = sorted(raw.items(), key=lambda kv: kv[1])
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        adjusted[key] = running
    return adjusted


def statistical_integrity_problems(findings: Mapping[str, Any], *, sizes: list[int], replicates: int) -> list[str]:
    """Problems that make the engine's statistics unusable for a positive result.

    The engine's Holm step drops non-finite p-values from the family (shrinking m, so the correction is
    weaker for the rest) and its paired t returns t = +-inf, p = 0 when every paired difference is identical.
    Neither may pass silently, nor may an incomplete pairing or a malformed field.
    """
    problems = []
    observations = findings.get("observations", [])
    cells = [(o.get("replicate"), o.get("training_samples")) for o in observations]
    want = [(r, n) for r in range(1, replicates + 1) for n in sizes]
    if sorted(cells, key=str) != sorted(want, key=str):
        problems.append(f"observations cover {len(cells)} (replicate, size) cells, not exactly the registered "
                        f"{len(want)}; pairing is incomplete or duplicated")
    for o in observations:
        if not (_finite(o.get("test_accuracy")) and _finite(o.get("test_loss"))):
            problems.append(f"non-finite observation for replicate {o.get('replicate')} n={o.get('training_samples')}")
    for n in sizes:
        row = findings.get("summary_by_training_size", {}).get(str(n), {})
        if row.get("replicates") != replicates:
            problems.append(f"summary for n={n} uses {row.get('replicates')!r} replicates, expected {replicates}")
        bounds = row.get("accuracy_ci95_bootstrap")
        if not (isinstance(bounds, list) and len(bounds) == 2 and all(_finite(b) for b in bounds)
                and _finite(row.get("accuracy_mean")) and _finite(row.get("accuracy_sd"))):
            problems.append(f"summary for n={n}: mean, SD or CI is missing or not finite")
    paired = findings.get("paired_contrasts", {})
    correction = paired.get("multiple_comparison_correction", {})
    adjusted = correction.get("adjusted_p_values", {})
    expected = [f"{sizes[0]}_vs_{n}" for n in sizes[1:]]
    if paired.get("baseline_training_size") != sizes[0]:
        problems.append(f"baseline is {paired.get('baseline_training_size')!r}, expected the smallest size {sizes[0]}")
    if correction.get("method") != "Holm step-down" or sorted(adjusted) != sorted(expected):
        problems.append(f"correction is {correction.get('method')!r} over {sorted(adjusted)}; expected Holm "
                        f"step-down over every registered contrast {expected}")
    raw = {}
    for key in expected:
        contrast = paired.get(key)
        if not isinstance(contrast, Mapping):
            problems.append(f"contrast {key} is missing")
            continue
        if contrast.get("replicates") != replicates:
            problems.append(f"contrast {key} pairs {contrast.get('replicates')!r} replicates, expected {replicates}")
        for field in ("mean_accuracy_difference", "paired_t_statistic", "sd_accuracy_difference"):
            if not _finite(contrast.get(field)):
                problems.append(f"contrast {key}: {field} is not finite ({contrast.get(field)!r})")
        if _finite(contrast.get("sd_accuracy_difference")) and not contrast["sd_accuracy_difference"] > 0:
            problems.append(f"contrast {key}: zero variance of paired differences; the paired t-test is degenerate")
        if not _probability(contrast.get("paired_p_value")):
            problems.append(f"contrast {key}: raw p {contrast.get('paired_p_value')!r} is not a probability")
        else:
            raw[key] = contrast["paired_p_value"]
        bounds = contrast.get("bootstrap_ci95")
        if not (isinstance(bounds, list) and len(bounds) == 2 and all(_finite(b) for b in bounds)):
            problems.append(f"contrast {key}: bootstrap CI {bounds!r} is missing or not finite")
        if key in adjusted and not _probability(adjusted[key]):
            problems.append(f"contrast {key}: Holm-adjusted p {adjusted[key]!r} is not a probability")
    if sorted(raw) == sorted(expected) == sorted(adjusted) and all(_probability(v) for v in adjusted.values()):
        for key, value in _holm(raw).items():
            if abs(value - adjusted[key]) > 1e-12:
                problems.append(f"contrast {key}: Holm-adjusted p {adjusted[key]!r} differs from Holm({value!r})")
    return problems


def supplementary_problems(supplement: Mapping[str, Any]) -> list[str]:
    """The target's own intervals and effect sizes must be finite before a classification is reported."""
    problems = []
    for key, entry in supplement.get("contrasts_vs_baseline", {}).items():
        interval = entry.get("t_interval", {})
        if not (_finite(entry.get("mean")) and _finite(interval.get("lower")) and _finite(interval.get("upper"))
                and _finite(entry.get("cohens_dz"))):
            problems.append(f"supplementary contrast {key}: mean, t interval or d_z is missing or not finite")
    for n, entry in supplement.get("per_training_size_accuracy", {}).items():
        if not (_finite(entry.get("lower")) and _finite(entry.get("upper"))):
            problems.append(f"supplementary size {n}: t interval is missing or not finite")
    return problems


# ============================================================
# Whole-run verification
# ============================================================


def repository_root() -> Path:
    """The repository that holds the approved preregistration and the sources (module global, read at call time)."""
    return Path(REPOSITORY_ROOT)


def _hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _git_blob(root: Path, commit: str, rel: str) -> bytes | None:
    out = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=str(root), capture_output=True, timeout=60)
    return out.stdout if out.returncode == 0 else None


def provenance_problems(claim: Mapping[str, Any]) -> list[str]:
    """Approval and source provenance of a run, re-derived from git at the commits the run claim records.

    The preregistration blob at the recorded commit must hash to the recorded anchor and be an approved protocol
    matching this driver, with the run's values and upstream pin; every registered source file must have been
    clean, its recorded digest must be its content at the recorded HEAD, and that HEAD must contain the
    preregistration commit.
    """
    from audit.targets.autond.d5 import training_data_scaling as D

    root = repository_root()
    problems: list[str] = []
    prereg = claim.get("preregistration")
    commit = prereg.get("commit") if isinstance(prereg, dict) else None
    sha = commit.get("sha") if isinstance(commit, dict) else None
    if not (isinstance(prereg, dict) and _hex(prereg.get("sha256"), 64) and _hex(sha, 40)
            and isinstance(prereg.get("path"), str)):
        return ["run claim has no anchored, committed preregistration record (no approval evidence)"]
    blob = _git_blob(root, sha, prereg["path"])
    if blob is None:
        problems.append(f"approved preregistration {prereg['path']} is not available at commit {sha[:12]}")
    elif hashlib.sha256(blob).hexdigest() != prereg["sha256"]:
        problems.append("preregistration at the recorded commit does not match the recorded anchor")
    else:
        record = json.loads(blob)
        review = D.review_preregistration(record)
        refused = review["problems"] + D.check_fixed_design(record) + D.check_decision_values(review["values"])
        if refused or review["pending"] or record.get("approved") is not True:
            problems.append(f"preregistration at the recorded commit is not an approved protocol: "
                            f"{refused + review['pending']}")
        if review["values"] != claim.get("values"):
            problems.append("run values differ from the approved preregistration's decisions")
        if record.get("protocol_id") != prereg.get("protocol_id"):
            problems.append("run protocol id differs from the approved preregistration's")
        pin = record.get("fixed_design", {}).get("upstream", {})
        upstream = claim.get("upstream") or {}
        bound = upstream.get("bound_source_digests") or {}
        if (upstream.get("commit") != pin.get("commit")
                or any(bound.get(k) != v for k, v in (pin.get("source_digests") or {}).items())):
            problems.append("bound upstream commit or source digests differ from the approved pin")

    sources = claim.get("source_identity")
    files = sources.get("files") if isinstance(sources, dict) else None
    if not isinstance(files, dict):
        return problems + ["run claim has no source identity"]
    if set(files) != set(D.SOURCE_PATHS):
        problems.append(f"source identity lists {sorted(files)}, not the registered sources")
    dirty = sorted(rel for rel, entry in files.items()
                   if not (isinstance(entry, dict) and entry.get("clean_at_head") is True))
    if dirty:
        problems.append(f"sources were not clean at HEAD when the run started: {dirty}")
    head = sources.get("git_head")
    if not _hex(head, 40):
        return problems + ["source identity has no git HEAD"]
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", sha, head], cwd=str(root),
                              capture_output=True, timeout=60)
    if ancestor.returncode != 0:
        problems.append(f"source HEAD {head[:12]} does not contain the preregistration commit {sha[:12]}")
    for rel, entry in sorted(files.items()):
        content = _git_blob(root, head, rel)
        if (content is None or not isinstance(entry, dict)
                or hashlib.sha256(content).hexdigest() != entry.get("sha256")):
            problems.append(f"source {rel}: recorded SHA-256 is not its committed content at {head[:12]}")
    return problems


def _read_json(path: Path, what: str) -> dict:
    if not Path(path).is_file():
        raise IntegrityError(f"{what} is missing: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_run(run_dir: Path, *, reevaluate: bool = True) -> dict:
    """Independently verify a completed run; never raises on content, every failure becomes a problem."""
    run_dir = Path(run_dir)
    engine = _engine()
    problems: list[str] = []
    report: dict[str, Any] = {"schema": VERIFICATION_SCHEMA, "run_dir": str(run_dir), "reevaluated": reevaluate}
    try:
        report["statistics_dependencies"] = require_statistics_dependencies()
    except IntegrityError as exc:
        return {**report, "status": "NOT_VERIFIED", "problems": [str(exc)]}

    try:
        claim = _read_json(run_dir / RUN_CLAIM, "run claim")
    except (IntegrityError, ValueError) as exc:
        return {**report, "status": "NOT_VERIFIED", "problems": [str(exc)]}
    values = claim["values"]
    sizes = sorted(int(n) for n in values["training_sizes"])
    replicates = int(values["replicates"])
    tolerance = values["verification_tolerance"]
    root = run_dir / ENGINE_DIR
    report["run_id"] = claim.get("run_id")

    # (0) approval and source provenance, re-derived from git
    try:
        problems.extend(provenance_problems(claim))
    except Exception as exc:  # noqa: BLE001
        problems.append(f"provenance: {exc}")

    # (1) engine manifest agrees with the registered run values
    try:
        manifest = _read_json(root / "run_manifest.json", "engine run manifest")
        for key, want in (("training_sizes", sizes), ("replicates", replicates),
                          ("test_samples", int(values["test_samples"])), ("total_epochs", int(values["epochs"])),
                          ("batch_size", int(values["batch_size"])), ("audit_seed", int(values["audit_seed"])),
                          ("bootstrap_replicates", int(values["bootstrap_replicates"]))):
            if manifest.get(key) != want:
                problems.append(f"engine manifest {key}={manifest.get(key)!r} differs from the registered {want!r}")
        if manifest.get("run_id") != claim.get("run_id"):
            problems.append("engine manifest run_id differs from the run claim (manifest not produced by this run)")
        if not manifest.get("config_hash"):
            problems.append("engine manifest has no config_hash")
        config_hash = manifest.get("config_hash")
    except (IntegrityError, ValueError) as exc:
        return {**report, "status": "NOT_VERIFIED", "problems": problems + [str(exc)]}

    # (2) every persisted dataset, re-verified by the engine's own loaders
    datasets: dict[str, Any] = {}
    test_x = test_y = None
    try:
        test_x, test_y, test_meta = engine.load_test_dataset(root, config_hash=config_hash)
        datasets["test"] = {k: test_meta[k] for k in ("samples", "test_x_file_sha256", "test_y_file_sha256",
                                                      "test_x_logical_sha256", "test_y_logical_sha256")}
    except Exception as exc:  # noqa: BLE001
        problems.append(f"fixed test dataset: {exc}")
    train_meta: dict[int, dict] = {}
    for r in range(1, replicates + 1):
        try:
            x, y, meta = engine.load_replicate_dataset(root, r, sizes=sizes, config_hash=config_hash)
            train_meta[r] = meta
            datasets[f"replicate_{r:02d}"] = {k: meta[k] for k in (
                "samples", "train_x_file_sha256", "train_y_file_sha256", "train_x_logical_sha256",
                "train_y_logical_sha256", "nested_prefix_hashes")}
            del x, y
        except Exception as exc:  # noqa: BLE001
            problems.append(f"replicate {r} dataset: {exc}")
    report["datasets"] = datasets

    # (2b) the pre-training data audits must exist and describe exactly the persisted arrays
    audits = {"test": ("test.json", "test", "test")}
    audits.update({f"replicate_{r:02d}": (f"replicate_{r:02d}.json", f"replicate_{r:02d}", "train")
                   for r in range(1, replicates + 1)})
    for label, (filename, key, prefix) in audits.items():
        path = run_dir / "data_audit" / filename
        if not path.is_file():
            problems.append(f"data audit {filename} is missing")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if key in datasets and (record.get("x_logical_sha256"), record.get("y_logical_sha256")) != (
                datasets[key][f"{prefix}_x_logical_sha256"], datasets[key][f"{prefix}_y_logical_sha256"]):
            problems.append(f"data audit {filename} does not describe the persisted {label} arrays")
        overlap = record.get("train_test_overlap", {}).get("rows", 0)
        if (prefix == "train" and overlap and values["partition_overlap_policy"]
                == "refuse_on_any_train_test_feature_row"):
            problems.append(f"data audit {filename}: {overlap} train/test overlapping rows under a refusing policy")

    # (3) every condition: state, checkpoint digest bound by the target log, independent re-evaluation
    try:
        log = read_log(run_dir / TARGET_LOG)
    except IntegrityError as exc:
        log = {}
        problems.append(str(exc))
    conditions: dict[str, Any] = {}
    observations: list[dict] = []
    for r, n in expected_conditions(sizes, replicates):
        name = condition_name(r, n)
        paths = engine.condition_paths(root, r, n)
        entry: dict[str, Any] = {}
        conditions[name] = entry
        if not paths["state"].is_file():
            problems.append(f"{name}: state.json is missing")
            continue
        if not paths["checkpoint"].is_file():
            problems.append(f"{name}: checkpoint latest.keras is missing")
            continue
        state = json.loads(paths["state"].read_text(encoding="utf-8"))
        entry["state"] = {k: state.get(k) for k in ("status", "completed_epochs", "total_epochs", "seed",
                                                    "test_accuracy", "test_loss")}
        if state.get("status") != "complete" or state.get("completed_epochs") != int(values["epochs"]):
            problems.append(f"{name}: state is not complete at the registered epoch budget")
        if state.get("config_hash") != config_hash:
            problems.append(f"{name}: state config_hash differs from the engine manifest")
        if r in train_meta:
            want_x, want_y = engine.prefix_hashes_for(train_meta[r], n)
            if (state.get("train_x_prefix_hash"), state.get("train_y_prefix_hash")) != (want_x, want_y):
                problems.append(f"{name}: state prefix hashes differ from the persisted nested-prefix hashes")
        if "test" in datasets and (state.get("test_x_hash"), state.get("test_y_hash")) != (
                datasets["test"]["test_x_logical_sha256"], datasets["test"]["test_y_logical_sha256"]):
            problems.append(f"{name}: state test hashes differ from the persisted test set")
        digest = sha256_file(paths["checkpoint"])
        entry["checkpoint_sha256"] = digest
        logged = log.get(name)
        if logged is None:
            problems.append(f"{name}: no target evaluation-log record (checkpoint digest unbound)")
        else:
            if logged.get("checkpoint_sha256") != digest:
                problems.append(f"{name}: checkpoint bytes differ from the digest recorded at evaluation time")
            if (logged.get("engine_test_accuracy"), logged.get("engine_test_loss")) != (
                    state.get("test_accuracy"), state.get("test_loss")):
                problems.append(f"{name}: state metrics differ from the metrics logged at evaluation time")
        if reevaluate and test_x is not None and not any(p.startswith(name) for p in problems):
            try:
                again = reevaluate_checkpoint(paths["checkpoint"], test_x, test_y)
                entry["independent_evaluation"] = again
                d_acc = abs(again["accuracy"] - float(state["test_accuracy"]))
                d_loss = abs(again["loss_objective"] - float(state["test_loss"]))
                entry["absolute_differences"] = {"accuracy": d_acc, "loss": d_loss}
                if d_acc > float(tolerance["accuracy_abs"]) or d_loss > float(tolerance["loss_abs"]):
                    problems.append(f"{name}: independent re-evaluation differs beyond tolerance "
                                    f"(|d_acc|={d_acc:.3g}, |d_loss|={d_loss:.3g})")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{name}: checkpoint could not be reloaded and re-evaluated: {exc}")
        if state.get("status") == "complete" and "test_accuracy" in state:
            observations.append({"replicate": r, "training_samples": n, "seed": state.get("seed"),
                                 "test_samples": int(values["test_samples"]),
                                 "test_accuracy": float(state["test_accuracy"]), "test_loss": float(state["test_loss"]),
                                 "completed_epochs": int(state["completed_epochs"]), "total_epochs": int(values["epochs"])})
    extra = sorted(set(log) - {condition_name(r, n) for r, n in expected_conditions(sizes, replicates)})
    if extra:
        problems.append(f"target log records unregistered conditions {extra}")
    report["conditions"] = conditions

    # (4) the engine certificate against the state files, and the engine's aggregates recomputed
    engine_cert = None
    try:
        engine_cert = _read_json(run_dir / RESULT_DIR / ENGINE_CERTIFICATE, "engine certificate")
        if canonical(engine_cert["findings"]["observations"]) != canonical(observations):
            problems.append("engine certificate observations differ from the per-condition state files")
        if len(observations) == len(expected_conditions(sizes, replicates)):
            obs = [engine.D5Observation(**o) for o in observations]
            summary, pairwise, scaling = engine._aggregate(obs, sizes, replicates, int(values["bootstrap_replicates"]),
                                                           int(values["audit_seed"]) + 991)
            for label, mine, theirs in (("summary", summary, engine_cert["findings"]["summary_by_training_size"]),
                                        ("paired contrasts", pairwise, engine_cert["findings"]["paired_contrasts"]),
                                        ("scaling", scaling, engine_cert["findings"]["scaling_analysis"])):
                if canonical(json.loads(json.dumps(mine))) != canonical(theirs):
                    problems.append(f"engine certificate {label} differ from their recomputation")
        problems.extend(statistical_integrity_problems(engine_cert["findings"], sizes=sizes, replicates=replicates))
    except (IntegrityError, KeyError, ValueError, TypeError) as exc:
        problems.append(f"engine certificate: {exc}")

    # (5) registered supplementary statistics
    if engine_cert is not None and len(observations) == len(expected_conditions(sizes, replicates)):
        try:
            holm = engine_cert["findings"]["paired_contrasts"]["multiple_comparison_correction"]["adjusted_p_values"]
            report["supplementary_statistics"] = supplementary_statistics(
                observations, sizes=sizes, replicates=replicates, alpha=float(values["alpha"]), holm=holm,
                practical_threshold=float(values["practical_effect_threshold"]),
                practical_rule=values["practical_effect_rule"])
            problems.extend(supplementary_problems(report["supplementary_statistics"]))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"supplementary statistics: {exc}")

    report["problems"] = problems
    if problems:
        report["status"] = "NOT_VERIFIED"
    else:
        report["status"] = "VERIFIED" if reevaluate else "INTEGRITY_CONSISTENT_NOT_REEVALUATED"
    return report


def build_certificate(run_dir: Path, report: Mapping[str, Any]) -> dict:
    """The final certificate: engine result and target verification kept in separate sections."""
    run_dir = Path(run_dir)
    claim = _read_json(run_dir / RUN_CLAIM, "run claim")
    engine_path = run_dir / RESULT_DIR / ENGINE_CERTIFICATE
    engine_section = None
    if engine_path.is_file():
        engine_section = {"path": f"{RESULT_DIR}/{ENGINE_CERTIFICATE}", "sha256": sha256_file(engine_path),
                          "certificate": json.loads(engine_path.read_text(encoding="utf-8")),
                          "generated_by": "audit.dataset.d5_training_data_scaling.generate_certificate (unmodified)"}
    return {
        "schema": CERTIFICATE_SCHEMA,
        "status": report["status"],
        "problems": list(report.get("problems", [])),
        "run_identity": {k: claim.get(k) for k in ("run_id", "created_utc", "preregistration", "upstream",
                                                     "source_identity", "environment", "engine_config_hash")},
        "engine_generated_result": engine_section,
        "target_independent_verification": dict(report),
        "claim_scope": claim.get("claim_scope"),
        "integrity_note": ("The engine section is the generic D5 engine's own output. The target section is an "
                           "independent re-verification by this AutoND driver: dataset re-hashing via the engine's "
                           "loaders, checkpoint digests bound at evaluation time, checkpoint reload and test-set "
                           "re-evaluation, and recomputation of the engine's aggregates. It does not alter, and does "
                           "not claim to secure, the engine's resume mechanism; this target never resumes a run."),
    }


def checkpoint_digests_from_certificate(certificate: Mapping[str, Any]) -> dict[str, str]:
    conditions = certificate["target_independent_verification"].get("conditions", {})
    return {name: entry["checkpoint_sha256"] for name, entry in conditions.items() if "checkpoint_sha256" in entry}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
