"""D3 method-validity calibration for the GOHR confirmatory design (evidence schema d3-method-validity-3).

GOHR DESIGN (the only differences from the AutoND tool d3_method_validity_calibration.py)
-----------------------------------------------------------------------------------------
R1 is CONFIRMATORY (the Gohr instance did not exist when the protocol was
fixed), so the per-guarantee budget 0.005 is split equally over SIX families
(R1, R2, A, B, C, D: 0.005 / 6 each) and every per-test alpha differs from
the AutoND calibration's. The engine verifier binds a calibration to the tool
FILE it is given, and this tool's worker processes re-import it, so the design
is declared in this file rather than patched onto the AutoND tool. Every
function except s5(), main() and s0_unit_suite() is identical to the AutoND
tool's (enforced by test_d3_gohr_calibration_tool.py); ROOT_SEED, FAMILY_ROLES
and UNAPPROVED_DESIGN_INPUTS are the changed constants. REPO_ROOT,
_repo_relative() and _environment_record() are added so the evidence records
repository-relative paths and the interpreter's name, not home-directory paths
(no verifier field reads them; the hashes are unchanged in meaning). Production sizes are the
canonical Gohr specification's (10^7 / 10^6 / 10^6), the same as AutoND's.

WHAT THIS RUN ESTABLISHES, AND WHAT IT DOES NOT
-----------------------------------------------
Every D3 decision rests on exact methods (engine guarantees G1, G2): exact
binomial tests, Fisher's exact conditional cell tests with Bonferroni, and
Clopper-Pearson bounds with Bonferroni. Their error control is a theorem, not
a simulation result. This run checks that the IMPLEMENTATION realises those
theorems at the design actually used. Its consistency checks (all must pass
for status IMPLEMENTATION_CONSISTENT):

    S0  the method-validity unit suite passes against the same engine;
    S1  exact size of the binomial test at the production n and alphas
        (computed exactly by summing pmfs);
    S6  exact non-coverage of the implemented Clopper-Pearson interval at
        every per-cell alpha the design uses, over an (n, p) grid that spans
        the supported class-stratum range (computed exactly);
    S8  exact conditional size of the implemented Fisher cell test at every
        per-cell alpha the design uses, over a margin grid spanning the
        production partitions and class strata (computed exactly);
    S7  simulated coverage of the TVD bounds (one-sided 99% lower confidence
        bound on coverage >= 1 - alpha);
    S2  the complete gated path, with the production roles and the joint
        alpha split, makes no more confirmatory false detections than the G2
        budget (global alpha / GUARANTEE_COUNT) under a synthetic null
        (one-sided exact binomial test of H0: rate <= budget; Bonferroni over
        the S2 checks). Because G2 is a theorem, S2 is an end-to-end
        consistency check, not the guarantee.

Reported only (never gating, never a criterion): the asymptotic chi-square
and Monte Carlo DIAGNOSTICS' simulated size (including the zero-column
regime at production alpha), each with its Monte Carlo uncertainty and the
inflation factor this run could detect; S3 power; S4 noise floor; S5
pre-data resolution. A non-rejected simulation is "no detectable inflation",
never a demonstration that a size is below a bound.

Nothing here reads Gohr (or any real) data. Production sizes come from the
canonical Gohr D0 specification (10^7 / 10^6 / 10^6), classes as Bernoulli(1/2).

USAGE
    python -B -m audit.dataset.tools.d3_gohr_method_validity_calibration \\
        --out-dir audit/dataset/evidence/d3_gohr_method_validity_<id> [--quick]
"""
from __future__ import annotations

import os

# One BLAS thread per process: the pool supplies the parallelism.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binom, chi2, hypergeom, ncx2, norm

from audit.dataset import d3_distribution_statistics as d3

SCHEMA_VERSION = d3.CALIBRATION_SCHEMA_VERSION
SUITE_PATH = Path(__file__).resolve().parents[1] / "tests" / "test_d3_method_validity.py"
ENGINE_PATH = Path(d3.__file__).resolve()
REPO_ROOT = SUITE_PATH.parents[3]
EVIDENCE_NAME = d3.CALIBRATION_EVIDENCE_NAME
MANIFEST_NAME = d3.CALIBRATION_MANIFEST_NAME
#: Distinct from the AutoND calibration's 20261022 so no random stream is shared.
ROOT_SEED = 20261023

#: Production design (D0 specification).
PRODUCTION_SIZES = {"train": 10_000_000, "validation": 1_000_000, "test": 1_000_000}
FEATURE_DIMENSION = 64
#: Global alpha, JOINT over G1 (bounds) and G2 (detections); each guarantee
#: gets GLOBAL_ALPHA / d3.GUARANTEE_COUNT (see d3.joint_alpha_problem).
GLOBAL_ALPHA = 0.01
GUARANTEE_ALPHA = GLOBAL_ALPHA / d3.GUARANTEE_COUNT
#: Gohr production roles: all six class-stratified families confirmatory (E, built
#: only with exploratory_alpha, is exploratory and outside this budget).
FAMILY_ROLES = {"R1": "confirmatory", "R2": "confirmatory", "A": "confirmatory", "B": "confirmatory",
                "C": "confirmatory", "D": "confirmatory"}
CONFIRMATORY = tuple(f for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES if FAMILY_ROLES[f] == "confirmatory")
FAMILY_ALPHA_ALLOCATION = d3.allocate_familywise_alpha(GUARANTEE_ALPHA, CONFIRMATORY)
#: Exploratory families are tested at the global alpha within each family, never gating.
EXPLORATORY_ALPHA = GLOBAL_ALPHA
assert d3.joint_alpha_problem(GLOBAL_ALPHA, FAMILY_ALPHA_ALLOCATION) is None
#: Class fractions the numerical verification spans (S6/S8 grids include both ends).
SUPPORTED_CLASS_FRACTION = (0.45, 0.55)
UNAPPROVED_DESIGN_INPUTS = ["joint global alpha 0.01 (0.005 per guarantee)", "equal six-family split of 0.005",
                            "R1 confirmatory", "family A confirmatory"]
COUNTS = d3.expected_hypothesis_counts(len(PRODUCTION_SIZES), FEATURE_DIMENSION)
PER_HYPOTHESIS_ALPHA = {f: FAMILY_ALPHA_ALLOCATION.get(f, EXPLORATORY_ALPHA) / COUNTS[f]
                        for f in d3.CLASS_STRATIFIED_CONFIRMATORY_FAMILIES}
ALPHAS_USED = d3._alphas_used(FAMILY_ALPHA_ALLOCATION, COUNTS)
CHECK_LEVEL = 0.05

VALIDATION_CRITERIA = {
    "S0": "the method-validity unit suite, run by this tool against the same engine, exits 0",
    "S1_binomial_exact": "exactly computed size of the binomial test <= alpha at every examined (n, alpha)",
    "S6": "exactly computed Clopper-Pearson non-coverage <= alpha for every design alpha and every S6 grid point",
    "S8": "exactly computed Fisher cell-test conditional size <= alpha for every design alpha and S8 margin",
    "S7": "one-sided 99% Clopper-Pearson lower bound on simulated TVD-bound coverage >= 1 - alpha, every cell",
    "S2": ("one-sided exact binomial test of H0: P(any confirmatory false detection on the gated path) <= "
           "the G2 budget (global alpha / GUARANTEE_COUNT); fails if p < CHECK_LEVEL / (number of S2 checks)"),
    "not_criteria": "diagnostics, S3 power, S4 noise floor and S5 resolution are reported only",
    "revision_history": [
        "schema 1-2 (superseded): status VALIDATED, gating on asymptotic chi-square, size checks with "
        "eps = 0.10. Superseded because a non-rejected simulation cannot establish a size guarantee; "
        "decisions now rest on exact methods only and eps no longer enters any check.",
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    """Path relative to the repository root, so evidence records no home-directory path."""
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def _environment_record() -> dict:
    """Interpreter and library identity, without the interpreter's absolute path."""
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "executable": Path(sys.executable).name,
            "virtual_environment": Path(sys.prefix).name}


def _rate(hits: int, reps: int) -> dict:
    lo, hi = d3.wilson_interval(int(hits), int(reps), 0.95)
    return {"hits": int(hits), "replicates": int(reps), "rate": hits / reps, "wilson95": [lo, hi]}


def detectable_inflation(reps: int, alpha: float, threshold: float, power: float = 0.8):
    """Smallest f such that a true size f * alpha is flagged (H0: size <= alpha) with >= power.

    None when no f < 100 is detectable with this budget."""
    crit = int(binom.isf(threshold, reps, alpha)) + 1
    while binom.sf(crit - 1, reps, alpha) >= threshold:
        crit += 1
    for f in np.arange(1.0, 100.0, 0.01):
        if binom.sf(crit - 1, reps, f * alpha) >= power:
            return float(round(f, 2))
    return None


# ---------------------------------------------------------------------------
# S0 — unit suite
# ---------------------------------------------------------------------------

def s0_unit_suite() -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", _repo_relative(SUITE_PATH)]
    t = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    tail = [ln for ln in out.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
    return {"command": " ".join(cmd[1:]), "exit_code": out.returncode, "summary": tail[0],
            "elapsed_seconds": round(time.time() - t, 1), "suite_sha256": _sha256(SUITE_PATH)}


# ---------------------------------------------------------------------------
# S1 — exact binomial size (check) and chi-square diagnostics (reported)
# ---------------------------------------------------------------------------

def exact_binomial_size(n: int, alpha: float) -> float:
    """Exact size of the two-sided minlike binomial test of p = 1/2 (sum of pmfs)."""
    sd = 0.5 * np.sqrt(n)
    lo, hi = max(0, int(n / 2 - 12 * sd)), min(n, int(n / 2 + 12 * sd))
    k = np.arange(lo, hi + 1)
    pv = np.minimum(1.0, 2.0 * binom.cdf(np.minimum(k, n - k), n, 0.5))
    tail_outside = 2.0 * binom.cdf(lo - 1, n, 0.5) if lo > 0 else 0.0
    return float(np.sum(binom.pmf(k, n, 0.5)[pv < alpha]) + tail_outside)


def _sim_chisq_task(args) -> dict:
    seed, na, nb, probs, reps, alphas = args
    rng = np.random.default_rng(seed)
    hits = np.zeros(len(alphas), dtype=np.int64)
    done = 0
    while done < reps:
        m = min(500_000, reps - done)
        p = d3._homogeneity_batch(rng.multinomial(na, probs, m), rng.multinomial(nb, probs, m))["asymptotic_p_value"]
        hits += np.array([(p < a).sum() for a in alphas])
        done += m
    return {"hits": hits.tolist(), "reps": done}


def s1(pool, reps: int, seeds, workers: int) -> dict:
    exact = []
    for label, n, fam in (("R1 train", 10_000_000, "R1"), ("R1 validation/test", 1_000_000, "R1"),
                          ("R2 train class (0.45)", 4_500_000, "R2"), ("R2 train class (0.55)", 5_500_000, "R2"),
                          ("R2 val/test class (0.45)", 450_000, "R2"), ("R2 val/test class (0.55)", 550_000, "R2")):
        for a in (1e-2, 1e-4, PER_HYPOTHESIS_ALPHA[fam]):
            size = exact_binomial_size(n, a)
            exact.append({"design": label, "n": n, "alpha": a, "exact_size": size, "valid": size <= a})
    diag_designs = [
        ("A pooled train vs validation 2x4", 10_000_000, 1_000_000, [0.25] * 4, "A"),
        ("B/C class validation vs test 2x4", 500_000, 500_000, [0.25] * 4, "B"),
        ("B/C zero-column (constant feature) 2x4, df=1", 500_000, 500_000, [0.0, 0.0, 0.5, 0.5], "B"),
        ("D cross-class 2x2 (validation/test)", 500_000, 500_000, [0.5, 0.5], "D"),
    ]
    diags = []
    for (label, na, nb, probs, fam), seed in zip(diag_designs, seeds):
        alphas = sorted({1e-3, 1e-4, 1e-5, 1e-6, PER_HYPOTHESIS_ALPHA[fam]}, reverse=True)
        ss = np.random.SeedSequence(seed).spawn(workers)
        res = list(pool.map(_sim_chisq_task, [(s, na, nb, probs, reps // workers, alphas) for s in ss]))
        tot = sum(r["reps"] for r in res)
        hits = np.sum([r["hits"] for r in res], axis=0)
        rows = []
        for a, h in zip(alphas, hits):
            r = _rate(h, tot)
            r.update(alpha=a, one_sided_p_size_le_alpha=float(binom.sf(h - 1, tot, a)),
                     detectable_inflation_factor_80pct_power=detectable_inflation(tot, a, CHECK_LEVEL))
            rows.append(r)
        diags.append({"design": label, "n_a": na, "n_b": nb, "cell_probabilities": probs, "results": rows})
    rng = np.random.default_rng(seeds[-1])
    na, nb, probs = 100_000, 20, [0.01, 0.09, 0.45, 0.45]
    ra = 2_000_000
    h = d3._homogeneity_batch(rng.multinomial(na, probs, ra), rng.multinomial(nb, probs, ra))
    sparse_asym = {f"{a:g}": _rate(int((h["asymptotic_p_value"] < a).sum()), ra) for a in (1e-2, 1e-3, 1e-4, 1e-5)}
    rm = 4000
    mc, ex = {}, {}
    for a in (0.05, 0.01):
        ta, tb = rng.multinomial(na, probs, rm), rng.multinomial(nb, probs, rm)
        t = d3.homogeneity_test_batch(ta, tb, per_hypothesis_alpha=a, mc_seed=int(rng.integers(2**31)))
        mc[f"{a:g}"] = dict(_rate(int(t["significant"].sum()), rm), alpha=a)
        e = d3.exact_homogeneity_test_batch(ta, tb, per_hypothesis_alpha=a)
        ex[f"{a:g}"] = dict(_rate(int(e["significant"].sum()), rm), alpha=a)
    return {
        "binomial_exact": exact,
        "diagnostics_chi_square_dense_production": diags,
        "diagnostics_sparse_unbalanced": {"design": {"n_a": na, "n_b": nb, "cell_probabilities": probs},
                                          "asymptotic_chi_square": sparse_asym, "monte_carlo": mc,
                                          "gating_exact_test": ex},
        "interpretation": ("binomial_exact is a check (exact). Every 'diagnostics_*' entry is a simulated size of a "
                           "NON-GATING test: 'no detectable inflation' at the stated detectable factor, never a "
                           "guarantee."),
    }


# ---------------------------------------------------------------------------
# S6 — exact Clopper-Pearson coverage; S8 — exact Fisher cell-test size
# ---------------------------------------------------------------------------

S6_N = (50, 1000, 450_000, 500_000, 550_000, 1_000_000, 4_500_000, 5_500_000, 10_000_000)
S6_P = (0.5, 0.25, 0.1, 1e-3, 1e-5)


def _exact_cp_noncoverage(n: int, p: float, alpha: float) -> float:
    sd = np.sqrt(n * p * (1 - p))
    lo_k, hi_k = max(0, int(n * p - 40 * sd - 40)), min(n, int(n * p + 40 * sd + 40))
    k = np.arange(lo_k, hi_k + 1, dtype=np.int64)
    lo, hi = d3.clopper_pearson_bounds(k, np.full_like(k, n), alpha=alpha)
    miss = (p < lo) | (p > hi)
    outside = binom.cdf(lo_k - 1, n, p) + binom.sf(hi_k, n, p)
    return float(np.sum(binom.pmf(k[miss], n, p)) + outside)


def s6() -> dict:
    rows = []
    for a in ALPHAS_USED["clopper_pearson"]:
        for n in S6_N:
            for p in S6_P:
                nc = _exact_cp_noncoverage(n, p, a)
                rows.append({"alpha": a, "n": n, "p": p, "exact_noncoverage": nc,
                             "ratio_to_alpha": nc / a, "valid": nc <= a * (1 + 1e-9)})
    return {"method": "exact sum of binomial pmf over counts whose interval excludes p",
            "alphas": ALPHAS_USED["clopper_pearson"], "rows": rows,
            "max_ratio_to_alpha": max(r["ratio_to_alpha"] for r in rows)}


#: (n_a, n_b) margins: pooled comparisons, class-stratum comparisons at the
#: supported ends, cross-class within a partition, and small-sample edges.
S8_DESIGNS = ((10_000_000, 1_000_000), (1_000_000, 1_000_000), (5_000_000, 500_000), (500_000, 500_000),
              (4_500_000, 550_000), (5_500_000, 450_000), (450_000, 550_000), (550_000, 450_000),
              (4_500_000, 5_500_000), (50, 40), (1000, 10))
S8_COLUMN = (0.5, 0.25, 0.1, 1e-3)
S8_SMALL_T = (1, 5, 50)


def _exact_fisher_conditional_size(n_a: int, n_b: int, t: int, alpha: float) -> float:
    total = n_a + n_b
    lo_x, hi_x = max(0, t - n_b), min(t, n_a)
    mean = t * n_a / total
    sd = np.sqrt(max(t * (n_a / total) * (n_b / total) * (total - t) / max(total - 1, 1), 1e-12))
    xs = np.arange(max(lo_x, int(mean - 40 * sd - 40)), min(hi_x, int(mean + 40 * sd + 40)) + 1)
    pc = d3.exact_cell_pvalues(np.stack([xs, n_a - xs], 1), np.stack([t - xs, n_b - t + xs], 1))[:, 0]
    pmf = hypergeom.pmf(xs, total, t, n_a)
    outside = hypergeom.cdf(xs[0] - 1, total, t, n_a) + hypergeom.sf(xs[-1], total, t, n_a)
    return float(pmf[pc < alpha].sum() + outside)


def _s8_task(args) -> list:
    a, n_a, n_b = args
    total = n_a + n_b
    ts = sorted({max(1, int(f * total)) for f in S8_COLUMN} | {t for t in S8_SMALL_T if t < total})
    out = []
    for t in ts:
        s = _exact_fisher_conditional_size(n_a, n_b, t, a)
        out.append({"alpha": a, "n_a": n_a, "n_b": n_b, "column_total": t, "exact_conditional_size": s,
                    "ratio_to_alpha": s / a, "valid": s <= a * (1 + 1e-9)})
    return out


# ---------------------------------------------------------------------------
# S7 — simulated bound coverage
# ---------------------------------------------------------------------------

S7_TVD = {
    "identical_uniform": (4000, 4000, [0.25] * 4, [0.25] * 4),
    "shifted": (5000, 3000, [0.25] * 4, [0.27, 0.24, 0.24, 0.25]),
    "rare_cells": (3000, 3000, [0.001, 0.009, 0.49, 0.5], [0.002, 0.008, 0.49, 0.5]),
    "unequal_sizes": (20000, 300, [0.25] * 4, [0.3, 0.2, 0.25, 0.25]),
    "zero_probability_cell": (2000, 2000, [0.0, 0.3, 0.3, 0.4], [0.0, 0.35, 0.25, 0.4]),
}
S7_ALPHAS = (0.05, 0.01, 0.001)
S7_REPS = 100_000


def _s7_task(args) -> dict:
    seed, regime, alpha, reps = args
    na, nb, pa, pb = S7_TVD[regime]
    rng = np.random.default_rng(seed)
    true = 0.5 * float(np.abs(np.subtract(pa, pb)).sum())
    cov = done = 0
    while done < reps:
        m = min(25_000, reps - done)
        bd = d3.tvd_confidence_bounds(rng.multinomial(na, pa, m), rng.multinomial(nb, pb, m), alpha=alpha)
        cov += int(np.sum((bd["lower"] <= true + 1e-15) & (true <= bd["upper"] + 1e-15)))
        done += m
    lo, _ = d3.clopper_pearson_bounds(np.int64(cov), np.int64(done), alpha=0.02)
    return {"regime": regime, "alpha": alpha, "covered": cov, "missed": done - cov, "replicates": done,
            "nominal_coverage": 1 - alpha, "coverage_lower_bound_99": float(lo),
            "valid": float(lo) >= 1 - alpha, "seed": int(seed)}


# ---------------------------------------------------------------------------
# S2/S3 — complete gated path on a synthetic law
# ---------------------------------------------------------------------------

def synthetic_partition(rng, n, *, positive_ratio=0.5, flip=0.3):
    """Rows i.i.d. Y ~ Bernoulli(positive_ratio). Y=0: 64 uniform bits.
    Y=1: C0 uniform on 32 bits, C1 = C0 XOR D with D_i ~ Bernoulli(flip).
    Every bit is exactly uniform in both classes; within-half pairs are
    independent; only the 32 aligned cross-half pairs carry class structure.
    A stand-in for the class signal, NOT a model of Speck."""
    y = (rng.random(n) < positive_ratio).astype(np.uint8)
    x = rng.integers(0, 2, (n, 64), dtype=np.uint8)
    pos = y == 1
    x[pos, 32:] = x[pos, :32] ^ (rng.random((int(pos.sum()), 32)) < flip).astype(np.uint8)
    return x, y


SYNTH_SIZES = (20_000, 5_000, 5_000)
TOLERANCES_FOR_PATH = {"class_ratio": 0.05, "class_conditional_bit": 0.05, "pairwise_tvd": 0.05, "cross_class_bit": 0.05}
GROUPS = {"C0": list(range(32)), "C1": list(range(32, 64))}


def _run_path(parts, alpha, seed):
    """The production roles with global alpha `alpha` split jointly over G1 and G2."""
    alloc = d3.allocate_familywise_alpha(alpha / d3.GUARANTEE_COUNT, CONFIRMATORY)
    audit = d3.class_stratified_audit(parts, expected_class_ratio=0.5, expected_bit_probability={0: 0.5, 1: 0.5},
                                      tolerances=TOLERANCES_FOR_PATH, family_alpha_allocation=alloc,
                                      mc_seed=seed, feature_groups=GROUPS, exploratory_alpha=alpha,
                                      family_roles=FAMILY_ROLES)
    # No preregistration or calibration exists for a synthetic path: the gated
    # outcome is INCONCLUSIVE by construction; the provisional outcome is tallied.
    dec = d3.evaluate_confirmatory_decision(audit)      # re-verifies the sealed audit internally
    return audit.payload(), dec


def _significant(fam) -> np.ndarray:
    if "comparisons" in fam["records"]:
        return np.concatenate([np.asarray(c["pairs"]["significant"]) for c in fam["records"]["comparisons"].values()])
    return np.asarray(fam["records"]["significant"])


def _s2_task(args) -> dict:
    seed, alpha, reps = args
    rng = np.random.default_rng(seed)
    any_rej = 0
    per_family = {f: 0 for f in CONFIRMATORY}
    outcomes: dict = {}
    for _ in range(reps):
        parts = {n: synthetic_partition(rng, s) for n, s in zip(("train", "validation", "test"), SYNTH_SIZES)}
        p, dec = _run_path(parts, alpha, int(rng.integers(2**31)))
        flags = {f["family_id"]: bool(_significant(f).any()) for f in p["families"] if f["role"] == "confirmatory"}
        any_rej += any(flags.values())
        for k, v in flags.items():
            per_family[k] += v
        outcomes[dec["provisional_outcome"]] = outcomes.get(dec["provisional_outcome"], 0) + 1
    return {"any": any_rej, "per_family": per_family, "reps": reps, "outcomes": outcomes}


INJECTIONS = {
    "class_ratio_shift_test": ("R1", (0.02, 0.04, 0.06)),
    "class1_bit_shift_test": ("R2|D", (0.02, 0.04, 0.06)),
    "class0_within_half_pair_dependence_test": ("C|A", (0.1, 0.2, 0.3)),
}


def _s3_task(args) -> dict:
    seed, kind, effect, reps, alpha = args
    rng = np.random.default_rng(seed)
    pos_rng = np.random.default_rng([seed, 99])
    iu, ju = np.triu_indices(64, 1)
    detect: dict = {}
    outcomes: dict = {}
    for _ in range(reps):
        parts = {n: synthetic_partition(rng, s) for n, s in zip(("train", "validation"), SYNTH_SIZES[:2])}
        if kind == "class_ratio_shift_test":
            parts["test"] = synthetic_partition(rng, SYNTH_SIZES[2], positive_ratio=0.5 + effect)
            target = {"R1": ("records", 2)}
        elif kind == "class1_bit_shift_test":
            x, y = synthetic_partition(rng, SYNTH_SIZES[2])
            j = int(pos_rng.integers(64))
            p = y == 1
            col = x[p, j]
            col[rng.random(int(p.sum())) < 2 * effect] = 1
            x[p, j] = col
            parts["test"] = (x, y)
            target = {"R2": ("records", 2 * 128 + j), "D": ("records", 2 * 64 + j)}
        else:
            x, y = synthetic_partition(rng, SYNTH_SIZES[2])
            half = int(pos_rng.integers(2)) * 32
            i, j = sorted(pos_rng.choice(32, 2, replace=False) + half)
            q = y == 0
            sub = x[q]
            cp = rng.random(len(sub)) < effect
            sub[cp, j] = sub[cp, i]
            x[q] = sub
            parts["test"] = (x, y)
            k = int(np.flatnonzero((iu == i) & (ju == j))[0])
            target = {"C": ("train_test", k), "A": ("train_test", k)}
        pl, dec = _run_path(parts, alpha, int(rng.integers(2**31)))
        fams = {f["family_id"]: f for f in pl["families"]}
        for fid, (where, idx) in target.items():
            f = fams[fid]
            sig = (f["records"]["comparisons"][where]["pairs"]["significant"] if where != "records"
                   else f["records"]["significant"])
            detect[fid] = detect.get(fid, 0) + bool(sig[idx])
        outcomes[dec["provisional_outcome"]] = outcomes.get(dec["provisional_outcome"], 0) + 1
    return {"detect": detect, "outcomes": outcomes, "reps": reps}


# ---------------------------------------------------------------------------
# S4 — TVD noise floor; S5 — pre-data resolution
# ---------------------------------------------------------------------------

def _null_class_counts(rng, n, chunk=500_000):
    n11 = {lab: np.zeros((64, 64), dtype=np.int64) for lab in (0, 1)}
    ones = {lab: np.zeros(64, dtype=np.int64) for lab in (0, 1)}
    cnt = {0: 0, 1: 0}
    done = 0
    while done < n:
        m = min(chunk, n - done)
        x = np.unpackbits(rng.integers(0, 256, (m, 8), dtype=np.uint8), axis=1)
        y = rng.integers(0, 2, m, dtype=np.uint8)
        for lab in (0, 1):
            xs = x[y == lab].astype(np.float32)
            n11[lab] += np.rint(xs.T @ xs).astype(np.int64)
            ones[lab] += xs.sum(axis=0).astype(np.int64)
            cnt[lab] += len(xs)
        done += m
    return n11, ones, cnt


def _tables(n11, ones, n):
    iu, ju = np.triu_indices(64, 1)
    a11 = n11[iu, ju]
    return np.stack([n - a11 - (ones[iu] - a11) - (ones[ju] - a11), ones[ju] - a11, ones[iu] - a11, a11], 1)


def _s4_task(seed) -> dict:
    rng = np.random.default_rng(seed)
    counts = {p: _null_class_counts(rng, n) for p, n in PRODUCTION_SIZES.items()}
    out = {}
    for stratum in ("pooled", "Y1", "Y0"):
        tabs, ns = {}, {}
        for p, (n11, ones, cnt) in counts.items():
            if stratum == "pooled":
                tabs[p], ns[p] = _tables(n11[0] + n11[1], ones[0] + ones[1], cnt[0] + cnt[1]), cnt[0] + cnt[1]
            else:
                lab = 1 if stratum == "Y1" else 0
                tabs[p], ns[p] = _tables(n11[lab], ones[lab], cnt[lab]), cnt[lab]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            tvd = 0.5 * np.abs(tabs[a] / ns[a] - tabs[b] / ns[b]).sum(1)
            ub = d3.tvd_confidence_bounds(tabs[a], tabs[b], alpha=PER_HYPOTHESIS_ALPHA["A"])["upper"]
            sig = d3.exact_homogeneity_test_batch(tabs[a], tabs[b], per_hypothesis_alpha=PER_HYPOTHESIS_ALPHA["A"])["significant"]
            out[f"{stratum}:{a}_{b}"] = {"max_tvd": float(tvd.max()), "q95_tvd": float(np.quantile(tvd, .95)),
                                         "median_tvd": float(np.median(tvd)), "max_tvd_upper_bound": float(ub.max()),
                                         "median_tvd_upper_bound": float(np.median(ub)),
                                         "exact_detections_under_null": int(sig.sum())}
    return out


def s5() -> dict:
    half = {p: n // 2 for p, n in PRODUCTION_SIZES.items()}
    effects = (0.005, 0.01, 0.02)

    def z_power(delta, se, a):
        z = norm.isf(a / 2)
        return float(norm.cdf(delta / se - z) + norm.cdf(-delta / se - z))

    rows = []
    for p, n in PRODUCTION_SIZES.items():
        a = PER_HYPOTHESIS_ALPHA["R1"]
        rows.append({"family": "R1", "role": "confirmatory", "design": p, "n": n, "per_hypothesis_alpha": a,
                     "minimum_certifiable_tolerance": d3.design_resolution(n, per_hypothesis_alpha=a, cell_probabilities=[0.5])["minimum_certifiable_tolerance"],
                     "approx_detection_power": {f"{e:g}": z_power(e, 0.5 / np.sqrt(n), a) for e in effects}})
        a = PER_HYPOTHESIS_ALPHA["R2"]
        rows.append({"family": "R2", "design": f"{p} per class", "n": half[p], "per_hypothesis_alpha": a,
                     "minimum_certifiable_tolerance": d3.design_resolution(half[p], per_hypothesis_alpha=a, cell_probabilities=[0.5])["minimum_certifiable_tolerance"],
                     "approx_detection_power": {f"{e:g}": z_power(e, 0.5 / np.sqrt(half[p]), a) for e in effects}})
        a = PER_HYPOTHESIS_ALPHA["D"]
        bd = d3.proportion_difference_bounds(np.int64(half[p] // 2), np.int64(half[p]), np.int64(half[p] // 2), np.int64(half[p]), alpha=a)
        se = np.sqrt(0.25 * 2 / half[p])
        rows.append({"family": "D", "design": f"{p} Y1 vs Y0", "n": [half[p], half[p]], "per_hypothesis_alpha": a,
                     "minimum_certifiable_tolerance": float(bd["abs_upper"]),
                     "approx_detection_power": {f"{e:g}": z_power(e, se, a) for e in effects}})
    for fam, sizes in (("A", PRODUCTION_SIZES), ("B/C", half)):
        a = PER_HYPOTHESIS_ALPHA["A"]
        for x, y in (("train", "validation"), ("train", "test"), ("validation", "test")):
            se = np.sqrt(0.1875 * (1 / sizes[x] + 1 / sizes[y]))
            n_eff = sizes[x] * sizes[y] / (sizes[x] + sizes[y])
            rows.append({"family": fam, "design": f"{x} vs {y}", "n": [sizes[x], sizes[y]], "per_hypothesis_alpha": a,
                         "minimum_certifiable_tolerance": d3.design_resolution(sizes[x], sizes[y], per_hypothesis_alpha=a)["minimum_certifiable_tolerance"],
                         "approx_detection_power_exact_cell_test_spread_pattern": {
                             f"{e:g}": float(1 - (1 - z_power(e / 2, se, a / 4)) ** 4) for e in effects},
                         "reference_power_asymptotic_chi_square_spread_pattern": {
                             f"{e:g}": float(ncx2.sf(chi2.isf(a, 3), 3, 4 * n_eff * e * e)) for e in effects}})
    return {"basis": ("pre-data: canonical Gohr D0 sizes with classes exactly n/2; confirmatory alphas from the equal "
                      "six-family split of 0.005 (joint global alpha 0.01), R1 confirmatory. "
                      "Power figures are normal approximations, labelled as such; the chi-square column is a "
                      "reference for the power cost of exact detection."), "rows": rows}


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--quick", action="store_true", help="tiny budgets; status is never IMPLEMENTATION_CONSISTENT")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    if a.out_dir.exists():
        raise SystemExit(f"REFUSED: {a.out_dir} exists. Evidence is never overwritten; choose a new directory.")
    budget = ({"s1_reps": 600_000, "s2_reps": 48, "s3_reps": 8, "s4_reps": 2, "s7_reps": 2_000} if a.quick else
              {"s1_reps": 200_000_000, "s2_reps": 1_000, "s3_reps": 100, "s4_reps": 12, "s7_reps": S7_REPS})
    run_id = uuid.uuid4().hex
    started = datetime.now(timezone.utc).isoformat()
    engine_sha = _sha256(ENGINE_PATH)
    tool_sha = _sha256(Path(__file__))
    suite_sha = _sha256(SUITE_PATH)
    seeds = np.random.SeedSequence(ROOT_SEED).generate_state(10).tolist()
    t0 = time.time()
    print("S0 unit suite", flush=True)
    r0 = s0_unit_suite() if not a.quick else {"exit_code": None, "summary": "not run (--quick)", "suite_sha256": suite_sha}
    print("S6 exact Clopper-Pearson coverage", flush=True)
    r6 = s6()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        print("S8 exact Fisher cell-test size", flush=True)
        r8_rows = [row for rows in pool.map(_s8_task, [(al, na, nb) for al in ALPHAS_USED["fisher_cell"]
                                                         for na, nb in S8_DESIGNS]) for row in rows]
        r8 = {"method": "exact sum of hypergeometric pmf over the conditional rejection region",
              "alphas": ALPHAS_USED["fisher_cell"], "rows": r8_rows,
              "max_ratio_to_alpha": max(r["ratio_to_alpha"] for r in r8_rows)}
        print("S7 bound coverage", flush=True)
        ss7 = np.random.SeedSequence([ROOT_SEED, 7]).spawn(len(S7_TVD) * len(S7_ALPHAS))
        cells = [(rg, al) for rg in S7_TVD for al in S7_ALPHAS]
        r7 = list(pool.map(_s7_task, [(int(s.generate_state(1)[0]), rg, al, budget["s7_reps"]) for s, (rg, al) in zip(ss7, cells)]))
        print("S5 design resolution", flush=True)
        r5 = s5()
        print("S1 exact binomial size and chi-square diagnostics", flush=True)
        r1 = s1(pool, budget["s1_reps"], seeds[:5], a.workers)
        print("S2 complete gated path, synthetic null", flush=True)
        r2 = {}
        for alpha, seed in ((0.05, seeds[5]), (0.01, seeds[6])):
            ss = np.random.SeedSequence(seed).spawn(a.workers * 4)
            per = -(-budget["s2_reps"] // len(ss))
            res = list(pool.map(_s2_task, [(s, alpha, per) for s in ss]))
            reps = sum(r["reps"] for r in res)
            any_ = sum(r["any"] for r in res)
            outc: dict = {}
            for r in res:
                for k, v in r["outcomes"].items():
                    outc[k] = outc.get(k, 0) + v
            # "alpha" is the G2 budget the size check tests against; global_alpha is joint over G1 and G2.
            r2[f"{alpha:g}"] = {"any_false_detection": dict(_rate(any_, reps), alpha=alpha / d3.GUARANTEE_COUNT,
                                                            global_alpha=alpha),
                                "per_family_false_detection": {f: _rate(sum(r["per_family"][f] for r in res), reps)
                                                               for f in CONFIRMATORY},
                                "provisional_outcomes": outc}
        print("S3 power", flush=True)
        ss3 = iter(np.random.SeedSequence(seeds[7]).spawn(100))
        tasks = [(int(next(ss3).generate_state(1)[0]), kind, e, -(-budget["s3_reps"] // 4), 0.01)
                 for kind, (_, effs) in INJECTIONS.items() for e in effs for _ in range(4)]
        res3 = list(pool.map(_s3_task, tasks))
        r3: dict = {}
        for (_, kind, e, _, _), r in zip(tasks, res3):
            agg = r3.setdefault(f"{kind}@{e:g}", {"detect": {}, "provisional_outcomes": {}, "reps": 0})
            agg["reps"] += r["reps"]
            for k, v in r["detect"].items():
                agg["detect"][k] = agg["detect"].get(k, 0) + v
            for k, v in r["outcomes"].items():
                agg["provisional_outcomes"][k] = agg["provisional_outcomes"].get(k, 0) + v
        for agg in r3.values():
            agg["power"] = {k: _rate(v, agg["reps"]) for k, v in agg["detect"].items()}
        print("S4 TVD noise floor at production sizes", flush=True)
        r4_reps = list(pool.map(_s4_task, np.random.SeedSequence(seeds[8]).spawn(budget["s4_reps"])))
    r4 = {key: {m: {"median": float(np.median([r[key][m] for r in r4_reps])),
                    "min": float(np.min([r[key][m] for r in r4_reps])),
                    "max": float(np.max([r[key][m] for r in r4_reps]))} for m in r4_reps[0][key]}
          for key in r4_reps[0]}

    for path, before in ((ENGINE_PATH, engine_sha), (Path(__file__), tool_sha), (SUITE_PATH, suite_sha)):
        if _sha256(path) != before:
            raise SystemExit(f"REFUSED: {path} changed during the run. Nothing was written.")
    s2_threshold = CHECK_LEVEL / len(r2)
    for v in r2.values():
        r = v["any_false_detection"]
        r["size_check"] = {"h0": f"rate <= {r['alpha']}", "p_value": float(binom.sf(r["hits"] - 1, r["replicates"], r["alpha"])),
                           "threshold": s2_threshold}
        r["valid"] = r["size_check"]["p_value"] >= s2_threshold
        r["detectable_inflation_factor_80pct_power"] = detectable_inflation(r["replicates"], r["alpha"], s2_threshold)
    checks = {
        "S0_unit_suite": r0.get("exit_code") == 0,
        "S1_binomial_exact": all(r["valid"] for r in r1["binomial_exact"]),
        "S2_complete_path": all(v["any_false_detection"]["valid"] for v in r2.values()),
        "S6_exact_clopper_pearson": all(r["valid"] for r in r6["rows"]),
        "S7_bound_coverage": all(r["valid"] for r in r7),
        "S8_exact_fisher": all(r["valid"] for r in r8["rows"]),
    }
    assert set(checks) == d3.REQUIRED_CALIBRATION_CHECKS
    status = (d3.CALIBRATION_STATUS if all(checks.values()) and not a.quick else
              "QUICK_RUN_NOT_CONSISTENCY_EVIDENCE" if a.quick else "NOT_CONSISTENT")
    evidence = {
        "schema_version": SCHEMA_VERSION, "status": status, "run_id": run_id, "checks": checks,
        "identity": {"engine_sha256": engine_sha, "tool_sha256": tool_sha, "test_suite_sha256": suite_sha,
                     "method_version": d3.D3_METHOD_VERSION, "decision_rule_version": d3.DECISION_RULE_VERSION,
                     "bound_construction": d3.BOUND_CONSTRUCTION, "detection_test": d3.DETECTION_TEST,
                     "engine_path": _repo_relative(ENGINE_PATH)},
        "design": {"global_alpha": GLOBAL_ALPHA, "guarantee_alpha": GUARANTEE_ALPHA,
                   "guarantee_count": d3.GUARANTEE_COUNT, "family_roles": FAMILY_ROLES,
                   "exploratory_alpha": EXPLORATORY_ALPHA, "family_alpha_allocation": FAMILY_ALPHA_ALLOCATION,
                   "partition_sizes": dict(PRODUCTION_SIZES), "feature_dimension": FEATURE_DIMENSION,
                   "supported_class_fraction": list(SUPPORTED_CLASS_FRACTION),
                   "hypothesis_counts": COUNTS, "per_test_alphas": ALPHAS_USED,
                   "unapproved_design_inputs": UNAPPROVED_DESIGN_INPUTS,
                   "tolerance_dependence": "no check depends on tolerances; no production tolerance is calibrated or implied"},
        "guarantee": {
            "decision_path": "EXACT_FINITE_SAMPLE",
            "G1_bounds": "Clopper-Pearson + Bonferroni + interval arithmetic; theorem, every n and law, rows i.i.d.",
            "G2_detection": "exact binomial and Fisher conditional cell tests + Bonferroni; theorem, independent samples",
            "joint": (f"G1 and G2 each at {GUARANTEE_ALPHA:g}; P(any G1 or G2 error) <= {GLOBAL_ALPHA:g} by the "
                      "union bound. The exploratory family E is outside this budget and carries no guarantee."),
            "what_this_run_adds": ("numerical verification that the implementation attains the exact guarantees at "
                                   "every per-test alpha the recorded design uses (S1, S6, S8), simulation consistency "
                                   "(S2, S7), and the unit suite (S0)"),
            "diagnostic_tests": ("asymptotic chi-square / Monte Carlo: size NOT guaranteed (no detectable inflation "
                                 "only); recorded, never gating"),
            "not_established": ["validity under dependent rows (D1/D2 probe specified duplicate and dependence "
                                "patterns only; neither establishes i.i.d. rows)",
                                "any design other than the recorded one", "scientific adequacy of any tolerance"],
        },
        "validation_criteria": VALIDATION_CRITERIA,
        "S0_unit_suite": r0, "S1_single_test_size": r1, "S2_complete_path_null": r2,
        "S6_exact_clopper_pearson": r6, "S7_bound_coverage": {"replicates": budget["s7_reps"], "results": r7},
        "S8_exact_fisher": r8,
        "S3_complete_path_power": {"sizes": SYNTH_SIZES, "global_alpha": 0.01,
                                   "injection_families": {k: v[0] for k, v in INJECTIONS.items()}, "results": r3},
        "S4_tvd_noise_floor_production": {"replicates": len(r4_reps), "law": "independent uniform bits; Y ~ Bernoulli(1/2)",
                                          "upper_bound_alpha": PER_HYPOTHESIS_ALPHA["A"], "results": r4},
        "S5_design_resolution": r5,
        "operating_conditions": {"synthetic_path_sizes": SYNTH_SIZES, "synthetic_path_tolerances": TOLERANCES_FOR_PATH,
                                 "synthetic_law": synthetic_partition.__doc__},
        "limitations": ["S2/S3 use reduced sizes (20000/5000/5000) and a synthetic stand-in law, not Speck or Gohr data.",
                        "S4 assumes independent uniform cells.",
                        ("Every guarantee assumes i.i.d. rows within each sample and, for the exact tests, independent "
                         "compared samples. D1 and D2 probe specified duplicate and dependence patterns only; they do "
                         "not establish these assumptions, and existing Gohr D1/D2 results concern other instances.")],
        "budget": budget,
        "rng": {"generator": "numpy.random.Generator(PCG64) via SeedSequence", "root_seed": ROOT_SEED, "stream_seeds": seeds},
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "environment": _environment_record(),
    }
    a.out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".d3_mv_staging_", dir=str(a.out_dir.parent)))
    try:
        (staging / EVIDENCE_NAME).write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
        manifest = {"run_id": run_id, "run_status": "COMPLETE", "status": status,
                    "files": {EVIDENCE_NAME: _sha256(staging / EVIDENCE_NAME)}}
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if a.out_dir.exists():
            raise SystemExit(f"REFUSED: {a.out_dir} appeared during the run. Nothing was written.")
        os.rename(staging, a.out_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps({"status": status, "checks": checks, "out": str(a.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
