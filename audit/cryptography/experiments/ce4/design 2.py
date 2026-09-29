"""
CE4: Ciphertext-Difference Intervention Sensitivity.

RENAMED AND RESCOPED. Historical CE4 described itself as establishing
"causal necessity of the analytical trail probability". The intervention
manipulates FINAL CIPHERTEXT-PAIR BITS; it never manipulates the
analytical single-trail probability, so that claim was stronger than the
design supports.

ESTIMAND (explicit)
    The mean within-sample difference in output-change magnitude between
    the structural-intervention policy and the magnitude-matched,
    XOR-preserving control policy, within the prespecified eligible
    population, for the tested frozen model instance.

The control is inert WITH RESPECT TO THE DECLARED XOR-DIFFERENCE TARGET -
not "cryptographically inert" in general.
"""

from __future__ import annotations

import numpy as np

EXPERIMENT_NAME = "Ciphertext-Difference Intervention Sensitivity"
ESTIMAND = (
    "mean within-sample difference in output-change magnitude between the structural "
    "intervention policy and the magnitude-matched XOR-preserving control policy, within "
    "the prespecified eligible population, for the tested frozen model instance"
)
CONTROL_DESCRIPTION = (
    "inert with respect to the declared XOR-difference target: flipping the same bit in "
    "both members of a pair preserves their XOR exactly, by (a^1)^(b^1) = a^b"
)


class InterventionError(RuntimeError):
    pass


def eligible_mask(bits_a: np.ndarray, bits_b: np.ndarray, *, n_flips: int) -> np.ndarray:
    """
    Predeclared eligibility rule.

    A sample is eligible iff it carries at least `n_flips` difference-bearing
    bit positions, i.e. positions where the two ciphertext members differ.
    For the magnitude-matched construction, `n_flips` must be 2k so the
    mirrored control can change the same number of bits.
    Both arms draw from THIS SAME pool, so eligibility cannot induce a
    selection difference between arms.
    """
    if bits_a.shape != bits_b.shape:
        raise InterventionError("ciphertext arrays differ in shape")
    return (bits_a ^ bits_b).sum(axis=1) >= n_flips


def build_matched_interventions(bits_a, bits_b, positions_2k):
    """
    Construct both arms with GENUINELY EQUAL numbers of changed bits.

    DESIGN CORRECTION (found by the magnitude invariant test): flipping k
    positions on one side changes k bits, while flipping k MIRRORED pairs
    changes 2k bits. The historical one-side-vs-mirrored-pair construction
    was therefore NOT bit-count matched (k vs 2k), so any output-change
    difference between the arms was partly attributable to the amount of
    perturbation rather than to its structure.

    Here both arms change exactly 2k bits:
        structural : flip all 2k chosen positions on side A only
                     -> XOR difference changes at 2k positions
        control    : flip the first k positions as MIRRORED PAIRS
                     -> 2k bits changed, XOR preserved exactly
    """
    positions_2k = np.asarray(positions_2k)
    if positions_2k.shape[1] % 2 != 0:
        raise InterventionError(
            "structural positions must come in an even count so the mirrored control can "
            "change the same number of bits")
    k = positions_2k.shape[1] // 2
    rows = np.arange(bits_a.shape[0])[:, None]
    s_a, s_b = bits_a.copy(), bits_b.copy()
    s_a[rows, positions_2k] ^= 1                       # 2k bits, one side
    c_a, c_b = bits_a.copy(), bits_b.copy()
    mirror = positions_2k[:, :k]
    c_a[rows, mirror] ^= 1
    c_b[rows, mirror] ^= 1                             # 2k bits, XOR preserved
    return (s_a, s_b), (c_a, c_b)


def apply_structural(bits_a, bits_b, positions):
    """Structural arm: flip ONE side at the chosen difference-bearing positions."""
    a = bits_a.copy()
    rows = np.arange(a.shape[0])[:, None]
    a[rows, positions] ^= 1
    return a, bits_b.copy()


def apply_control(bits_a, bits_b, positions):
    """
    Control arm: flip the MIRRORED pair at the same positions.

    Same number of changed bits as the structural arm, and the pair XOR is
    preserved exactly.
    """
    a, b = bits_a.copy(), bits_b.copy()
    rows = np.arange(a.shape[0])[:, None]
    a[rows, positions] ^= 1
    b[rows, positions] ^= 1
    return a, b


def verify_intervention_invariants(orig_a, orig_b, s_a, s_b, c_a, c_b) -> dict:
    """
    Per-sample invariants that must hold for EVERY analyzed sample.

    Checked before any inference: if an invariant fails the comparison is
    not magnitude-matched and the contrast is uninterpretable.
    """
    problems = []
    struct_changed = (orig_a ^ s_a).sum(axis=1) + (orig_b ^ s_b).sum(axis=1)
    control_changed = (orig_a ^ c_a).sum(axis=1) + (orig_b ^ c_b).sum(axis=1)
    if not np.array_equal(struct_changed, control_changed):
        problems.append("intervention magnitude not matched between arms")
    orig_xor = orig_a ^ orig_b
    if not np.array_equal(c_a ^ c_b, orig_xor):
        problems.append("control did NOT preserve the pair XOR difference")
    if np.array_equal(s_a ^ s_b, orig_xor):
        problems.append("structural arm did not change the XOR-difference structure")
    if problems:
        raise InterventionError("; ".join(problems))
    return {
        "magnitude_matched": True,
        "control_preserves_xor": True,
        "structural_changes_xor_structure": True,
        "bits_changed_per_sample": int(struct_changed[0]) if struct_changed.size else 0,
        "control_description": CONTROL_DESCRIPTION,
    }


def population_accounting(n_total: int, eligible: np.ndarray) -> dict:
    """
    Exclusion bookkeeping as part of the ESTIMAND, not as a failure gate.

    The historical `exclusion_rate > 0.01` abort was an arbitrary
    threshold: restricting to the eligible population defines WHO the
    claim is about, and is reported, not judged.
    """
    n_analyzed = int(np.count_nonzero(eligible))
    n_excluded = int(n_total - n_analyzed)
    return {
        "N_total": int(n_total),
        "N_excluded": n_excluded,
        "N_analyzed": n_analyzed,
        "exclusion_rate": (n_excluded / n_total) if n_total else None,
        "eligibility_rule": ("sample carries at least n_flips difference-bearing bit "
                             "positions; both arms draw from this same pool"),
        "note": ("the eligibility restriction is part of the estimand - it defines the "
                 "population the claim is about, and is not evidence of failure"),
    }


def paired_contrast(structural_delta, control_delta, *, alpha: float = 0.05) -> dict:
    """Paired per-sample contrast; Wilcoxon signed-rank with honest p reporting."""
    from scipy import stats

    from audit.cryptography.statistics import p_value_report

    s = np.asarray(list(structural_delta), dtype=float)
    c = np.asarray(list(control_delta), dtype=float)
    if s.shape != c.shape:
        raise InterventionError("structural and control arrays must be paired per sample")
    gap = s - c
    n = gap.size
    mean = float(gap.mean())
    sd = float(gap.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else None
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1) if n > 1 else None
    try:
        p = float(stats.wilcoxon(gap, alternative="greater").pvalue)
    except ValueError:
        p = None
    return {
        "experiment": EXPERIMENT_NAME,
        "estimand": ESTIMAND,
        "statistical_unit": "eligible sample (paired within-sample contrast)",
        "structural_mean": float(s.mean()), "control_mean": float(c.mean()),
        "mean_gap": mean, "sd_gap": sd,
        "ci95": [mean - tcrit * se, mean + tcrit * se] if se and tcrit else None,
        "effect_size_dz": (mean / sd) if sd > 0 else None,
        "wilcoxon": p_value_report(p) if p is not None else None,
        "claim_scope": (
            "sensitivity of THIS frozen model's output to the specified difference-bearing "
            "ciphertext structure, relative to a magnitude-matched XOR-preserving control, "
            "within the eligible population. Does NOT establish that the model uses the "
            "analytical single-trail probability."),
    }
