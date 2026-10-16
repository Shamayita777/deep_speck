#!/usr/bin/env python3
"""
AutoND Implementation Integrity successor preregistration (v2).

STATUS: NOT FROZEN, AND NOT READY TO BE FROZEN.

This plan governs one experiment: evaluating AutoND's two shipped
polished checkpoints on freshly generated test data with the target's own
evaluation harness, and comparing the result against the published
accuracies. It authorises nothing else. No training, no search, no
modification of the upstream checkout.

WHAT CHANGED IN THIS REVISION, AND WHY
    The previous revision's decision rule was "the 95% interval of the
    five-set mean contains the published value". That rule is unsound and
    is replaced here.

      * Containment of a point value by a confidence interval is not
        evidence of equivalence. It is failure to reject a difference,
        which with five observations is the expected outcome whether or
        not the checkpoint reproduces the published number. The rule was
        biased toward SUPPORTED by construction.
      * The arithmetic is not marginal. The project's own five-set
        staged-training evaluation had sample SD 6.707e-4, giving a 95%
        interval roughly 1.7e-3 wide. The paper and README disagree about
        SIMON by 3e-4. An interval that wide contains BOTH published
        values, so the old rule would have returned SUPPORTED for either
        one and could never have identified which.
      * Holm correction was named but applied to interval containment,
        which has no p-value to correct. The multiplicity statement was
        therefore inoperable.

    The replacement is a two-one-sided-tests (TOST) equivalence
    procedure against a prespecified margin, with Holm applied to real
    TOST p-values, plus a separate difference test so that "shown to
    differ" and "too imprecise to tell" are distinct outcomes rather than
    both falling into NOT_SUPPORTED. See DECISION_RULE and `decide`.

WHY IT IS STILL NOT READY
    `unverified_fields()` and `confirmatory_run_permitted()` report the
    remaining blockers from the plan itself. Freezing is a deliberate act
    performed by the operator on a machine holding the pinned checkout;
    this module refuses to do it for them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from audit.common.equivalence import (
    DIFFERENT,
    EQUIVALENT,
    INDETERMINATE,
    classify,
    decide_one,
    holm as holm_family,
    tost_power_exact,
    tost_power_mc,
)
from audit.targets.autond.evaluation_results import mean_and_t_interval

try:
    from audit.targets.autond import PINNED_UPSTREAM_COMMIT as PINNED_COMMIT
except Exception:                                                 # noqa: BLE001
    PINNED_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"

PREREGISTRATION_ID = "autond-ii-v2"

#: v1 remains immutable and in force for its own hypothesis family.
V1_PLAN_HASH = "e51ece73deaa076a27cd1f490e7f4f6287f54785e932abdc1b5da49d118bf0ed"
V1_FILE_SHA256 = "0a226e5caf7ec49aa7740923558b880b17a24426c17f63cd6e13f58f0e38d9db"

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
INCONCLUSIVE = "INCONCLUSIVE"
SCIENTIFIC_DECISIONS = (SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE)

#: Per-comparison outcomes, which are NOT hypothesis outcomes. Imported
#: from audit.common.equivalence rather than redefined, so the adapter
#: cannot drift from the framework's vocabulary.

#: Any field left at this value makes freeze() refuse.
TODO = "<<FILL BEFORE FREEZING>>"

FROZEN_SIDECAR = Path(__file__).with_suffix(".frozen.json")
PLAN_FILE = Path(__file__).resolve()

_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


# --------------------------------------------------------------------------
# SECTION 1 - the target and the artifacts under evaluation
#
# Every digest here is a REAL digest computed from the pinned checkout.
# There is no placeholder and no "verified" Boolean: a Boolean asserting
# that a hash was checked is worth nothing, because it can be set without
# checking. `verify_source_hashes()` recomputes each value against an
# actual checkout, and the gate calls it.
# --------------------------------------------------------------------------
TARGET = {
    "name": "AutoND",
    "upstream": "https://github.com/Crypto-TII/AutoND",
    "pinned_commit": PINNED_COMMIT,
    "relationship": "third-party; independently developed; not retrained by us",
    "execution_mode": ("INFERENCE ONLY on the shipped weights. No training, no "
                       "fine-tuning, no modification of the upstream checkout."),
    "checkpoints": {
        "speck3264": {
            "path": "model_weights/speck3264_single-key_0x400000_round8_polished.h5",
            "sha256": "a7be788dbbb760a7be9053748e6e0f37dedc5768db3a74a5c3ebd294dca4cc26",
            "cipher": "speck3264", "scenario": "single-key",
            "input_difference": "0x400000", "round_number": 8,
            "model_type": "dbitnet",
        },
        "simon3264": {
            "path": "model_weights/simon3264_single-key_0x400_round11_polished.h5",
            "sha256": "769becfe854acd238f45ed4ade2810bb1e454166db3f5c6d557aa817275d25af",
            "cipher": "simon3264", "scenario": "single-key",
            "input_difference": "0x400", "round_number": 11,
            "model_type": "dbitnet",
        },
    },
    #: Source files whose content defines the evaluation. Each digest was
    #: computed on the pinned checkout at commit PINNED_COMMIT and is
    #: re-verified by verify_source_hashes() before any run is permitted.
    "source_files": {
        "eval_nets.py":
            "838587127c08834f6f9ecd7b27fc578da3942141fdcf5d9c513d46c646d63048",
        "dbitnet.py":
            "b19adceb6e6a9ad39c648aa41e58d23b6a0b338c9bd2be9c4b0c7799ea631736",
        "main.py":
            "52554c2966edefdc718f298d12476e7cd5f1faf8e99eafad2fc278a523690665",
    },
    "source_hash_note": (
        "Computed with sha256 over the pinned checkout. The previous revision "
        "of this plan carried a FABRICATED digest for eval_nets.py "
        "('838587127c08834f0dfe56dbbbaa0b4a6fd3b8b73e8a0b7a0e24e8e1e4e8e8e8'), "
        "whose first 16 characters matched the real value while the remainder "
        "was padding. It is recorded here so the error is not merely deleted: a "
        "fake hash that shares a real prefix survives eyeballing, which is "
        "exactly why the gate recomputes rather than trusts."),
}


# --------------------------------------------------------------------------
# SECTION 2 - what the published sources say. Kept SEPARATE, not merged.
# --------------------------------------------------------------------------
PUBLISHED_VALUES = {
    "speck3264_round8": {
        "paper_table_6": 0.5144,
        "readme_mean": 0.5144,
        "readme_sd": 0.0003,
        "agree": True,
    },
    "simon3264_round11": {
        "paper_table_6": 0.5179,
        "readme_mean": 0.5182,
        "readme_sd": 0.0004,
        "agree": False,
        "discrepancy_note": ("The paper's Table 6 and the repository README give "
                             "DIFFERENT means for SIMON32/64: 0.5179 and 0.5182. "
                             "This plan does not choose between them. Both are "
                             "carried into the analysis and the result is compared "
                             "against each separately."),
    },
    "table_6_measurement_basis": (
        "Table 6's caption states the reported value is the mean accuracy of five "
        "evaluations on freshly generated test datasets of 10^6 samples each - the "
        "same protocol eval_nets.py implements. The reported figure is therefore a "
        "fresh-test mean, NOT a selection maximum."),
    "polished_versus_unpolished": (
        "These values are for the POLISHED checkpoints. The project's own "
        "staged-training reproduction evaluated the UNPOLISHED round-8 checkpoint "
        "against a different published figure (0.5096) and must not be compared "
        "against 0.5144. The two checkpoints have different digests: "
        "815062f1... unpolished, a7be788d... polished."),
}

HISTORICAL_UNRETAINED = {
    "status": "HISTORICAL_NO_ARTIFACT",
    "speck3264_round8": {
        "accuracies": [0.515295, 0.513953, 0.513720, 0.514351, 0.514299],
        "mean": 0.5143236, "sample_sd": 0.0006014,
    },
    "simon3264_round11": {
        "accuracies": [0.517707, 0.517114, 0.517025, 0.517875, 0.518252],
        "mean": 0.5175946, "sample_sd": 0.0005193,
    },
    "note": ("These were produced by an earlier evaluation whose raw NPZ outputs "
             "were NOT retained. They are recorded for continuity only. They are "
             "NOT confirmatory evidence, must never be described as retained "
             "artifacts, and are not used as a prior for the analysis below."),
    "planning_use_only": (
        "One number from this record IS used, and only for design: the SD scale "
        "that feeds POWER. Using an observed SD to choose a sample size or to "
        "report expected power is design, not inference; it does not enter the "
        "analysis, shift a margin, or condition any verdict. The margin is fixed "
        "by EQUIVALENCE_MARGIN below, which depends only on published values."),
}


# --------------------------------------------------------------------------
# SECTION 3 - the equivalence margin, fixed before any data exist
# --------------------------------------------------------------------------
#: Fraction of the published distinguishing advantage admitted as
#: equivalent. The margin is a proportion of (p_published - 0.5), not of
#: p_published, because the scientific quantity is the advantage over
#: random guessing; an accuracy of 0.5144 carries an advantage of 0.0144,
#: and a tolerance on 0.5144 itself would be meaningless at this scale.
MARGIN_FRACTION = 0.10


def equivalence_margin(published: float) -> float:
    """Prespecified tolerance for one published reference value."""
    return MARGIN_FRACTION * (published - 0.5)


EQUIVALENCE_MARGIN = {
    "rule": "delta = MARGIN_FRACTION * (p_published - 0.5)",
    "margin_fraction": MARGIN_FRACTION,
    "values": {
        "speck3264_round8 @ 0.5144": equivalence_margin(0.5144),
        "simon3264_round11 @ 0.5179 (paper)": equivalence_margin(0.5179),
        "simon3264_round11 @ 0.5182 (readme)": equivalence_margin(0.5182),
    },
    "justification": [
        "SCALE. The quantity of scientific interest is the distinguishing "
        "advantage p - 0.5, not p. Expressing the tolerance as a fraction of the "
        "advantage makes one rule apply across ciphers and rounds whose "
        "advantages differ by an order of magnitude, without retuning.",
        "MAGNITUDE. Ten percent of the advantage is small enough that no "
        "cryptanalytic conclusion in the paper turns on it: it does not change "
        "the number of rounds distinguishable, nor the ordering of the "
        "architectures compared, nor whether a distinguisher beats random.",
        "ABOVE REPORTING GRANULARITY. Published accuracies are given to four "
        "decimals, so rounding alone admits +-5e-5. The smallest margin here is "
        f"{equivalence_margin(0.5144):.5f}, about "
        f"{equivalence_margin(0.5144) / 5e-5:.0f} times that, so the test is not "
        "measuring the publication's rounding.",
"ABOVE THE SOURCES' OWN DISAGREEMENT. The paper and README differ about "
        "SIMON by 3e-4, and the SIMON margin is "
        f"{equivalence_margin(0.5179):.5f}. A margin below the gap could not "
        "admit equivalence to both values at once. The consequence is stated "
        "rather than hidden, and stated NARROWLY: because the margin exceeds "
        "the gap, the two registered equivalence regions overlap over most of "
        "their extent, so NEAR THE PUBLISHED VALUES a result equivalent to one "
        "is very likely equivalent to the other. This experiment is therefore "
        "not expected to resolve which published SIMON value is correct, and "
        "the SIMON hypothesis predicts INCONCLUSIVE on that basis. It is NOT "
        "claimed that the design is incapable of separating them under every "
        "conceivable outcome: a mean falling in the non-overlapping sliver, or "
        "one far enough out to be DIFFERENT from one reference while "
        "INDETERMINATE against the other, would discriminate. Those outcomes "
        "are possible but not expected.",
    ],
    "what_a_margin_is_not": (
        "The margin is not a tolerance chosen to make the result pass. It is "
        "fixed here, before the data exist, as a function of published values "
        "only. Nothing observed may change MARGIN_FRACTION; a successor plan "
        "that wants a different margin must say so and be frozen separately."),
}


# --------------------------------------------------------------------------
# SECTION 4 - the experiment
# --------------------------------------------------------------------------

#: Sets produced by ONE eval_nets.py invocation. Fixed by the target's
#: module constant nEval at eval_nets.py:22; not a choice of ours and not
#: a command-line argument, so it cannot be raised without editing the
#: pinned repository.
SETS_PER_INVOCATION = 5

#: Invocations per checkpoint. THIS is the design parameter. Two
#: invocations with distinct --add_str give ten independent test sets
#: while leaving the target untouched.
INVOCATIONS_PER_CHECKPOINT = 2

#: The registered sample size per checkpoint, n = 10.
N_TEST_SETS = SETS_PER_INVOCATION * INVOCATIONS_PER_CHECKPOINT

#: The --add_str suffix for each invocation, in order. These select the
#: NPZ filenames eval_nets.py writes (it appends the suffix before
#: '.npz' at eval_nets.py:152), so distinct values stop the second
#: invocation overwriting the first's artifact.
INVOCATION_SUFFIXES = ("_a", "_b")

assert len(INVOCATION_SUFFIXES) == INVOCATIONS_PER_CHECKPOINT
assert len(set(INVOCATION_SUFFIXES)) == INVOCATIONS_PER_CHECKPOINT

DESIGN = {
    "research_question": (
        "Do AutoND's two shipped polished checkpoints reproduce their published "
        "accuracies when evaluated on freshly generated test data using the "
        "target's own evaluation harness?"),
    "experimental_unit": (
        "One freshly generated test dataset of 10^6 labelled examples. Ten per "
        "checkpoint, obtained as TWO invocations of eval_nets.py, each of which "
        "generates SETS_PER_INVOCATION = 5 sets because its module constant "
        "nEval = 5 fixes that and nEval is not a command-line argument."),
    "n_test_sets": N_TEST_SETS,
    "sets_per_invocation": SETS_PER_INVOCATION,
    "invocations_per_checkpoint": INVOCATIONS_PER_CHECKPOINT,
    "samples_per_test_set": 10 ** 6,
    "protocol_source": (
        "Fixed by upstream eval_nets.py module constants, NOT chosen by us: "
        "nEval = 5, num_val_samples = 10**6, batchsize = 5000, decision threshold "
        "Zbin = (Z > 0.5). Nothing WITHIN an invocation is a free parameter. Each "
        "constant was confirmed against the pinned source at eval_nets.py:22-24 "
        "and :33."),
    "how_n_was_raised": (
        "Revision 3c registered n = 5. This revision registers n = 10, with the "
        "equivalence margin UNCHANGED. The sample size was raised because power "
        "at n = 5 was 0.7965 at the conservative Holm step and collapsed to 0.339 "
        "if the true SD were 1.5x the planning estimate - and the planning SD is "
        "itself a 4-df estimate whose 95% interval spans roughly 0.6x to 2.9x. "
        "Raising n improves precision; widening the margin would instead relax "
        "the criterion for agreement, which the margin's justification does not "
        "permit. See POWER['superseded_design'] for the n = 5 figures, retained."),
    "why_two_invocations_and_not_a_changed_constant": (
        "nEval = 5 is a module-level constant in the target, at eval_nets.py:22, "
        "and is NOT exposed as an argument. Reaching n = 10 by editing it would "
        "modify the pinned upstream repository, which this audit forbids "
        "absolutely. Instead eval_nets.py is invoked TWICE per checkpoint with "
        "distinct --add_str values. The target is not modified, its defaults are "
        "not changed, and the second invocation is the same command as the first "
        "but for the output-name suffix."),
    "pooling_rule": (
        "The ten accuracies are the 5 from invocation 'a' followed by the 5 from "
        "invocation 'b', pooled into one sample of size 10. The analysis is the "
        "registered TOST on that pooled sample: d = mean - p_ref, "
        "SE = s/sqrt(10), df = 9. No per-invocation means are averaged, no "
        "weighting is applied, and no between-invocation variance component is "
        "estimated or removed."),
    "pooling_assumption": (
        "Pooling treats all ten sets as exchangeable draws, which requires that "
        "an accuracy depend only on the dataset drawn and not on which process "
        "produced it. Verified against the pinned source rather than assumed: "
        "eval_nets.py:208-209 draws a FRESH dataset inside each loop iteration "
        "via make_train_data from os.urandom, so sets are independent within an "
        "invocation; the checkpoint is a fixed file whose sha256 is verified "
        "before each invocation, and evaluate_X_Y at :27-42 is a deterministic "
        "function of (model, X, Y), so a given dataset yields the same accuracy "
        "in either process. The two invocations therefore differ only in their "
        "random draws. This is a RECORDED ASSUMPTION, not a tested hypothesis: "
        "with 5 sets per invocation a between-invocation effect could not be "
        "detected with any useful power, so the design does not claim to have "
        "excluded one. The per-invocation accuracies are retained individually "
        "so a reader can inspect the two groups directly."),
    "data_generation": (
        "main.make_train_data via eval_nets, drawing plaintexts, keys and labels "
        "from os.urandom. Half the pairs are real, half replaced by random "
        "ciphertexts, as the upstream generator does."),
    "randomness_policy": (
        "NOT SEEDED, and deliberately so: eval_nets.py draws from os.urandom and "
        "seeding would require modifying the target. Each of the ten sets is an "
        "independent draw. The run is therefore NOT bit-reproducible, and the "
        "record must say so rather than implying otherwise."),
    "accuracy_computation": (
        "acc = mean(Zbin == Y) with Zbin = (prediction > 0.5), computed by "
        "eval_nets.evaluate_X_Y. TPR and TNR recorded alongside as secondary."),
    "primary_endpoint": ("mean accuracy over the ten fresh test sets, per "
                         "checkpoint, pooled across the two invocations"),
    "secondary_endpoints": [
        "per-set accuracies, retained individually and never only as a mean",
        "sample standard deviation across the ten pooled sets",
        "TPR and TNR per set (SIMON's published TPR/TNR are notably asymmetric)",
        ("the two per-invocation means and SDs, reported for inspection only; "
         "they are NOT compared to each other as a test, because the design "
         "has no power to do so and such a test was not registered"),
    ],
    "exclusion_rules": (
        "A test set is excluded only on a technical failure - a crash, a truncated "
        "NPZ, a wrong sample count - and the exclusion is recorded with its reason "
        "before the accuracy is read. No set is excluded for its value."),
    "abort_conditions": [
        "checkpoint sha256 does not match the value recorded in TARGET",
        "any file in TARGET.source_files does not match its recorded sha256",
        "upstream HEAD is not the pinned commit",
        "the evaluation writes into the upstream checkout",
        "fewer than ten sets complete for either checkpoint",
        ("either invocation of a checkpoint fails to produce its NPZ, which "
         "costs five sets at once and so cannot be made up by the other"),
    ],
    "missing_data": (
        "If fewer than ten sets complete, the result is reported as INCONCLUSIVE "
        "for that checkpoint. A partial mean is not substituted, and a single "
        "surviving invocation is NOT analysed as an n = 5 result: that would be "
        "an unregistered fallback design chosen after seeing which run failed. "
        "The remedy for a failed invocation is to rerun the whole checkpoint "
        "into a fresh run directory, with the failure and its reason recorded."),
    "environment": {
        "python": "3.11.x", "tensorflow": "2.15.1", "keras": "2.15.0",
        "numpy": "1.26.4",
        "rationale": ("The README states the code is incompatible with TensorFlow "
                      "2.16+, and dbitnet.make_model applies tf.subtract/tf.divide "
                      "to a KerasTensor, which Keras 3 rejects."),
    },
    "outputs_to_preserve": [
        "the raw .npz per checkpoint, containing accs, tprs, tnrs",
        "full stdout/stderr log of each evaluation",
        "sha256 of each checkpoint copy evaluated",
        "sha256 of each produced .npz",
        "pip freeze and python -V from the executing environment",
        "the exact command line",
        "this plan's hash and the sidecar freeze record",
    ],
    "result_bundle_schema": (
        "audit.targets.autond.evaluation_results.RESULT_SCHEMA. A bundle that "
        "does not validate against it is not ingested, and a bundle whose "
        "recorded plan_hash is not this plan's frozen hash is rejected outright."),
    "output_location": (
        "A dedicated directory OUTSIDE the upstream checkout. eval_nets.py writes "
        "its NPZ beside the model path it is given, so the checkpoints are copied "
        "out first and the copies are evaluated. The upstream checkout is never "
        "written to."),
}


# --------------------------------------------------------------------------
# SECTION 5 - the decision rule
# --------------------------------------------------------------------------
ALPHA = 0.05

COMPARISONS = (
    {"comparison_id": "CMP-SPECK-PAPER", "checkpoint": "speck3264",
     "reference": 0.5144, "reference_source": "paper Table 6 and README (they agree)"},
    {"comparison_id": "CMP-SIMON-PAPER", "checkpoint": "simon3264",
     "reference": 0.5179, "reference_source": "paper Table 6"},
    {"comparison_id": "CMP-SIMON-README", "checkpoint": "simon3264",
     "reference": 0.5182, "reference_source": "repository README"},
)

DECISION_RULE = {
    "family_size": len(COMPARISONS),
    "alpha": ALPHA,
    "test": (
        "TOST equivalence on the ten-set pooled mean, per comparison. With "
        "d = mean - p_ref, SE = s/sqrt(10) and df = 9, compute two one-sided "
        "t statistics: "
        "t_lower = (d + delta)/SE tested against the upper tail, and t_upper = "
        "(d - delta)/SE tested against the lower tail. p_TOST = max(p_lower, "
        "p_upper). Equivalence is concluded when the Holm-adjusted p_TOST < "
        "alpha, which is exactly the condition that the (1 - 2*alpha_adj) "
        "interval for d lies inside [-delta, +delta]."),
    "outside_margin_test": (
        "DIFFERENT requires evidence that the discrepancy lies OUTSIDE the "
        "margin, which is a different hypothesis from 'not zero'. Two "
        "one-sided tests at df = 9: H0: d <= +delta via t = (d - delta)/SE on "
        "the upper tail, and H0: d >= -delta via t = (d + delta)/SE on the "
        "lower tail. p_outside is the SMALLER of the two. For delta > 0 the "
        "two rejection regions are disjoint - rejecting both would require "
        "d > delta and d < -delta simultaneously - so the minimum controls "
        "type-I error at the nominal level, and the family-wise correction is "
        "applied separately by Holm."),
    "point_null_is_diagnostic_only": (
        "A two-sided test of d = 0 is computed and reported, but DOES NOT "
        "enter the decision. CORRECTION RECORDED: the previous revision "
        "classified a comparison DIFFERENT when the point-null test rejected "
        "and TOST failed. That is unsound. A discrepancy can be significantly "
        "non-zero while lying INSIDE the equivalence interval - observed mean "
        "0.515600 against reference 0.5144 gives d = +0.00120 inside "
        "delta = 0.00144, with p_tost = 0.2344 and p_diff = 0.0162, which the "
        "old rule called DIFFERENT. Failing to establish equivalence is not "
        "evidence of a meaningful discrepancy. The regression test is "
        "tests/test_r3c_statistics.TestDifferentRequiresOutsideMargin."),
    "multiplicity": (
        f"Holm-Bonferroni across the family of {len(COMPARISONS)} comparisons "
        f"({', '.join(c['comparison_id'] for c in COMPARISONS)}), at alpha = "
        f"{ALPHA}. Procedure, stated exactly: sort the family's p-values "
        "ascending as p_(1) <= ... <= p_(m) with m = 3; for i = 1..m compare "
        "p_(i) against alpha / (m - i + 1), i.e. alpha/3, alpha/2, alpha; stop "
        "at the first i that fails and reject no hypothesis from that rank "
        "onward. Holm is applied INDEPENDENTLY to the equivalence family and "
        "to the outside-margin family, because they test different null "
        "hypotheses; a p-value is never carried between the two. Each family "
        "has its own family-wise guarantee at alpha. Implemented by "
        "audit.common.equivalence.holm."),
    "per_comparison_outcomes": {
        EQUIVALENT: ("Holm-adjusted p_tost < alpha. The data exclude a true "
                     "discrepancy larger than the margin in either direction."),
        DIFFERENT: ("Holm-adjusted p_outside < alpha. The data support a true "
                    "discrepancy OUTSIDE [-delta, +delta]. Not reachable from a "
                    "point-null rejection alone."),
        INDETERMINATE: ("neither null was rejected. The data establish neither "
                        "equivalence nor an outside-margin discrepancy at this "
                        "precision. An honest statement about power, not a "
                        "result - and the correct outcome for a mean that sits "
                        "inside the margin without enough precision to prove "
                        "it."),
    },
    "mutual_exclusivity": (
        "EQUIVALENT requires the interval for d to lie inside "
        "(-delta, +delta); DIFFERENT requires it to lie wholly outside. No "
        "single dataset can satisfy both, and "
        "audit.common.equivalence.classify raises rather than silently "
        "preferring one if a future change ever makes both true."),
    "why_not_interval_containment": (
        "A 95% interval from ten observations is roughly 9.6e-4 wide at the SD "
        "scale this project has observed - and was 1.7e-3 wide at the n = 5 "
        "this design superseded. Any published value within 4.8e-4 of the mean "
        "would be 'contained', which at n = 10 still spans both disagreeing "
        "SIMON values (they differ by 3e-4), so raising n does not rescue "
        "containment. Containment therefore cannot discriminate and its apparent "
        "success carries no information. TOST inverts the logic: it requires the "
        "data to be precise enough to EXCLUDE a difference larger than the "
        "margin, so an underpowered run fails to conclude equivalence instead of "
        "appearing to confirm it."),
}

#: Power is a design statement, computed from a planning SD rather than
#: asserted. The planning SD comes from HISTORICAL_UNRETAINED and is used
#: for design only; see planning_use_only there.
PLANNING_SD = 0.00067

#: Where the planning SD comes from, and what it is NOT. Recorded as a
#: named constant so every power figure carries its provenance.
PLANNING_SD_PROVENANCE = (
    "An ASSUMED planning value, not a known population parameter and not an "
    "observed value for the checkpoints under test. It is the sample SD of "
    "this project's own five-set evaluation of the staged-training round-8 "
    "checkpoint (6.707e-4), rounded up to 6.7e-4. That is a DIFFERENT "
    "artifact - the unpolished checkpoint - so it is an estimate from a "
    "related experiment, itself carrying four degrees of freedom and "
    "therefore substantial uncertainty: a 95% interval for a SD estimated on "
    "4 df spans roughly 0.6x to 2.9x the point estimate. Power computed from "
    "it is indicative of the design, not a guarantee about this run. It is "
    "used for DESIGN only: it never enters the analysis, shifts the margin, "
    "or conditions any outcome. The observed replicate-level SD of the "
    "confirmatory run is computed from that run's own ten accuracies and is "
    "reported separately."
)


def power_at(delta: float, *, sd: float = PLANNING_SD, n: int = N_TEST_SETS,
             alpha: float = ALPHA, m: int = 1,
             true_difference: float = 0.0, monte_carlo: bool = False) -> dict:
    """
    Power of the registered TOST procedure, with the variance ESTIMATED.

    CORRECTION RECORDED. The previous revision computed
    `2*Phi(delta/se - t_crit) - 1` with se treated as a known constant.
    Two faults: it ignored the sampling variability of s, which with
    n = 5 is estimated on four degrees of freedom and is therefore a poor
    estimate; and the expression goes NEGATIVE whenever
    delta/se < t_crit, which was clamped to zero and then reported as
    though a power of zero had been computed. The margin-sensitivity
    fractions 0.02 and 0.05 were both reported as 0.000 on that basis.
    Their real values at the conservative Holm step are 0.0023 and
    0.1291.

    `m` applies the most conservative Holm step, alpha/m, so the figure
    is a LOWER BOUND across the family rather than the optimistic first
    step. That is a planning approximation and is not the registered
    decision procedure: the registered procedure is the full step-down in
    DECISION_RULE["multiplicity"], whose actual per-comparison threshold
    depends on the observed ordering and is therefore unknowable in
    advance. Reporting alpha/m is conservative by construction.

    Set monte_carlo=True for the independent simulation of the same rule.
    """
    a = alpha / m
    res = tost_power_exact(delta, sd, n, a, true_difference)
    out = {
        "delta": delta, "sd": sd, "n": n, "df": n - 1,
        "alpha": alpha, "holm_m": m, "alpha_adjusted": a,
        "t_critical": res["t_critical"],
        "power": res["power"],
        "true_difference": true_difference,
        "method": res["method"],
        "alpha_m_is_planning_only": (
            "alpha/m is the most conservative Holm step, used for planning. "
            "The registered decision procedure is the full step-down."),
        "planning_sd_provenance": PLANNING_SD_PROVENANCE,
    }
    if monte_carlo:
        mc = tost_power_mc(delta, sd, n, a, true_difference, n_sim=200_000)
        out["monte_carlo"] = {
            "power": mc["power"], "n_sim": mc["n_sim"], "seed": mc["seed"],
            "interval": mc["mc_interval"], "confidence": mc["confidence"],
            "method": mc["method"],
        }
    return out


def margin_sensitivity(fractions=(0.02, 0.05, 0.10, 0.15, 0.20),
                       reference: float = 0.5144, m: int = len(COMPARISONS)
                       ) -> list[dict]:
    """
    Power across candidate margin fractions, COMPUTED on every call.

    Nothing here is transcribed. The previous revision's table was quoted
    from the broken approximation; this recomputes it, so a future change
    to the power calculation cannot leave a stale table behind.
    """
    rows = []
    for frac in fractions:
        d = frac * (reference - 0.5)
        rows.append({
            "fraction": frac,
            "delta": d,
            "power_at_alpha": tost_power_exact(d, PLANNING_SD, N_TEST_SETS,
                                              ALPHA)["power"],
            "power_at_conservative_holm_step":
                tost_power_exact(d, PLANNING_SD, N_TEST_SETS,
                                 ALPHA / m)["power"],
            "is_registered_margin": abs(frac - MARGIN_FRACTION) < 1e-12,
        })
    return rows


POWER = {
    "planning_sd": PLANNING_SD,
    "planning_sd_provenance": PLANNING_SD_PROVENANCE,
    "n_test_sets": N_TEST_SETS,
    "sets_per_invocation": SETS_PER_INVOCATION,
    "invocations_per_checkpoint": INVOCATIONS_PER_CHECKPOINT,
    "superseded_design": (
        "Revision 3c registered n = 5 (df = 4), giving power 0.9732 at the "
        "nominal alpha and 0.7965 at the conservative Holm step alpha/3 for the "
        "registered margin. Those figures are retained here rather than deleted: "
        "the margin did not change, so they remain the honest record of what the "
        "earlier design could detect. Under a planning SD 1.5x the estimate, "
        "n = 5 power at the conservative step fell to 0.339 and at 2.0x to 0.129; "
        "this fragility, not the point estimate, is why n was raised. The "
        "sensitivity figures are recomputed by margin_sensitivity() and are not "
        "transcribed."),
    "robustness_note": (
        "The planning SD carries 4 df, so its own 95% interval spans roughly "
        "0.6x to 2.9x the point estimate. Power is therefore reported as a "
        "design indication across a range of SD multiples, not as a single "
        "number. At the registered n = 10 the design retains power above 0.9 at "
        "1.5x the planning SD; at 2.9x - the pessimistic end of the SD interval - "
        "no feasible n rescues the design, and the honest outcome in that case is "
        "INDETERMINATE, which the decision rule returns rather than defaulting to "
        "agreement."),
    "method": ("audit.common.equivalence.tost_power_exact: P(reject | s) "
               "integrated over the chi distribution of s, so the variance is "
               "estimated from the sample as the analysis will estimate it. "
               "Cross-checked against tost_power_mc, a Monte Carlo simulation "
               "of the same rule with a fixed recorded seed."),
    "superseded_method": (
        "Revision 3 used 2*Phi(delta/se - t_crit) - 1 with se held fixed. It "
        "treated the SD as known and returned negative values that were "
        "clamped to zero; fractions 0.02 and 0.05 were reported as 0.000 "
        "power, which was an artifact rather than a calculation."),
    "note": ("Computed by margin_sensitivity() on every call, never "
             "transcribed. Reported at both the nominal alpha and the most "
             "conservative Holm step alpha/3; the latter is a planning lower "
             "bound, not the registered decision procedure."),
    "power_does_not_justify_a_margin": (
        "Power informs whether a design can detect what it is looking for. It "
        "does NOT make a tolerance scientifically acceptable: a margin wide "
        "enough to guarantee power could be wide enough to admit a "
        "discrepancy that matters. The margin's justification is in "
        "EQUIVALENCE_MARGIN and rests on the published advantage, the "
        "reporting granularity and the sources' own disagreement; power only "
        "rules out margins so tight that the experiment could conclude "
        "nothing."),
}

HYPOTHESES = {
    "H-AUTOND-SPECK-REPRO": {
        "question": ("Does the shipped SPECK32/64 round-8 checkpoint reproduce the "
                     "published accuracy on fresh test data, to within the "
                     "prespecified margin?"),
        "comparisons": ["CMP-SPECK-PAPER"],
        "reference_values": {"paper_table_6": 0.5144, "readme": "0.5144 +- 0.0003"},
        "margin": equivalence_margin(0.5144),
        "prediction": SUPPORTED,
        "rationale": ("The paper and README agree for SPECK, Table 6's figure is "
                      "itself a five-set fresh-test mean, and the project's own "
                      "unpolished round-8 reproduction landed within 1.4e-4 of "
                      "its own Table 6 counterpart (0.5096). At the planning SD "
                      "and the registered n = 10 the margin is about 6.8 "
                      "standard errors wide (it was 4.8 at n = 5), so the "
                      "design has adequate power to conclude equivalence if it "
                      "holds."),
        "supported_if": ("CMP-SPECK-PAPER is EQUIVALENT: Holm-adjusted p_TOST < "
                         "alpha against 0.5144 with margin "
                         f"{equivalence_margin(0.5144):.5f}"),
        "not_supported_if": ("CMP-SPECK-PAPER is DIFFERENT: Holm-adjusted p_diff "
                             "< alpha and equivalence not concluded"),
        "inconclusive_if": ("CMP-SPECK-PAPER is INDETERMINATE, or fewer than ten "
                            "sets complete"),
    },
    "H-AUTOND-SIMON-REPRO": {
        "question": ("Does the shipped SIMON32/64 round-11 checkpoint reproduce a "
                     "published accuracy on fresh test data, and can this "
                     "experiment identify which of the two disagreeing published "
                     "values it matches?"),
        "comparisons": ["CMP-SIMON-PAPER", "CMP-SIMON-README"],
        "reference_values": {"paper_table_6": 0.5179, "readme": "0.5182 +- 0.0004"},
        "margin": equivalence_margin(0.5179),
        "prediction": INCONCLUSIVE,
        "rationale": ("The two published values differ by 3e-4 while the margin "
                      f"is {equivalence_margin(0.5179):.5f}, so a result "
                      "equivalent to one is almost certainly equivalent to the "
                      "other. The design is expected to establish that the "
                      "checkpoint reproduces A published value while being unable "
                      "to say WHICH. Predicting that in advance is more honest "
                      "than predicting a match to whichever turns out closer, and "
                      "it is a consequence of the margin fixed above rather than "
                      "a hedge."),
        "supported_if": ("exactly one of CMP-SIMON-PAPER and CMP-SIMON-README is "
                         "EQUIVALENT, identifying which published value the "
                         "checkpoint reproduces"),
        "not_supported_if": ("both comparisons are DIFFERENT: the checkpoint "
                             "reproduces neither published value"),
        "inconclusive_if": ("both are EQUIVALENT (the experiment cannot separate "
                            "them, which is the predicted outcome), or either is "
                            "INDETERMINATE, or fewer than ten sets complete"),
    },
}

SCOPE_LIMITS = [
    "inference only; the shipped weights are never retrained or fine-tuned",
    "a result about the shipped checkpoints is not a result about the training "
    "pipeline that produced them, nor about the search that chose their difference",
    "ten fresh test sets estimate THIS checkpoint's accuracy precisely; they say "
    "nothing about variation across independently trained models",
    "loading and evaluating a checkpoint is not evidence of its provenance",
    "equivalence within the margin is not identity: a result may be equivalent to "
    "a published value and still differ from it by up to the margin",
    "this experiment cannot adjudicate the paper-versus-README SIMON discrepancy; "
    "the margin exceeds the gap, by necessity and by prior declaration",
    "this plan does not amend, reinterpret or supersede frozen v1",
]

PLAN = {
    "preregistration": PREREGISTRATION_ID,
    "supersedes": None,
    "extends": {"v1_plan_hash": V1_PLAN_HASH, "v1_file_sha256": V1_FILE_SHA256,
                "relationship": ("v1 remains immutable and in force for its own "
                                 "hypothesis family; this plan covers an "
                                 "experiment v1 does not")},
    "target": TARGET,
    "published_values": PUBLISHED_VALUES,
    "historical_unretained": HISTORICAL_UNRETAINED,
    "equivalence_margin": EQUIVALENCE_MARGIN,
    "design": DESIGN,
    "comparisons": list(COMPARISONS),
    "decision_rule": DECISION_RULE,
    "power": POWER,
    "hypotheses": HYPOTHESES,
    "scope_limits": SCOPE_LIMITS,
    "alpha": ALPHA,
    "evidence_tier": "confirmatory",
    "immutability": (
        "Predictions are never revised in response to results. A wrong prediction "
        "is retained and reported as wrong. Amendments are made only by a successor "
        "file recording what changed and why."),
    "permitted_deviations": [
        "a different CPU or OS, recorded in the environment manifest",
        "re-running a test set that failed technically, with the failure recorded",
    ],
    "deviation_procedure": (
        "Any deviation is appended to the run record with its UTC time, what "
        "changed, why, and the effect on interpretation. Earlier entries are never "
        "edited."),
}


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def plan_hash() -> str:
    return hashlib.sha256(_canonical(PLAN).encode("utf-8")).hexdigest()


def plan_file_sha256() -> str:
    return hashlib.sha256(PLAN_FILE.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# The decision rule, executable
# --------------------------------------------------------------------------

def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """
    Holm-Bonferroni over a family. Delegates to the generic framework
    implementation so there is exactly one copy of the procedure.
    """
    return holm_family(pvalues, alpha)


#: Relative floor below which a sample SD is treated as degenerate
#: rather than as a real measurement.
#:
#: WHY A FLOOR AND NOT `sd == 0`. The guard used to be `if not se:`,
#: which is exact-zero. Whether ten identical accuracies sum to exactly
#: n*x is a floating-point accident that differs between interpreter
#: versions: on CPython 3.13 sum([0.5144]*10)/10 is exactly 0.5144 and
#: the SD is 0.0, while on CPython 3.11 - the REGISTERED environment -
#: the sum is 5.144000000000001, the SD is 1.17e-16 and the exact-zero
#: guard does not fire. The TOST then ran with SE = 3.7e-17 and returned
#: EQUIVALENT on degenerate data: a maximally confident verdict from a
#: sample carrying no information. This floor removes the dependence on
#: that accident.
#:
#: WHY 1e-12. Each accuracy is a mean of 10^6 Bernoulli trials, whose own
#: standard deviation near p = 0.5 is about 5e-4, so the cross-set SD of
#: genuinely independent draws is of that order. Float noise from summing
#: ten values of this magnitude is of order 1e-15 relative. 1e-12 sits
#: three orders above the noise and nine orders below any physically
#: possible real value, so it cannot reclassify a real result in either
#: direction. It is a numerical guard, not a statistical threshold, and
#: it is deliberately NOT tunable per comparison.
DEGENERATE_SD_RELATIVE_FLOOR = 1e-12


def _degenerate_sd_floor(mean) -> float:
    """The absolute SD below which the sample is treated as degenerate."""
    scale = abs(mean) if mean else 1.0
    return scale * DEGENERATE_SD_RELATIVE_FLOOR


def _variance_is_degenerate(mean, sd) -> bool:
    """
    True when the sample SD is zero or indistinguishable from zero.

    Returns True for a missing SD as well: no SD means no t statistic,
    which is the same refusal.
    """
    if sd is None:
        return True
    return float(sd) <= _degenerate_sd_floor(mean)


def decide(accuracies_by_checkpoint: dict[str, list[float]],
           alpha: float = ALPHA) -> dict:
    """
    Apply this plan's decision rule to observed per-set accuracies.

    Pure function of its input: reads no file, consults no gate, writes
    nothing. A checkpoint with fewer than DESIGN["n_test_sets"]
    accuracies yields INDETERMINATE by the missing-data rule rather than
    a computed verdict.

    THE RULE, after the revision-3c correction:

        EQUIVALENT     Holm-adjusted p_tost < alpha over the equivalence
                       family. The data exclude |d| > delta.
        DIFFERENT      Holm-adjusted p_outside < alpha over the
                       outside-margin family. The data support
                       |d| > delta.
        INDETERMINATE  neither.

    The point-null test of d = 0 is computed and reported but takes no
    part in the decision. Revision 3 used it for DIFFERENT, which let a
    discrepancy INSIDE the margin be reported as a meaningful difference
    whenever it was merely distinguishable from zero.
    """
    n_required = DESIGN["n_test_sets"]
    per_comparison: dict[str, dict] = {}
    tost_p: dict[str, float] = {}
    outside_p: dict[str, float] = {}

    for cmp_ in COMPARISONS:
        cid = cmp_["comparison_id"]
        raw = accuracies_by_checkpoint.get(cmp_["checkpoint"], [])
        delta = equivalence_margin(cmp_["reference"])
        rec: dict = {"comparison_id": cid, "checkpoint": cmp_["checkpoint"],
                     "reference": cmp_["reference"],
                     "reference_source": cmp_["reference_source"],
                     "margin": delta, "n": len(raw)}

        bad = [a for a in raw
               if not isinstance(a, (int, float)) or isinstance(a, bool)
               or not math.isfinite(float(a))]
        if bad:
            rec.update({"outcome": INDETERMINATE,
                        "reason": f"non-numeric or non-finite accuracies: {bad}"})
            per_comparison[cid] = rec
            continue
        accs = [float(a) for a in raw]

        if len(accs) < n_required:
            rec.update({"outcome": INDETERMINATE,
                        "reason": (f"{len(accs)} of {n_required} test sets; the "
                                   "missing-data rule forbids a partial mean")})
            per_comparison[cid] = rec
            continue

        stats = mean_and_t_interval(accs, confidence=1.0 - 2.0 * alpha)
        mean, sd, se, df = (stats["mean"], stats["sample_sd"],
                            stats["se"], stats["df"])
        if se is None or _variance_is_degenerate(mean, sd):
            rec.update({
                "mean": mean, "sample_sd": sd, "se": se, "df": df,
                "outcome": INDETERMINATE,
                "reason": ("degenerate variance across the ten sets "
                           f"(sample SD {sd!r}); no usable t statistic. "
                           "Independent draws from os.urandom over 10^6 "
                           "samples each cannot produce accuracies this "
                           "close, so this indicates one dataset evaluated "
                           "repeatedly or a harness fault, NOT an unusually "
                           "precise measurement."),
                "degenerate_variance": True,
                "degenerate_variance_floor": _degenerate_sd_floor(mean),
            })
            per_comparison[cid] = rec
            continue

        d = mean - cmp_["reference"]
        stat = decide_one(d=d, se=se, df=df, delta=delta)
        rec.update({"mean": mean, "sample_sd": sd, "se": se, "df": df,
                    "difference_from_reference": d,
                    "mean_inside_margin": stat["mean_inside_margin"],
                    "t_lower": stat["t_lower"], "t_upper": stat["t_upper"],
                    "p_lower": stat["p_lower"], "p_upper": stat["p_upper"],
                    "p_tost": stat["p_tost"],
                    "t_above": stat["t_above"], "t_below": stat["t_below"],
                    "p_above": stat["p_above"], "p_below": stat["p_below"],
                    "p_outside": stat["p_outside"],
                    "outside_direction": stat["direction"],
                    "t_diff": stat["t_diff"], "p_diff": stat["p_diff"],
                    "p_diff_role": stat["role"]})
        tost_p[cid] = stat["p_tost"]
        outside_p[cid] = stat["p_outside"]
        per_comparison[cid] = rec

    # Two families, each with its own family-wise guarantee at alpha.
    tost_holm = holm_family(tost_p, alpha) if tost_p else {}
    outside_holm = holm_family(outside_p, alpha) if outside_p else {}

    for cid, rec in per_comparison.items():
        if "outcome" in rec:
            continue
        eq_reject = tost_holm.get(cid, {}).get("reject", False)
        out_reject = outside_holm.get(cid, {}).get("reject", False)
        rec["holm_equivalence"] = tost_holm.get(cid)
        rec["holm_outside_margin"] = outside_holm.get(cid)
        rec["outcome"] = classify(equivalence_rejected=eq_reject,
                                  outside_rejected=out_reject)

    # --- hypothesis level ------------------------------------------------
    hypotheses: dict[str, dict] = {}
    for name, h in HYPOTHESES.items():
        outs = [per_comparison[c]["outcome"] for c in h["comparisons"]
                if c in per_comparison]
        n_equiv = outs.count(EQUIVALENT)
        n_diff = outs.count(DIFFERENT)
        n_indet = outs.count(INDETERMINATE)
        if len(h["comparisons"]) == 1:
            decision = (SUPPORTED if n_equiv == 1
                        else NOT_SUPPORTED if n_diff == 1
                        else INCONCLUSIVE)
        else:
            if n_equiv == 1 and n_indet == 0:
                decision = SUPPORTED
            elif outs and n_diff == len(outs):
                decision = NOT_SUPPORTED
            else:
                decision = INCONCLUSIVE
        hypotheses[name] = {
            "decision": decision,
            "prediction": h["prediction"],
            "prediction_held": decision == h["prediction"],
            "comparison_outcomes": {c: per_comparison[c]["outcome"]
                                    for c in h["comparisons"]
                                    if c in per_comparison},
        }

    return {"alpha": alpha, "family_size": len(COMPARISONS),
            "rule_version": "autond-ii-v2-decision-r3c",
            "per_comparison": per_comparison,
            "holm_equivalence": tost_holm,
            "holm_outside_margin": outside_holm,
            "hypotheses": hypotheses,
            "note": ("Outcomes are computed from the frozen rule. DIFFERENT "
                     "requires rejecting H0: |d| <= delta, never a point-null "
                     "rejection. A prediction that did not hold is reported as "
                     "such and never revised.")}


# --------------------------------------------------------------------------
# Verification of the plan's own recorded facts
# --------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_source_hashes(upstream: Optional[Path]) -> tuple[bool, list[str]]:
    """
    Recompute every digest in TARGET against an actual checkout.

    This replaces the previous revision's `verified: True` Boolean. A
    Boolean records a claim; this records a computation. Without a
    checkout to check against, the answer is False with a reason, never a
    pass by default.
    """
    problems: list[str] = []
    if upstream is None:
        return False, ["no upstream checkout supplied; source digests were not "
                       "verified. A plan may not be frozen on an unchecked "
                       "digest."]
    up = Path(upstream).expanduser()
    if not up.is_dir():
        return False, [f"no checkout at {up}"]

    for rel, expected in sorted(TARGET["source_files"].items()):
        p = up / rel
        if not p.is_file():
            problems.append(f"source file {rel} missing from {up}")
            continue
        got = _sha256_file(p)
        if got != expected:
            problems.append(f"{rel}: sha256 {got} != recorded {expected}")

    for cid, rec in sorted(TARGET["checkpoints"].items()):
        p = up / rec["path"]
        if not p.is_file():
            problems.append(f"checkpoint {cid} missing at {rec['path']}")
            continue
        got = _sha256_file(p)
        if got != rec["sha256"]:
            problems.append(f"checkpoint {cid}: sha256 {got} != recorded "
                            f"{rec['sha256']}")

    ok, head, detail = verify_upstream_commit(up)
    if not ok:
        problems.append(f"upstream commit: {detail}")
    return (not problems), problems


def verify_upstream_commit(upstream: Path) -> tuple[bool, str, str]:
    """HEAD of the checkout must be the pinned commit."""
    try:
        res = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
    except Exception as exc:                                        # noqa: BLE001
        return False, "", f"git could not be run: {type(exc).__name__}: {exc}"
    if res.returncode != 0:
        return False, "", f"git rev-parse failed: {res.stderr.strip()}"
    head = res.stdout.strip()
    if head != TARGET["pinned_commit"]:
        return False, head, (f"checkout is at {head}, not the pinned "
                             f"{TARGET['pinned_commit']}")
    return True, head, ""


def _placeholders(obj, path="") -> list:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _placeholders(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += _placeholders(v, f"{path}[{i}]")
    elif obj == TODO:
        found.append(path.lstrip("."))
    return found


def unverified_fields(upstream: Optional[Path] = None) -> list[str]:
    """
    What must still be confirmed before freezing.

    With no checkout supplied, the source digests are unverified by
    definition and are reported as such.
    """
    out: list[str] = []
    for rec in TARGET["source_files"]:
        if not _SHA256.match(TARGET["source_files"][rec] or ""):
            out.append(f"target.source_files['{rec}'] is not a sha256")
    ok, problems = verify_source_hashes(upstream)
    if not ok:
        out.extend(problems)
    return out


def read_freeze_record() -> Optional[dict]:
    if not FROZEN_SIDECAR.is_file():
        return None
    try:
        return json.loads(FROZEN_SIDECAR.read_text())
    except json.JSONDecodeError:
        return None


def _git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        res = subprocess.run(["git", "-C", str(cwd)] + args,
                             capture_output=True, text=True, timeout=30)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as exc:                                        # noqa: BLE001
        return 127, "", f"{type(exc).__name__}: {exc}"


def _git_bytes(args: list[str], cwd: Path) -> tuple[int, bytes, str]:
    """
    Run git and return stdout as RAW BYTES, unstripped.

    `_git` decodes to text and strips, which is right for a commit id and
    wrong for file content: a stripped, newline-decoded blob cannot hash
    to the file's byte digest, so comparing it against a recorded sha256
    would fail for every file that ends in a newline - that is, all of
    them.
    """
    try:
        res = subprocess.run(["git", "-C", str(cwd)] + args,
                             capture_output=True, timeout=30)
        return (res.returncode, res.stdout,
                res.stderr.decode("utf-8", "replace").strip())
    except Exception as exc:                                        # noqa: BLE001
        return 127, b"", f"{type(exc).__name__}: {exc}"


def discover_repo_root(start: Optional[Path] = None) -> Path:
    """
    The repository root, asked of git rather than assumed.

    Revision 2 defaulted to PLAN_FILE.parents[4] and generated `git add`
    commands using only the file's basename, which are wrong from the
    repository root. The root is now read from git when git can answer,
    so the generated commands carry real repository-relative paths.
    """
    base = Path(start) if start else PLAN_FILE.parent
    code, out, _err = _git(["rev-parse", "--show-toplevel"], base)
    if code == 0 and out:
        return Path(out)
    return PLAN_FILE.parents[4]


def package_relative(path: Path) -> str:
    """
    The path a file has INSIDE the audit package, derived from the layout.

    Used when the file is not under the root it was asked about - which
    happens when a test or a caller passes an unrelated directory. The
    previous fallback returned the bare basename, producing
    `git add autond_ii_v2.py`, a command that does nothing from the
    repository root and silently hid the mistake.
    """
    p = Path(path).resolve()
    parts = p.parts
    if "audit" in parts:
        i = len(parts) - 1 - parts[::-1].index("audit")
        return "/".join(parts[i:])
    return p.name


def repo_relative(path: Path, repo_root: Optional[Path] = None) -> str:
    """Path relative to the repository root, for use in a git command."""
    root = Path(repo_root) if repo_root else discover_repo_root()
    p = Path(path).resolve()
    try:
        return str(p.relative_to(Path(root).resolve()))
    except ValueError:
        return package_relative(p)


def verify_sidecar_committed(repo_root: Optional[Path] = None
                             ) -> tuple[bool, list[str]]:
    """
    The sidecar must be committed, and the committed copy must be the one
    on disk.

    An uncommitted sidecar is a note to oneself: it can be written,
    deleted or edited with no trace, so it cannot establish that the plan
    was frozen before the run. A sidecar that differs from its committed
    version is worse, because the repository and the filesystem then
    disagree about what was approved.
    """
    problems: list[str] = []
    root = Path(repo_root) if repo_root else discover_repo_root()
    rel = repo_relative(FROZEN_SIDECAR, root)

    code, _out, err = _git(["ls-files", "--error-unmatch", "--", rel], root)
    if code != 0:
        problems.append(
            f"the freeze record {rel} is not tracked by git ({err or 'untracked'}). "
            "An uncommitted sidecar cannot establish that the plan was frozen "
            "before the run; commit it.")
        return False, problems

    code, blob, err = _git_bytes(["show", f"HEAD:{rel}"], root)
    if code != 0:
        problems.append(f"{rel} is tracked but not present in HEAD "
                        f"({err or 'not found'}); commit the sidecar.")
        return False, problems
    try:
        on_disk = FROZEN_SIDECAR.read_bytes()
    except OSError as exc:
        problems.append(f"{rel} could not be read from disk: {exc}")
        return False, problems
    if hashlib.sha256(blob).hexdigest() != hashlib.sha256(on_disk).hexdigest():
        problems.append(
            f"the freeze record on disk differs from the committed version of "
            f"{rel}. The repository and the filesystem disagree about what was "
            "approved; commit the change or restore the committed copy.")
        return False, problems
    return True, []


def verify_freeze_commit(rec: dict, repo_root: Optional[Path] = None
                         ) -> tuple[bool, list[str]]:
    """
    Confirm the sidecar's commit is real and actually contains the frozen
    plan file.

    THE SELF-REFERENCE PROBLEM, AND HOW IT IS SOLVED. The previous
    revision told the operator to record `git rev-parse HEAD` *after*
    committing the sidecar. That is impossible: committing the sidecar
    changes HEAD, so the value recorded can never be the commit that
    contains it. The fix is to bind the commit that contains the PLAN
    FILE, which is created before the sidecar, and to verify it by
    reading the plan file back out of that commit:

        git show <plan_file_commit>:<plan_file_path>

    and checking that it hashes to the recorded plan_file_sha256. The
    sidecar is then committed afterwards in a separate commit, and
    nothing needs to know that commit's hash.
    """
    problems: list[str] = []
    root = Path(repo_root) if repo_root else PLAN_FILE.parents[4]

    commit = str(rec.get("plan_file_commit") or "")
    if not _COMMIT.match(commit):
        problems.append(
            f"plan_file_commit {commit!r} is not a full 40-character sha. A "
            "shortened or invented commit is not acceptable.")
        return False, problems

    code, _out, err = _git(["cat-file", "-e", f"{commit}^{{commit}}"], root)
    if code != 0:
        problems.append(f"plan_file_commit {commit[:12]} does not exist in the "
                        f"repository at {root}: {err or 'not found'}")
        return False, problems

    rel = rec.get("plan_file_path")
    if not rel:
        problems.append("sidecar does not record plan_file_path, so the commit "
                        "cannot be checked against the plan file")
        return False, problems

    code, blob, err = _git_bytes(["show", f"{commit}:{rel}"], root)
    if code != 0:
        problems.append(f"plan file {rel} is not present in commit "
                        f"{commit[:12]}: {err or 'not found'}")
        return False, problems

    recorded = str(rec.get("plan_file_sha256") or "")
    if not _SHA256.match(recorded):
        problems.append(f"plan_file_sha256 {recorded!r} is not a sha256")
        return False, problems
    got = hashlib.sha256(blob).hexdigest()
    if got != recorded:
        problems.append(
            f"the plan file inside commit {commit[:12]} hashes to {got[:16]}... "
            f"but the sidecar records {recorded[:16]}.... The sidecar does not "
            "describe that commit's plan.")
    return (not problems), problems


def is_frozen(repo_root: Optional[Path] = None) -> tuple[bool, str]:
    """
    The executable gate. True only when a sidecar exists, is complete,
    binds the CURRENT plan hash, carries a well-formed UTC timestamp, and
    names a real commit that actually contains the frozen plan file.
    """
    rec = read_freeze_record()
    if rec is None:
        return False, (f"no freeze record at {FROZEN_SIDECAR.name}. Run this module "
                       "to obtain the plan hash and the exact freeze procedure.")
    required = ("plan_hash", "frozen_at_utc", "plan_file_commit",
                "plan_file_path", "plan_file_sha256", "approved_by")
    missing = [k for k in required if not str(rec.get(k) or "").strip()]
    if missing:
        return False, f"freeze record is incomplete: missing {missing}"

    live = plan_hash()
    if rec["plan_hash"] != live:
        return False, (f"freeze record binds plan hash {rec['plan_hash'][:16]}... but "
                       f"the live plan hashes to {live[:16]}.... The plan changed "
                       "after it was frozen; re-freeze deliberately or restore it.")

    # THE PLAN HASH IS NOT ENOUGH. plan_hash() covers the declarative PLAN
    # object only, so editing the executable logic in this file - decide(),
    # holm(), equivalence_margin(), power_at(), the gate itself - leaves it
    # unchanged. Revision 2 stopped here, which meant the decision rule
    # could be rewritten after freezing without the gate noticing. The
    # frozen record therefore also binds this FILE's bytes, and the live
    # file must still match them.
    live_file = plan_file_sha256()
    if rec["plan_file_sha256"] != live_file:
        return False, (
            f"freeze record binds plan FILE sha256 {rec['plan_file_sha256'][:16]}... "
            f"but the file on disk hashes to {live_file[:16]}.... The plan "
            "module's code changed after it was frozen, even though the "
            "declarative PLAN object did not. The decision rule, the margin and "
            "this gate all live in that code; re-freeze deliberately or restore "
            "the frozen file.")

    ts = str(rec["frozen_at_utc"])
    if not _ISO_UTC.match(ts):
        return False, (f"frozen_at_utc {ts!r} is not an ISO-8601 UTC instant of the "
                       "form YYYY-MM-DDTHH:MM:SSZ. A freeze time that cannot be "
                       "parsed cannot establish that the plan preceded the run.")

    ok, problems = verify_freeze_commit(rec, repo_root)
    if not ok:
        return False, "; ".join(problems)

    ok, problems = verify_sidecar_committed(repo_root)
    if not ok:
        return False, "; ".join(problems)

    return True, (f"frozen {ts} at plan-file commit "
                  f"{rec['plan_file_commit'][:12]} by {rec['approved_by']}")


def confirmatory_run_permitted(upstream: Optional[Path] = None,
                               repo_root: Optional[Path] = None
                               ) -> tuple[bool, list[str]]:
    """
    The gate the runner and the evaluation launcher consult.

    Fails closed: with no checkout to verify against, it refuses.
    """
    blockers: list[str] = []
    gaps = _placeholders(PLAN)
    if gaps:
        blockers.append(f"plan still contains placeholders: {gaps}")
    unver = unverified_fields(upstream)
    if unver:
        blockers.append(f"unverified required fields: {unver}")
    ok, detail = is_frozen(repo_root)
    if not ok:
        blockers.append(f"not frozen: {detail}")
    return (not blockers), blockers


def freeze_instructions(repo_root: Optional[Path] = None) -> str:
    """The exact, non-self-referential freeze procedure."""
    digest = plan_hash()
    root = Path(repo_root) if repo_root else discover_repo_root()
    plan_rel = repo_relative(PLAN_FILE, root)
    side_rel = repo_relative(FROZEN_SIDECAR, root)
    return f"""\
FREEZE PROCEDURE (two commits, in this order)

  1. Finish editing this file. Do not edit it again after step 2.

  2. Commit the plan file BY ITSELF, then read back its commit:

       cd {root}
       git add {plan_rel}
       git commit -m "Freeze {PREREGISTRATION_ID}"
       PLAN_COMMIT=$(git rev-parse HEAD)
       PLAN_SHA=$(shasum -a 256 {plan_rel} | cut -d" " -f1)

     PLAN_COMMIT is the commit that CONTAINS the plan. It is knowable
     because the sidecar does not exist yet. This is the whole point of
     splitting the two commits: the earlier procedure asked for the hash
     of the commit containing the sidecar, which the sidecar would then
     have to contain, and no such value exists.

  3. Write {FROZEN_SIDECAR.name} with those real values:

{json.dumps({
    "preregistration": PREREGISTRATION_ID,
    "plan_hash": digest,
    "frozen_at_utc": "<date -u +%Y-%m-%dT%H:%M:%SZ>",
    "plan_file_commit": "<$PLAN_COMMIT>",
    "plan_file_path": plan_rel,
    "plan_file_sha256": "<$PLAN_SHA>",
    "approved_by": "<your name>",
}, indent=6)}

     plan_file_sha256 is the sha256 of the FILE BYTES, which is what
     is_frozen() recomputes against the file on disk. It binds the
     EXECUTABLE LOGIC in this module - decide(), holm(),
     equivalence_margin(), power_at() - not just the declarative PLAN
     object, so the decision rule cannot be rewritten after freezing.
     Do not use `git hash-object`: it prints git's own blob id, a
     different value.

     plan_file_path must be the repository-relative path printed above,
     because is_frozen() reads the plan back out of the commit with
     `git show <commit>:<plan_file_path>`.

  4. Commit the sidecar in a SECOND commit. is_frozen() requires it to be
     tracked AND to match its committed version:

       git add {side_rel}
       git commit -m "Record freeze record for {PREREGISTRATION_ID}"

  5. Confirm the gate now opens:

       python3 -m audit.targets.autond.preregistrations.autond_ii_v2 \\
           --autond-root /path/to/pinned/AutoND

Do not write a commit hash you have not read from git, and do not
back-date frozen_at_utc.
"""


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Report this preregistration's hash and freeze readiness. "
                    "Executes no evaluation and freezes nothing.")
    ap.add_argument("--autond-root", type=Path, default=None,
                    help="pinned upstream checkout, against which every recorded "
                         "digest is recomputed")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="repository root, for verifying the freeze commit")
    args = ap.parse_args(argv)

    digest = plan_hash()
    print(f"AutoND Implementation Integrity successor preregistration "
          f"({PREREGISTRATION_ID})\n")
    print(f"  plan hash          {digest}")
    print(f"  plan file sha256   {plan_file_sha256()}")
    print(f"  comparisons        {len(COMPARISONS)}  "
          f"(Holm family size {len(COMPARISONS)}, alpha {ALPHA})")
    print(f"  equivalence margin delta = {MARGIN_FRACTION} * (p_published - 0.5)")
    for k, v in EQUIVALENCE_MARGIN["values"].items():
        print(f"      {k:40} delta = {v:.5f}")
    print(f"\n  hypotheses         {len(HYPOTHESES)}")
    for name, h in HYPOTHESES.items():
        print(f"      {name:26} predicts {h['prediction']}")

    print("\n  POWER (exact; variance estimated from the sample, not assumed)")
    print(f"      planning SD {PLANNING_SD} - an ASSUMED value, see "
          "PLANNING_SD_PROVENANCE")
    for label, ref in (("SPECK @ 0.5144", 0.5144), ("SIMON @ 0.5179", 0.5179)):
        pw = power_at(equivalence_margin(ref), m=len(COMPARISONS))
        print(f"      {label:16} delta = {pw['delta']:.5f}  "
              f"t_crit = {pw['t_critical']:.3f}  power = {pw['power']:.4f}  "
              f"(at alpha/{len(COMPARISONS)}, a planning lower bound)")
    print("\n  MARGIN SENSITIVITY (computed, not transcribed)")
    print(f"      {'frac':>6} {'delta':>9} {'power@alpha':>12} "
          f"{'power@alpha/3':>14}  registered")
    for row in margin_sensitivity():
        print(f"      {row['fraction']:>6.2f} {row['delta']:>9.5f} "
              f"{row['power_at_alpha']:>12.4f} "
              f"{row['power_at_conservative_holm_step']:>14.4f}"
              f"  {'<-- registered' if row['is_registered_margin'] else ''}")

    gaps = _placeholders(PLAN)
    if gaps:
        print("\n  REFUSED: plan still contains placeholders:")
        for g in gaps:
            print(f"      {g}")
        return 1

    for name, h in HYPOTHESES.items():
        if h["prediction"] not in SCIENTIFIC_DECISIONS:
            print(f"\n  REFUSED: {name}: prediction {h['prediction']!r} is not one "
                  f"of {SCIENTIFIC_DECISIONS}.")
            return 1
        for field in ("question", "rationale", "supported_if", "not_supported_if",
                      "inconclusive_if"):
            if not str(h.get(field) or "").strip():
                print(f"\n  REFUSED: {name}: {field!r} is empty. A hypothesis "
                      "without a stated falsification condition is not a "
                      "prediction.")
                return 1

    unver = unverified_fields(args.autond_root)
    ok_frozen, detail = is_frozen(args.repo_root)
    print(f"\n  source digests     "
          f"{'VERIFIED against ' + str(args.autond_root) if not unver else 'NOT VERIFIED'}")
    print(f"  frozen             {ok_frozen}  -  {detail}")

    permitted, blockers = confirmatory_run_permitted(args.autond_root, args.repo_root)
    print(f"  run permitted      {permitted}")
    if blockers:
        print("\n  BLOCKERS:")
        for b in blockers:
            print(f"      - {b}")
        print("\n" + freeze_instructions(args.repo_root))
        return 1

    print("\n  All gates open. The evaluation may be launched under this plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
