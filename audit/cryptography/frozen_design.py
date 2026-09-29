"""
FROZEN experimental design specification (adjudicated, preregistration-grade).

Every value here was fixed BEFORE any production observation and must not
be changed to accommodate a result. Each carries its adjudicated reason;
where the evidence did not support a value, the design records that the
capability is DISABLED rather than inventing a number.

Provenance note: `delta_materiality = 0.01` belongs to II-4, which asks a
different question (does architecture depth change accuracy). It is NOT
imported here. Historical CE3 code used alpha = 0.025 via "Bonferroni,
m=2"; that is provenance, not justification, and is superseded below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

DESIGN_SPECIFICATION_VERSION = "CE-frozen-design-2026-03"


# =====================================================================
# CE1 - Distinguishing-Signal Destruction
# =====================================================================

@dataclass(frozen=True)
class CE1Design:
    estimand: str = "E[Delta], Delta_k = acc_intact,k - acc_destroyed,k"
    statistical_unit: str = "dataset block"
    #: RE-FROZEN (methodological correction, design version ...-02).
    #: The previous wording "H0: E[Delta] = 0" is a WEAK null and does not
    #: justify the exactness of a sign-flip test: a zero mean does not imply
    #: that the sign of Delta_k is exchangeable. The procedure, alpha,
    #: estimand, K and CI are UNCHANGED; only the null's statement is
    #: corrected to the one the test actually provides exact inference for.
    primary_hypothesis: str = (
        "H0 (sharp): within every block, permuting the training labels has no effect on "
        "evaluation accuracy - the two arms' accuracies are exchangeable given the block. "
        "This implies E[Delta] = 0."
    )
    #: RANDOMIZATION (freeze amendment -03). Earlier versions asserted
    #: exchangeability from the sharp null alone. That was NOT sufficient:
    #: the two arms use DIFFERENT model seeds, so even under "labels have no
    #: effect" Delta_k is a seed-to-seed difference, and with the assignment
    #: fixed (intact always got seed base+2k, destroyed base+2k+1) there was
    #: no randomization distribution to enumerate - symmetry rested on an
    #: unstated modelling assumption that PRNG seeds behave as i.i.d. draws,
    #: and the arm was moreover confounded with seed parity.
    #:
    #: CE1 now RANDOMIZES, prospectively and auditably, which of the block's
    #: two model seeds is assigned to the intact arm. That coin flip is the
    #: randomization the sign-flip test enumerates.
    arm_assignment_randomized: bool = True
    assignment_seed: int = 20260101
    exactness_assumption: str = (
        "Per block k the design draws two model seeds {s_A, s_B} and randomizes, by a "
        "prospectively recorded fair coin flip, which one is assigned to the intact arm. "
        "Under H0 the label condition has no effect, so each seed's accuracy f(s) is the "
        "same whichever arm it is assigned to; the realized PAIR {f(s_A), f(s_B)} is "
        "therefore fixed and only the assignment varies. Swapping the assignment maps "
        "Delta_k -> -Delta_k, and each of the two assignments had probability 1/2 by "
        "construction, independently across blocks. The randomization distribution of "
        "(sign(Delta_1), ..., sign(Delta_K)) is thus exactly uniform on {-1,+1}^K, which is "
        "precisely what enumerating all 2^K sign assignments computes: the test is EXACT by "
        "DESIGN, not by assumption."
    )
    rejection_interpretation: str = (
        "Rejection licenses: 'the training-label permutation had SOME effect on evaluation "
        "accuracy in at least one block.' It does NOT by itself license the weak-null "
        "statement 'the mean difference is non-zero'; the mean Delta and its t-CI are "
        "reported as the estimand's point and interval estimate alongside."
    )
    primary_test: str = "exact_paired_sign_flip"
    alpha: float = 0.05
    ci_method: str = "t_based_95"
    effect_size: str = "mean_delta_raw_accuracy_and_cohens_dz"

    #: Equivalence-to-chance is DISABLED. A cryptanalytically meaningful
    #: margin (advantage <= 2^-10 => eps ~ 4.9e-4) lies at or below the
    #: 95% CI half-width of the sealed 10^6 test set (9.8e-4), so it is
    #: not estimable at this test-set size. eps = 0.01 was rejected: at
    #: accuracy 0.51 a distinguisher needs only ~27,050 samples
    #: (one-sided, alpha=.05, power=.95), so +/-0.01 would certify a
    #: working distinguisher as "chance".
    equivalence_enabled: bool = False
    equivalence_margin: float | None = None

    #: min_valid_blocks >= 6 is MANDATORY: the exact paired sign-flip test
    #: has minimum attainable two-sided p = 2/2^K, which is 0.0625 at K=5
    #: and 0.03125 at K=6. Below 6 valid blocks the primary test cannot
    #: reach alpha at any effect size.
    n_blocks: int = 8
    min_valid_blocks: int = 6
    #: n_blocks = min_valid_blocks + failure_tolerance. Operational, declared
    #: in advance: each block is two depth-10 x 200-epoch trainings spanning
    #: multiple GPU sessions.
    failure_tolerance: int = 2

    checkpoint_rule: str = "FINAL_EPOCH"
    terminal_epoch: int = 200
    #: Frozen sample counts. Part of the scientific protocol, not operator knobs.
    train_samples: int = 10 ** 7
    validation_samples: int = 10 ** 6
    evaluation_samples: int = 10 ** 6
    #: Frozen seed base; the full assignment is derived from it by a fixed rule.
    seed_base: int = 1000
    test_set_policy: str = "single_sealed_shared_across_blocks"
    inference_conditioning: str = (
        "conditional on the fixed sealed test set; its measurement error is COMMON "
        "across blocks and is NOT cancelled")
    multiplicity: str = "none (single primary hypothesis)"

    def to_dict(self) -> dict:
        return asdict(self)


CE1 = CE1Design()

CE1_REPORTING_RULE = (
    "CE1 reports mean Delta with its 95% t-CI, every individual Delta_k, and the "
    "exact sign-flip p-value. For the destroyed arm it reports mean accuracy with a "
    "95% CI and the distinguishing advantage that interval EXCLUDES. It issues NO "
    "equivalence-to-chance verdict: a non-significant difference is not evidence of "
    "equivalence, and no defensible margin exists at this test-set size."
)


# =====================================================================
# CE3 - Representation Decodability
# =====================================================================

@dataclass(frozen=True)
class CE3Design:
    n_replicates: int = 20
    n_splits_per_replicate: int = 5
    statistical_unit: str = "independent evaluation replicate"
    primary_hypothesis: str = "H0: mean replicate-level selectivity <= 0"

    #: Calibration is a METHODOLOGICAL GATE (positive control), not a
    #: reportable cryptographic claim.
    calibration_role: str = "methodological_gate"

    #: FIXED-SEQUENCE (hierarchical) testing: pre-ordered hypotheses are
    #: each tested at FULL alpha with strong FWER control and no
    #: adjustment, stopping at the first non-rejection (Maurer/Hothorn/
    #: Lehmacher 1995; Westfall & Krishen 2001; Edwards & Madsen 2007).
    #: The historical Bonferroni alpha=0.025 is unnecessarily conservative
    #: under BOTH the gate and the confirmatory reading.
    multiplicity: str = "fixed_sequence_calibration_then_primary"
    alpha: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)


CE3 = CE3Design()

CE3_DECISION_RULE = (
    "Fixed sequence at alpha=0.05. Step 1: the calibration positive control must "
    "validate the probing pipeline; if it does not, CE3 is INCONCLUSIVE regardless of "
    "the primary result and the sequence stops. Step 2: SUPPORTED iff mean "
    "replicate-level selectivity > 0 AND p < 0.05; otherwise NOT_SUPPORTED (mean not "
    "positive) or INCONCLUSIVE. No additional practical-magnitude threshold is applied."
)

#: Distinct estimands answering distinct questions; a conjunctive claim across
#: CEs is an intersection-union test, controlled at alpha without correction.
CROSS_EXPERIMENT_MULTIPLICITY = "none (intersection-union for any conjunctive claim)"
