"""
FROZEN DESIGN + STATISTICAL ANALYSIS PLAN for CE2, CE3 and CE4.

ISOLATION. This package (audit/cryptography/ce234/) is additive. It imports
from the existing audit.cryptography modules read-only and modifies none of
them, so an in-flight CE1 production process that imports those modules is
unaffected. Nothing here reads or writes any CE1 run directory.

PROVENANCE OF EVERY VALUE BELOW is one of:

  SOURCE-GOHR     taken from Gohr, "Improving Attacks on Round-Reduced
                  Speck32/64 Using Deep Learning", CRYPTO 2019, and the
                  accompanying deep_speck code (train_nets.py / speck.py).
  SOURCE-FROZEN   taken from this project's own frozen design
                  (audit/cryptography/frozen_design.py, CE-frozen-design-2026-03).
  SOURCE-METHOD   taken from the supplied methodology chapter (section cited).
  RECOMMENDATION  ADDITIONAL SCIENTIFIC RECOMMENDATION - introduced here
                  because the sources do not specify it. Each carries a
                  rationale and a citation. These are NOT "Gohr protocol"
                  and are NOT methodology requirements.
  UNSPECIFIED     genuinely unspecified in the sources; production fails
                  closed until the operator signs the value off.

Nothing in this file may be changed after a production run has been
started under it; `plan_hash()` binds every certificate to its contents.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PLAN_VERSION = "CE234-frozen-plan-2026-10-01-r2"

# =====================================================================
# Shared reference configuration (SOURCE-GOHR / SOURCE-FROZEN)
# =====================================================================

ROUNDS = 5                                  # SOURCE-GOHR (5-round problem)
DIFFERENTIAL = (0x0040, 0x0000)             # SOURCE-GOHR
DEPTH = 10                                  # SOURCE-GOHR (depth-10 residual net)
L2_REG = 1e-5                               # SOURCE-GOHR
PREDICT_BATCH = 5000                        # SOURCE-GOHR (batch size 5000)

#: The ONE model CE2-CE4 may use. Never a CE1 model, historical or in-flight.
REFERENCE_CHECKPOINT_SHA256 = (
    "256eb4a5ba93414f0a46ffd498d6121a03705698ac2a81764c2375207c5d6737")

#: Properties that MUST be verified from the file itself before production.
#: `rounds` and `differential` are NOT recorded inside a Keras file, so they
#: cannot be read back; they are verified BEHAVIOURALLY instead (see
#: production.verify_reference_model and BEHAVIOURAL_BINDING below).
REQUIRED_MODEL_PROPERTIES = {
    "conv1d_layers": 1 + 2 * DEPTH,         # 21
    "residual_merges": DEPTH,               # 10
    "l2_values": [L2_REG],
    "input_shape": [None, 64],
    "output_shape": [None, 1],
}

#: BEHAVIOURAL ROUND BINDING (RECOMMENDATION).
#: Rationale: a filename cannot establish the round count, and the historical
#: artifact "best5depth10 (10).h5" is empirically a depth-5 network whose
#: accuracy peaks at 7 rounds, i.e. filenames in this project have already
#: been wrong once. Gohr reports ~0.929 validation accuracy for the 5-round
#: depth-10 distinguisher; a model trained for a different round count scores
#: ~0.5 at 5 rounds. The gate below is therefore wide enough to be a sanity
#: check rather than a tuned number, and is evaluated on freshly generated
#: data before any CE2-CE4 measurement is taken.
BEHAVIOURAL_BINDING = {
    "n_samples": 100_000,
    "min_accuracy_at_declared_rounds": 0.85,   # RECOMMENDATION
    "max_accuracy_at_other_rounds": 0.60,      # RECOMMENDATION
    "other_rounds": [6, 7],
    "reference_accuracy_reported_by_gohr": 0.9291,   # SOURCE-GOHR
}

ALPHA = 0.05                                # SOURCE-FROZEN (CE3Design.alpha)

#: Cross-CE multiplicity. SOURCE-FROZEN: frozen_design.CROSS_EXPERIMENT_
#: MULTIPLICITY - distinct estimands answering distinct questions; a
#: conjunctive claim is an intersection-union test, controlled at alpha
#: without correction. Multiplicity is therefore handled WITHIN each CE.
CROSS_CE_MULTIPLICITY = "none (intersection-union for any conjunctive claim)"

#: Within-CE multiplicity: Holm (1979), "A simple sequentially rejective
#: multiple test procedure", Scand. J. Statist. 6:65-70. Chosen because it
#: controls FWER under arbitrary dependence and is uniformly more powerful
#: than Bonferroni. RECOMMENDATION (the sources fix a procedure only for CE3).
WITHIN_CE_MULTIPLICITY = "holm"


# =====================================================================
# CE2 - association with the analytical single-trail quantity
# =====================================================================

@dataclass(frozen=True)
class CE2Plan:
    experiment_id: str = "CE2-THEORY-CONSISTENCY"

    question: str = (
        "Does the frozen depth-10 distinguisher's output vary monotonically with an "
        "independently computed analytical single-trail probability, on real "
        "difference-bearing ciphertext pairs of the same 5-round Speck32/64 problem?")

    #: SOURCE-FROZEN (experiments/ce2/design.ANALYTICAL_TARGET), traced to code
    #: in gohr/speck.estimate_trail_probabilities and verified numerically
    #: against brute-force sampling of 16-bit modular-addition differentials.
    #: CONSTRUCT CLASSIFICATION (mathematically warranted label):
    #: it is a SINGLE-TRAIL quantity and a LOWER BOUND on the true differential
    #: propagation probability. It is NOT the exact differential probability,
    #: NOT the full differential effect, NOT a Markov-model probability over
    #: all trails, and NOT "the differential probability". The Markov/
    #: independence assumption (Lai-Massey-Murphy 1991) is used only to chain
    #: per-round factors along ONE realized trail.
    construct_class: str = "single-trail quantity; lower bound; proxy"
    target_quantity: str = (
        "product over rounds of xdp+(alpha_r, beta_r -> gamma_r), the Lipmaa-Moriai "
        "(FSE 2001) closed-form XOR-differential probability of modular addition, "
        "evaluated on each sample's REALIZED round-by-round differences under the "
        "Markov/independence assumption. This is a SINGLE-TRAIL quantity and a LOWER "
        "BOUND on the true differential propagation probability (trail clustering: "
        "Benamira et al., EUROCRYPT 2021). It is NOT the exact differential "
        "probability of the cipher and NOT 'what the network learned'.")

    #: H0/H1. Two-sided: the sources do not justify a predicted direction, and
    #: the historical negative correlation is itself under investigation.
    h0: str = "H0: the run-level population median Spearman rho is 0."
    h1: str = "H1: it is not 0."
    direction: str = "two-sided (no direction is justified by the sources)"

    #: EXPERIMENTAL UNIT (RECOMMENDATION, and the central correction to the
    #: historical CE2). The model is FIXED, so individual ciphertext samples
    #: are not independent replicates of anything model-level: they only
    #: quantify sampling uncertainty of rho for THIS model. The independent
    #: unit is therefore one evaluation RUN = one freshly generated dataset
    #: with its own evaluation seed. Cf. the pseudo-replication literature
    #: (Hurlbert 1984, Ecol. Monogr. 54:187-211).
    statistical_unit: str = "independent evaluation replicate (fresh dataset, fresh seed)"
    sampling_unit: str = (
        "ciphertext pair. The 10^6 pairs inside a replicate are SAMPLING units used "
        "to estimate that replicate's rho; they are NEVER represented as 10^6 "
        "independent scientific replicates and never enter the primary inference.")

    #: n per replicate: SOURCE-GOHR - Gohr evaluates on 10^6 test examples.
    samples_per_run: int = 10 ** 6
    #: OPERATOR-FROZEN (2026-10-01): 20 independent evaluation replicates.
    #: An exact two-sided sign test over replicate-level rho signs attains
    #: p = 2^-(n-1) at best; n = 20 gives a floor of 1.9e-6, which survives Holm
    #: across the CE2 endpoint family with room to spare.
    n_runs: int = 20

    #: PRIMARY ENDPOINT and test. Distribution-free, and does not inherit the
    #: n = 10^6 sample size: the sign test uses only the 10 run-level signs.
    primary_endpoint: str = "replicate-level Spearman rho between target and model output"
    reporting_requirements: tuple = (
        "all 20 replicate-level rho values", "per-replicate n",
        "median rho", "95% percentile-bootstrap CI over replicates",
        "primary replicate-level directional inference (exact sign test)",
        "secondary sensitivity analyses (last-round / prefix decomposition)")
    historical_status: str = (
        "the historical CE2 rho ~ -0.18 is evidence from a MISMATCHED historical "
        "configuration (depth-10 checkpoint evaluated on 7-round data) and is NOT a "
        "result for the current five-round question. The 5-round vs 7-round "
        "decomposition is a PRE-PRODUCTION DIAGNOSTIC, already seen, therefore "
        "exploratory and non-evidentiary; it may never be promoted to a confirmatory "
        "result.")
    primary_test: str = "exact two-sided sign test over run-level rho signs"
    effect_size: str = "median run-level rho; 95% percentile-bootstrap CI over runs"
    #: Spearman inference within a run is reported for completeness only and is
    #: NEVER the primary claim (Spearman 1904; exact small-sample theory is not
    #: applicable at n = 10^6 where any epsilon is 'significant').
    within_run_inference_role: str = "descriptive only, never primary"

    #: PREREGISTERED CONTROL FAMILY. Three endpoints -> Holm across them.
    controls: tuple = (
        ("C2-PERMUTED-TARGET",
         "the same model outputs paired against an independently permuted copy of the "
         "target vector. Breaks the per-sample pairing while preserving both marginal "
         "distributions exactly. Expected |rho| ~ 0 if the association is real.",
         "RECOMMENDATION"),
        ("C2-UNTRAINED-MODEL",
         "an untrained, randomly initialised network of the IDENTICAL architecture, "
         "same inputs. Separates association attributable to training from "
         "association attributable to architecture/input encoding alone.",
         "RECOMMENDATION"),
    )

    #: PREREGISTERED SECONDARY ENDPOINT (not a control, and never primary).
    #: The final round's xdp+ factor is a deterministic function of the
    #: ciphertext pair the model is given, so part of the target is visible in
    #: the input by construction. This decomposition is declared in advance so
    #: that a positive primary result cannot later be explained away, or
    #: inflated, by that visible component.
    secondary_endpoints: tuple = (
        ("S2-LAST-ROUND-ONLY", "rho(model output, final-round xdp+ factor alone)"),
        ("S2-PREFIX-ONLY", "rho(model output, product of rounds 1..n-1), the part NOT "
                           "determined by the ciphertext pair"),
    )

    exclusion_rule: str = (
        "a run is excluded only if the target or output vector is constant (rho "
        "undefined) or non-finite. Exclusions are reported, never silently dropped.")
    failure_rule: str = (
        "a run that raises is recorded as FAILED and retained; it is not replaced.")
    minimum_valid_runs: int = 20
    stopping_rule: str = (
        "exactly n_runs runs are executed. No interim analysis, no early stop, no "
        "extension on an inconvenient p-value.")
    decision_rule: str = (
        "SUPPORTED iff the Holm-adjusted primary p < alpha AND both controls behave as "
        "preregistered (|median control rho| < |median primary rho| / 2). "
        "NOT_SUPPORTED iff the primary is significant with the opposite sign to the "
        "controls' separation requirement or the controls reproduce the effect. "
        "INCONCLUSIVE if fewer than minimum_valid_runs valid runs, or the primary is "
        "non-significant - which is NOT evidence that rho = 0.")
    limitations: tuple = (
        "association only; no causal interpretation is licensed",
        "one model instance; the unit of replication is the evaluation run, so the "
        "claim does not generalise over independently trained models",
        "the target is a single-trail LOWER BOUND, not the exact differential "
        "probability",
        "evaluated on real difference-bearing pairs only, not on the mixed real/random "
        "population the distinguisher is trained to separate",
    )


# =====================================================================
# CE3 - representation decodability
# =====================================================================

@dataclass(frozen=True)
class CE3Plan:
    experiment_id: str = "CE3-REPRESENTATION-INTERPRETATION"

    question: str = (
        "Is the analytical single-trail quantity decodable from the frozen model's "
        "penultimate representation beyond matched controls?")

    #: RELATIONSHIP TO CE2 (traced in code, not assumed): the CE3 target is the
    #: CE2 quantity discretised into 5 bins at its own empirical quintiles
    #: (adapters/gohr.generate_primary_representation_task). Same construct,
    #: different measurement scale. Documented so the two CEs are not treated
    #: as independent confirmations of each other.
    target_quantity: str = (
        "CE2's analytical single-trail probability, discretised into 5 classes at the "
        "empirical quintiles of that run's own target vector (MULTICLASS). Identical "
        "construct to CE2, coarsened; NOT an independent quantity.")

    #: Layer fixed PROSPECTIVELY, before any result is seen.
    representation: str = (
        "model.layers[-2] output: the 64-unit post-activation penultimate layer "
        "immediately before the output classifier, inference mode, no pooling, no "
        "normalisation, sample order preserved.")

    #: SOURCE-FROZEN: frozen_design.CE3 (n_replicates=20, n_splits=5,
    #: statistical unit = independent evaluation replicate, fixed-sequence
    #: calibration-then-primary at full alpha).
    statistical_unit: str = "independent evaluation replicate"
    n_replicates: int = 20
    n_splits_per_replicate: int = 5
    samples_per_replicate: int = 100_000      # SOURCE: historical CE3 practice
    multiplicity: str = "fixed_sequence_calibration_then_primary"  # SOURCE-FROZEN

    #: PROBE: fixed hyperparameters, NO tuning anywhere, therefore no
    #: opportunity for test-fold leakage into probe selection. If tuning were
    #: ever introduced it would require nested CV (Varma & Simon, BMC
    #: Bioinformatics 2006; Cawley & Talbot, JMLR 2010).
    probe: str = ("multinomial logistic regression, library defaults fixed in "
                  "probe/evaluation.LogisticRegressionProbe, seeded per fold, "
                  "identical seed and identical fold partition for every arm. "
                  "No hyperparameter search is performed at any stage.")
    metric: str = "balanced accuracy (class-imbalance robust)"

    #: CONTROL AUDIT. The historical control (signal_destroyed.h5) is
    #: DEGENERATE: it emits a constant output and an all-zero 64-dimensional
    #: representation, so any probe on it scores exactly chance. Historical CE3
    #: therefore measured "real minus chance", not "real minus control".
    historical_control_defect: str = (
        "signal_destroyed.h5 has an all-zero penultimate representation (verified); "
        "its probe score is exactly 1/n_classes. The same collapse is present in the "
        "CE1 destroyed arm, so a trained-destroyed control cannot be assumed non-"
        "degenerate and must be checked before use.")
    #: OPERATOR-FROZEN (2026-10-01): the untrained twin is the PRIMARY control;
    #: the raw-input control is SECONDARY and INFORMATIONAL ONLY - it is not a
    #: pass/fail criterion, because no source specifies one. Its purpose is to
    #: distinguish target recoverability from target-specific, task-induced
    #: representation. The same evaluation examples are used for the trained,
    #: twin and raw-input representations within each replicate.
    primary_control: str = "C3-UNTRAINED-ARCH"
    secondary_control_role: str = ("C3-RAW-INPUT is informational; it never changes the "
                                   "decision, it caps the defensible wording")
    shared_examples_within_replicate: bool = True
    #: Required audit: if the untrained twin itself decodes the target above
    #: chance, that must be reported explicitly, because selectivity is then a
    #: difference between two non-trivial decodabilities.
    twin_decodability_audit: str = (
        "report the twin's absolute probe score against the 1/n_classes chance level "
        "for every replicate; a non-trivial twin score is reported, never hidden")
    controls: tuple = (
        ("C3-UNTRAINED-ARCH",
         "untrained randomly-initialised network of the identical architecture. "
         "Controls for architecture, dimensionality and input encoding; does NOT "
         "control for training. Standard random-network baseline in the probing "
         "literature (Zhang & Bowman, BlackboxNLP 2018; Hewitt & Liang, EMNLP 2019).",
         "RECOMMENDATION"),
        ("C3-RAW-INPUT",
         "the same probe applied to the raw 64-bit ciphertext-pair encoding. If the "
         "target is equally decodable from the raw input, the REPRESENTATION adds "
         "nothing and no statement about the representation is licensed.",
         "RECOMMENDATION"),
        ("C3-CALIBRATION-GATE",
         "differential-class probing as a methodological positive control for the "
         "probing pipeline; a gate, never cryptographic evidence.",
         "SOURCE-FROZEN"),
    )
    #: A degenerate control is a reason to REFUSE, never a result.
    control_admissibility: str = (
        "every control representation must be non-degenerate: at least 2 features with "
        "non-zero variance, else the run FAILS CLOSED.")

    primary_endpoint: str = "replicate-level selectivity = real score - C3-UNTRAINED-ARCH score"
    primary_test: str = ("one-sided Wilcoxon signed-rank over the 20 replicate-level "
                         "selectivities (SOURCE-FROZEN), nested CV folds are a "
                         "variance-reduction device and are NEVER treated as "
                         "independent observations")
    effect_size: str = "mean selectivity, Cohen's dz, 95% t-CI and percentile-bootstrap CI"
    exclusion_rule: str = "a replicate is excluded only on probe failure or non-finite score"
    minimum_valid_replicates: int = 20
    decision_rule: str = (
        "fixed sequence at alpha. Step 1: the calibration gate must validate the "
        "probing pipeline, else INCONCLUSIVE and stop. Step 2: SUPPORTED iff mean "
        "replicate-level selectivity against the PRIMARY control > 0 and p < alpha; "
        "NOT_SUPPORTED if the mean is not positive; otherwise INCONCLUSIVE. "
        "C3-RAW-INPUT and the twin-decodability audit are reported alongside and cap "
        "the permissible wording; neither can change the decision.")
    limitations: tuple = (
        "decodability is NOT use: a probe can recover information the model does not "
        "rely on (Hewitt & Liang 2019; Belinkov, Computational Linguistics 2022)",
        "the target is CE2's construct coarsened, so CE3 is not independent "
        "confirmation of CE2",
        "one model instance; replicates vary the evaluation data, not the training",
    )


# =====================================================================
# CE4 - intervention sensitivity
# =====================================================================

@dataclass(frozen=True)
class CE4Plan:
    experiment_id: str = "CE4-INTERVENTION-SENSITIVITY"

    question: str = (
        "Is the frozen model's output more sensitive to perturbation of the "
        "difference-bearing ciphertext structure than to a magnitude-matched, "
        "XOR-preserving perturbation?")

    estimand: str = (
        "mean within-replicate paired difference in output-change magnitude between "
        "the structural policy and the matched control policy at the PRIMARY "
        "intervention magnitude, within the prespecified eligible population, for this "
        "frozen model instance; and the reproducibility of that quantity across "
        "independent evaluation replicates.")

    # -------------------- intervention magnitude --------------------
    #: DERIVATION (model-free, pre-production, outcome-blind).
    #: The declared target structure is the ciphertext-pair XOR, which occupies
    #: exactly 32 mirrored bit-position pairs of the 64-bit encoding
    #: (columns j / 32+j for the left word, 16+j / 48+j for the right word, from
    #: speck.convert_to_binary([c0l, c0r, c1l, c1r])).
    #:   * a single-sided flip at position p toggles the pair XOR at p;
    #:   * a mirrored flip at p leaves it invariant, by (a^1)^(b^1) = a^b.
    #: An arm that changes the XOR at m positions therefore costs m bits, while
    #: an XOR-preserving arm costs 2 bits per position. Equal INPUT HAMMING
    #: displacement 2k is attainable (structural: 2k single-sided flips;
    #: control: k mirrored flips), and equal POSITION-SET SIZE is then
    #: impossible - 2k vs k. The two matching criteria are mutually exclusive;
    #: this design fixes equal Hamming displacement and declares the position-set
    #: difference as an unavoidable consequence, not an oversight.
    #: Magnitude itself is NOT fixed by any source. Measured on 500,000 real
    #: 5-round pairs (model-free): the pair-XOR Hamming weight has mean 15.37,
    #: median 15, min 4. Eligibility coverage by magnitude: 2k=2 100.00%,
    #: 2k=4 100.00%, 2k=8 99.712%, 2k=12 90.39%, 2k=16 48.05%.
    #: PRESPECIFIED SELECTION RULE (declared before any effect was computed):
    #: admit only magnitudes with >= 95% population coverage, and take the
    #: SMALLEST constructible magnitude as primary, because it minimises
    #: off-manifold displacement - the principal alternative explanation for a
    #: structural/control gap - while covering the whole population.
    magnitude_justification: str = "ADDITIONAL SCIENTIFIC RECOMMENDATION (option C)"
    primary_magnitude_bits: int = 2          # 2k = 2, i.e. k = 1 mirrored pair
    magnitude_ladder_bits: tuple = (2, 4, 8)   # all >= 95% coverage
    magnitude_coverage_rule: str = "admit 2k only if eligibility coverage >= 95%"
    n_intervention_bits_historical: int = 8
    historical_magnitude_status: str = (
        "the historical 8 was an inherited adapter default with no source "
        "justification; it is retained only as a ladder point, never as the primary.")

    #: Secondary dose-response family: the two non-primary ladder points, Holm
    #: corrected. A monotone increase of the gap with magnitude is evidence of
    #: structural sensitivity; a flat or non-monotone profile is reported as such.
    secondary_family: str = "dose-response across the non-primary ladder magnitudes"
    multiplicity: str = "Holm within the CE4 secondary family; the primary is a single test"

    # -------------------- construction --------------------
    #: SIDE-BALANCED construction (design correction found by the control-validity
    #: audit). Draw 2k eligible positions S per sample, in k same-word pairs so
    #: the two halves have identical per-word composition.
    #:   control    : flip BOTH sides at S[:k]   -> 2k bits, k in c0 and k in c1,
    #:                pair XOR exactly invariant
    #:   structural : flip c0 at S[:k] and c1 at S[k:] -> 2k bits, k in c0 and k
    #:                in c1, pair XOR toggled at 2k positions
    #: The c0-side perturbation is IDENTICAL between arms; the arms differ only
    #: in which positions of c1 are flipped. The earlier one-sided structural
    #: construction put all 2k flips in c0, which did NOT match the control on
    #: per-ciphertext or per-word perturbation.
    construction: str = "side-balanced, same-word-paired position draw"
    matching_criterion: str = (
        "matched: total input Hamming displacement (2k), per-ciphertext displacement "
        "(k in c0 and k in c1), per-word composition, eligible-position pool, and the "
        "identical c0-side flips. NOT matched (mathematically impossible under "
        "XOR preservation): the number of distinct positions touched, 2k vs k. "
        "NOT matched and reported as a limitation: displacement in the model's "
        "representation space and distance to the cipher's output manifold.")
    eligibility: str = (
        "sample admits k disjoint same-word pairs of difference-bearing positions "
        "(hence 2k eligible positions). BOTH arms draw from this same pool, so "
        "eligibility cannot induce a between-arm selection difference. The "
        "restriction defines the population the claim is about and is reported.")

    manipulation_checks: tuple = (
        "both arms change exactly 2k input bits per sample",
        "both arms change exactly k bits of c0 and k bits of c1",
        "both arms have identical per-word perturbation counts",
        "the c0-side flips are identical between arms",
        "the control leaves the pair XOR exactly invariant",
        "the structural arm changes the pair XOR at exactly 2k positions",
        "no ineligible position is touched by either arm",
        "marginal per-column bit frequencies are reported for both arms",
    )
    controls: tuple = (
        ("C4-XOR-PRESERVING", "mirrored flips at difference-bearing positions",
         "SOURCE-FROZEN (construction corrected here)"),
        ("C4-NULL", "zero bits flipped; the measured gap must be exactly 0",
         "RECOMMENDATION"),
    )

    # -------------------- units and inference --------------------
    #: TWO DISTINCT ANALYSES, each with its own inferential target.
    analysis_1: str = ("WITHIN-REPLICATE PAIRED EFFECT: per-sample paired gap "
                       "|f(x)-f(structural)| - |f(x)-f(control)|, Wilcoxon signed-rank "
                       "over eligible samples. Inferential target: the paired effect "
                       "WITHIN that replicate's sample. Does NOT generalise across "
                       "replicates.")
    analysis_2: str = ("ACROSS-REPLICATE REPRODUCIBILITY: the 10 replicate-level mean "
                       "gaps, exact sign test + bootstrap CI. Inferential target: "
                       "stability of the effect over independent evaluation data. This "
                       "is the PRIMARY inference.")
    statistical_unit: str = "independent evaluation replicate (primary); eligible sample (within-replicate)"
    n_runs: int = 10                           # OPERATOR-FROZEN 2026-10-01
    samples_per_run: int = 100_000             # OPERATOR-FROZEN 2026-10-01
    retain_per_sample_gaps: bool = True
    effect_size: str = ("per-replicate mean gap and Cohen's dz; across replicates the "
                        "mean of means with a 95% percentile-bootstrap CI")
    threshold_policy: str = (
        "NO magnitude threshold is applied. The historical 0.10 x 'ceiling' rule is "
        "superseded: the historical ceiling (0.00719) was SMALLER than the observed "
        "gap (0.0723), so it bounded nothing.")
    exclusion_rule: str = "ineligible samples are excluded by the predeclared rule and counted"
    minimum_valid_runs: int = 10
    decision_rule: str = (
        "SUPPORTED iff all manipulation checks pass in every replicate, C4-NULL is "
        "exactly 0, and at the PRIMARY magnitude the across-replicate exact sign test "
        "is significant with a positive mean of replicate means. NOT_SUPPORTED if that "
        "mean is not positive. INCONCLUSIVE otherwise or on any failed check. The "
        "dose-response family is reported but cannot change the primary decision.")
    limitations: tuple = (
        "intervention sensitivity is NOT causal necessity of the analytical quantity: "
        "the intervention never manipulates that quantity",
        "both arms move the input off the cipher's output manifold; the contrast is "
        "between two off-manifold policies",
        "the arms cannot be matched on the number of distinct positions touched; this "
        "is a mathematical consequence of XOR preservation",
        "one model instance; no claim about models in general",
        "the control is inert with respect to the DECLARED XOR target only",
    )


CE2 = CE2Plan()
CE3 = CE3Plan()
CE4 = CE4Plan()

#: Values the sources do not fix and that an operator must sign off before
#: production. Production FAILS CLOSED while any of these is unsigned.
UNSPECIFIED_IN_SOURCE = {
    "CE2.n_replicates": "no source fixes it; OPERATOR-FROZEN at 20 on 2026-10-01",
    "CE3.control_model": "no source fixes a non-degenerate CE3 control; the untrained "
                         "twin is OPERATOR-FROZEN as primary on 2026-10-01",
    "CE4.n_replicates": "no source fixes it; OPERATOR-FROZEN at 10 on 2026-10-01",
    "CE4.intervention_magnitude": "no source fixes it; derived model-free from the "
                                  "encoding and the coverage rule, primary 2k=2, "
                                  "ladder (2,4,8); ADDITIONAL SCIENTIFIC RECOMMENDATION",
}


def plan_dict() -> dict:
    return {
        "plan_version": PLAN_VERSION,
        "reference": {"rounds": ROUNDS, "differential": list(DIFFERENTIAL),
                      "depth": DEPTH, "l2_reg": L2_REG,
                      "checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
                      "required_model_properties": REQUIRED_MODEL_PROPERTIES,
                      "behavioural_binding": BEHAVIOURAL_BINDING},
        "alpha": ALPHA,
        "cross_ce_multiplicity": CROSS_CE_MULTIPLICITY,
        "within_ce_multiplicity": WITHIN_CE_MULTIPLICITY,
        "ce2": asdict(CE2), "ce3": asdict(CE3), "ce4": asdict(CE4),
        "unspecified_in_source": UNSPECIFIED_IN_SOURCE,
    }


def plan_hash() -> str:
    return hashlib.sha256(
        json.dumps(plan_dict(), sort_keys=True, default=str).encode()).hexdigest()


def holm(pvalues: dict) -> dict:
    """Holm (1979) step-down adjusted p-values; monotone, FWER-controlling."""
    items = sorted(((k, v) for k, v in pvalues.items() if v is not None),
                   key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    for k, v in pvalues.items():
        if v is None:
            out[k] = None
    return out


def exact_sign_test(values, *, alternative: str = "two-sided") -> dict:
    """
    Exact binomial sign test over the SIGNS of run-level statistics. Zeros are
    discarded (Dixon & Mood 1946). Distribution-free and independent of the
    per-run sample size, which is the point: it cannot be driven by n = 10^6.
    """
    from scipy import stats

    v = [float(x) for x in values]
    pos = sum(1 for x in v if x > 0)
    neg = sum(1 for x in v if x < 0)
    n = pos + neg
    if n == 0:
        return {"n_effective": 0, "n_positive": 0, "p_value": None,
                "note": "all run-level statistics are exactly zero"}
    p = float(stats.binomtest(pos, n, 0.5, alternative=alternative).pvalue)
    return {"test": "exact binomial sign test", "alternative": alternative,
            "n_effective": n, "n_positive": pos, "n_negative": neg,
            "p_value": p, "smallest_attainable_p": float(2.0 ** -(n - 1))}


def bootstrap_ci(values, *, n_boot: int = 10000, seed: int = 0, alpha: float = ALPHA):
    import numpy as np

    v = np.asarray(list(values), dtype=float)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return [float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2)))]


def write_plan(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**plan_dict(), "plan_hash": plan_hash()},
                               indent=2, sort_keys=True, default=str))
    return path
