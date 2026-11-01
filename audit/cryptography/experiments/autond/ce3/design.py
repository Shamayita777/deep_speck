from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

EXPERIMENT_ID = "AUTOND-CE3-REPRESENTATION-DECODABILITY"
DESIGN_VERSION = "AutoND-CE3-design-2026-11-r1"
CERTIFICATE_SCHEMA = "autond-ce3-certificate-1"
PREREGISTRATION_SCHEMA = "autond-ce3-preregistration-1"
PREREGISTRATION_RELPATH = "experiments/autond/ce3/preregistration/preregistration.json"
CE3_RELPATH = "experiments/autond/ce3"
EVIDENCE_RELPATH = "experiments/autond/ce3/evidence"
VALIDATION_RELPATH = "experiments/autond/ce3/validation"

PROPOSED = "PROPOSED_PENDING_APPROVAL"
FROZEN = "FROZEN"
UNSET = "NOT_YET_RECORDED"

UPSTREAM_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"
PUBLISHED_CHECKPOINT_SHA256 = "a7be788dbbb760a7be9053748e6e0f37dedc5768db3a74a5c3ebd294dca4cc26"

CE2_RUN_RELPATH = "evidence/ce2/autond_ce2_production_20261031T155733Z"
CE2_TWIN_RELPATH = "controls/untrained_twin.weights.h5"
CE2_TWIN_FILE_SHA256 = "adda08a31be371229424d464273689784c61a4f16a4852f9f3e2223d424982b4"
CE2_TWIN_WEIGHTS_SHA256 = "237722786db3075a8501b670d51f0009edaa4e02e2b367460441978e8c0a239a"

DEPENDENCIES = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "experiments/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "experiments/autond/__init__.py": "6bce8850ca27829b0b89c85b31e0a4e711e37243f3fd452d5fa0af2ba0dd699b",
    "experiments/autond/ce2/__init__.py": "c27d83dd9a6804d47d9353b09a6096205d0a5c5cb3cf570b31365a73ee786d8b",
    "experiments/autond/ce2/adapter.py": "2726c66361e2c445d3226d079e9e3ce03bf99304b1b149ad203e7a713239d25e",
    "experiments/autond/ce2/design.py": "1cf3ca643e4acf1aaaed73870a2fb39502a8e92648e5d14424527bc422306631",
    "experiments/autond/ce2/theory.py": "1b13c89fd6387da6d507663d283e1f2b566793aeefa5c1f41698a2d50ad3c45b",
    "experiments/autond/ce2/verify.py": "5eedafcaa8545730fcb71df662d762f18a1492bfe83976bfbc063bbd0a9e8331",
    "experiments/autond/ce2/preregistration/preregistration.json":
        "b8629e155e1054636622b119d9349fb27aaa3ed220ff13949f704ea1ab035741",
    "targets/__init__.py": "26371bc876181812de568c1bc57b9d1c36b48a36e9fd56fbd5ba62aaa2f644b1",
    "targets/autond/__init__.py": "a60fa8668433880ad9d8174accb4e724849041f6f2565aa7306871a60194cb8d",
    "targets/autond/upstream.py": "99d703135e1c569f74da3e20d741aab6cf411d9c1e21abfedacee1a1b923dd95",
    "targets/autond/model_binding.py": "32e96468ef7d18e060e33182461d42250f00cda8b421c73234c4a6c6b216950a",
    "core/__init__.py": "e60cf7a039fdd4f75354acef18b2ad962a8df2380203bb6af0e87a8559e1f740",
    "core/evidence.py": "187cb6faeddb68b19a73232f31b0d295d9efd3fef417c5400915b1fb192ba08b",
    "upstream_sha256.json": "5c9fe81107091e81804c7fa606039018c8ad56b3dce6e85e02fc6336251ebf56",
}

CE2_SOURCE_RUN = {
    "relpath": CE2_RUN_RELPATH,
    "design_version": "AutoND-CE2-frozen-design-2026-10-r3",
    "design_sha256": "3f6340e8625749d1a1281ea44984ea4ee48521e00c6c774598398680b8a0635f",
    "preregistration_sha256": "b8629e155e1054636622b119d9349fb27aaa3ed220ff13949f704ea1ab035741",
    "certificate_sha256": "4eaa7ce86ae4681befbd032eefa9657ec82a68e33ac4b60f816dbd0d6ddb3f85",
    "run_manifest_sha256": "8976f00cbf04196ed1f28f4f4237e1a0c0f8c7e4ea50c6e0c36421b852df6d6d",
    "verification_report": "verification_20261031T170659Z.json",
    "verification_report_sha256": "e27123a5f65742c490e097da12bf5659077cb3b922bc500cfda7e835c8f15c59",
    "decision": "SUPPORTED",
    "twin_file": CE2_TWIN_RELPATH,
    "twin_file_sha256": CE2_TWIN_FILE_SHA256,
    "twin_weights_sha256": CE2_TWIN_WEIGHTS_SHA256,
    "twin_seed": 20261031,
    "use": ("read-only: the fixed untrained twin saved by this run is copied byte for byte into each CE3 output "
            "directory and hash-checked; no other CE2 evidence is used and CE2 is not amended"),
}

P1_RECORD_RELPATH = "experiments/autond/ce3/validation/p1/p1_record.json"
P1_RECORD_SHA256 = "69b430ecd0112e857ee8d71cc7a56b96f079a236181c155db594f1e1e80bad22"
POWER_EVIDENCE_RELPATH = "experiments/autond/ce3/validation/power/power.json"
POWER_EVIDENCE_SHA256 = "ecc61ce0aa22b18ce2eadab77bfeb3b95f0253169f10c75335373289a0d5226c"
BENCHMARK_RELPATH = "experiments/autond/ce3/validation/benchmark/benchmark.json"
BENCHMARK_SHA256 = "c7d063461e74795b513294d937440ba7790ea878ca7a22512ceb5bc7c14629d7"
REPRESENTATION_DIAGNOSTICS_RELPATH = "experiments/autond/ce3/validation/representation/representation_diagnostics.json"
REPRESENTATION_DIAGNOSTICS_SHA256 = "ffc19d390a061ea3f235f2f090a57b46ffb14ec266224392e4c5912fb55e4147"

N_CLASSES = 5
TARGET_EDGES = [-68, -63, -59, -55]
P1_PAIRS = 1_000_000
MIN_CLASS_FRACTION = 0.15
MIN_CLASS_COUNT_PER_FOLD = 1_000

N_REPLICATES = 20
N_MAIN_PAIRS = 200_000
N_CALIBRATION_PER_CLASS = 50_000
N_FOLDS = 5
N_PERMUTATIONS = 5
ALPHA = 0.05
KAPPA = 0.5
PUBLISHED_ACCURACY = 0.5144
PUBLISHED_ACCURACY_SOURCE = (
    "autond_ce1/AutoND-upstream/README.md lines 151-181 at the pinned upstream commit: eval_nets.py on the polished "
    "round-8 SPECK32/64 checkpoint, 5 evaluation sets of 10^6 samples from main.make_train_data (label Y ~ "
    "Bernoulli(1/2), Y = 0 inputs uniform), metric accuracy of 1[score > 0.5] (eval_nets.py evaluate_X_Y), "
    "reported as mean 0.5144 +- 0.0003 (standard deviation over the 5 sets); the per-set TPR 0.512-0.514 and TNR "
    "0.515-0.517 make the balanced accuracy equal to the accuracy to the reported precision")
MAX_ITER = 2_000
FALLBACK_MAX_ITER = 5_000
GRADIENT_TOLERANCE = 1e-3
HEAD_TOLERANCE = 1e-5

FOLD_SEED_BASE = 40_000_000
CALIBRATION_FOLD_SEED_BASE = 60_000_000
CALIBRATION_SHUFFLE_SEED_BASE = 30_000_000
PERMUTATION_SEED_BASE = 50_000_000
BENCHMARK_REPLICATE_INDEX = 999

ENDPOINTS = ("C_task", "C_twin", "S1", "S2")
FIXED_SEQUENCE = ("C", "S1", "S2")

PROBE = {
    "pipeline": "sklearn.pipeline.Pipeline([StandardScaler(), LogisticRegression(...)])",
    "scaler": {"class": "sklearn.preprocessing.StandardScaler", "with_mean": True, "with_std": True,
               "fit_on": "training fold only"},
    "logistic": {"class": "sklearn.linear_model.LogisticRegression", "penalty": "l2", "C": 1.0,
                 "solver": "lbfgs", "tol": 1e-4, "max_iter": MAX_ITER, "fit_intercept": True,
                 "class_weight": None, "random_state": 0},
    "feature_dtype": "float64",
    "blas_threads": 1,
    "scikit_learn": "1.9.1",
    "convergence": ("a fit converges when n_iter_ < max_iter, no ConvergenceWarning is raised and the "
                    "maximum absolute gradient of the training objective at the fitted parameters, computed "
                    "independently of scikit-learn, is at most gradient_tolerance"),
    "gradient_tolerance": GRADIENT_TOLERANCE,
    "objective": ("mean multinomial (or binary) log-loss over the training fold plus ||W||^2 / (2 C n_train); "
                  "the intercept is not penalised"),
    "max_iter_rule": ("max_iter = 2000 unless the blinded benchmark records a non-converged fit at 2000 and every "
                      "fit converges at 5000, in which case 5000; no other value and no retry"),
    "non_convergence": "the replicate is FAILED; nothing is retried or changed",
}

REPRESENTATION = {
    "layer_index": 32,
    "layer_class": "Activation",
    "activation": "relu",
    "output_shape": [None, 64],
    "inbound": {"index": 31, "class": "BatchNormalization"},
    "pre_activation_dense": {"index": 30, "class": "Dense", "units": 64, "activation": "linear"},
    "consumer": {"index": 33, "class": "Dense", "units": 1, "activation": "sigmoid", "is_output": True},
    "n_layers": 34,
    "published_layer_name": "activation_2",
    "identity_rule": ("identity is the graph position and structure checked above; Keras layer names depend on "
                      "how many models a process has built (the twin built second is named activation_5) and "
                      "are recorded but never used for identity"),
    "extraction": "keras.Model(model.input, model.layers[32].output).predict(X, batch_size=5000), float32",
    "head_consistency": ("sigmoid(h @ W_dense_3 + b_dense_3) must equal model.predict(X) to within "
                         "head_tolerance on every main and calibration example"),
    "head_tolerance": HEAD_TOLERANCE,
    "twin_admissibility": ("the twin representation must have at least 2 features with nonzero variance on every "
                           "main and calibration set; otherwise the replicate is FAILED"),
}

DESIGN: dict = {
    "experiment_id": EXPERIMENT_ID,
    "design_version": DESIGN_VERSION,
    "status": FROZEN,
    "research_question": (
        "Across independent evaluation replicates, is the AutoND CE2 trajectory-derived target, coarsened into "
        "frozen whole-value classes, linearly decodable from the 64-unit penultimate ReLU representation of the "
        "published AutoND SPECK32/64 round-8 DBitNet, beyond a within-replicate permuted-label control and beyond "
        "one fixed untrained DBitNet, once the probe pipeline has shown that it recovers the known "
        "genuine-versus-random task?"),
    "claim_scope": {
        "positive_claim": ("linear-probe decodability of the coarsened CE2 target from the specified fixed "
                           "representation under the specified probe"),
        "not_claimed": [
            "that the network uses the target (A3 shows decodability without use)",
            "a learned cryptanalytic strategy or an internal decision process",
            "the exact fixed-key characteristic probability or the full-cipher differential probability",
            "independence from CE2: the target is the CE2 construct",
            "information-theoretic content: a deterministic representation of the input cannot contain more "
            "information about the target than the input does",
            "anything about other checkpoints, layers, twins, probes, ciphers or round counts",
            "absence of decodability after a non-rejection",
        ],
        "accessibility_vs_information": (
            "every comparison is between balanced accuracies of one fixed linear probe; a higher score means the "
            "target is more linearly accessible from that representation under that probe, not that the "
            "representation holds more information. By the data-processing inequality I(T; h(X)) <= I(T; X) "
            "for every representation h of the 64-bit input X."),
        "ce1_ce2_relation": "uses the CE2 target construct and CE2's untrained twin; does not amend CE1 or CE2",
    },
    "target": {
        "source": ("T = sum over the 8 rounds of log2 xdp+ along the reconstructed genuine-pair trajectory, "
                   "computed by experiments/autond/ce2/adapter.py trajectory_and_weights (r3, pinned)"),
        "pilot": {
            "name": "P1",
            "pairs": P1_PAIRS,
            "model_free": True,
            "evidentiary": False,
            "record": P1_RECORD_RELPATH,
            "record_sha256": P1_RECORD_SHA256,
            "rule": ("among all whole-value partitions of the P1 support into K classes whose every class holds at "
                     "least min_class_fraction of P1, choose the one minimising sum_j |K * N(T <= e_j) - j * N|, "
                     "ties broken by the lexicographically smallest edge tuple; K = 5, and K = 3 only if no "
                     "five-class partition is feasible"),
            "exploratory_pilot_disclosed": ("an earlier unrecorded in-memory pilot of 10^6 pairs (P0) showed 57 "
                                            "integer values in [-91, -35]; it was used only to judge that five "
                                            "classes are feasible and is not evidence"),
        },
        "n_classes": N_CLASSES,
        "edges": TARGET_EDGES,
        "class_rule": "class(T) = number of frozen edges e with T > e; equal T values are never split",
        "class_order": "ascending T",
        "fallback_resolution": (
            "resolved before production by P1: the class count and edges are frozen in this design and applied "
            "unchanged to every replicate; no replicate is rebinned and no three-class branch exists in the "
            "confirmatory protocol; a replicate violating the class-coverage minimum is FAILED"),
        "min_class_fraction": MIN_CLASS_FRACTION,
        "min_class_count_per_fold": MIN_CLASS_COUNT_PER_FOLD,
        "coverage_checked_for": ("the actual target labels and every permuted label vector on the main folds, and "
                                 "the calibration labels on the calibration folds, in every training and test "
                                 "partition"),
    },
    "sampling": {
        "n_replicates": N_REPLICATES,
        "main_pairs_per_replicate": N_MAIN_PAIRS,
        "main_pairs_status": FROZEN,
        "main_pairs_rule": ("the smallest n in {50000, 100000, 200000} whose uncertainty-propagated lower bound "
                            "for P(SUPPORTED) (5th percentile over the bootstrap distribution of the synthetic "
                            "pipeline parameters, evaluated with B = 21) is at least 0.90 at the reference scenario "
                            "of the power evidence; if none qualifies the design is not frozen"),
        "pair_construction": ("fresh genuine pairs from os.urandom with the construction of upstream "
                              "main.make_train_data (Y = 1): experiments/autond/ce2/adapter.py "
                              "generate_genuine_pairs (pinned), gates E and F required"),
        "calibration_per_class": N_CALIBRATION_PER_CLASS,
        "calibration_construction": (
            "a separate set per replicate: 50000 fresh genuine pairs (CE2 generate_genuine_pairs) and 50000 uniform "
            "64-bit inputs (CE2 random_pairs_input), labels 1 and 0, concatenated and shuffled by "
            "PCG64(30000000 + replicate_index).permutation"),
        "folds": N_FOLDS,
        "main_fold_assignment": ("sklearn KFold(n_splits=5, shuffle=True, random_state=40000000 + replicate_index) "
                                 "applied to sample indices only, shared by every main-set arm"),
        "calibration_fold_assignment": "as main with random_state 60000000 + replicate_index",
        "permutations_per_replicate": N_PERMUTATIONS,
        "permutations_status": FROZEN,
        "permutation_construction": (
            "a cyclic permutation group of order B + 1 per replicate: o = PCG64(50000000 + replicate_index)"
            ".permutation(n) is cut, in order, into floor(n / (B + 1)) cycles of length B + 1 followed by cycles whose "
            "lengths divide B + 1 (largest divisor first) covering the remainder; sigma maps each element of a cycle "
            "to the next one (sigma[c] = roll(c, -1)); permuted arm b = 0 .. B-1 uses labels[sigma^(b+1)], where "
            "sigma^(B+1) is the identity, so {id, sigma, ..., sigma^B} is a group and the observed assignment is its "
            "identity element; global class counts are preserved exactly"),
        "s1_tiebreak": (
            "per replicate, a uniformly random permutation (t_0, ..., t_B) of {0, ..., B} drawn with "
            "random.SystemRandom (os.urandom) when the data are generated and committed in data.npz as s1_tiebreak "
            "before any probe is fitted; t_0 belongs to the observed assignment and t_(b+1) to permuted arm b"),
        "permutations_rule": ("B odd, so that B + 1 assignments split into two equal halves of ranks; the smallest "
                              "B in {5, 11, 21} whose planning P(SUPPORTED) at the reference scenario is within "
                              "0.01 of B = 21 and whose uncertainty-propagated lower bound still meets the n rule"),
        "all_replicates_required_valid": True,
        "reserve_replicates": 0,
        "early_stopping": False,
        "interim_analysis": False,
    },
    "model": {
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_sha256": PUBLISHED_CHECKPOINT_SHA256,
        "loading": "targets/autond/model_binding.py load_published (strict, pinned)",
        "representation": REPRESENTATION,
        "untrained_twin": {
            "source": CE2_SOURCE_RUN,
            "loading": "targets/autond/model_binding.py load_untrained_twin with the weights hash",
            "scope": "one fixed initialisation; results are conditional on it and say nothing about untrained "
                     "DBitNets in general",
        },
    },
    "probe": PROBE,
    "arms": {
        "A": "trained representation, actual classes, main folds",
        "P_b": "trained representation, permuted classes b = 0 .. B-1, main folds",
        "T": "twin representation, actual classes, main folds",
        "R": "raw 64-bit model input as float64, actual classes, main folds (informational only)",
        "CAL_trained": "trained representation, calibration labels, calibration folds",
        "CAL_twin": "twin representation, calibration labels, calibration folds",
        "CAL_model": "the published model's own decision 1[score > 0.5] on the calibration set (no probe)",
        "score": "balanced accuracy of the pooled held-out predictions (each example predicted exactly once)",
    },
    "endpoints": {
        "C_task": ("BA(CAL_trained) - [0.5 + kappa * max(BA(CAL_model) - 0.5, 0)], kappa = 0.5: the probe must "
                   "exceed a per-replicate recovery threshold set by the published model's own balanced accuracy on "
                   "the same calibration examples (at the published reference accuracy 0.5144 the threshold is "
                   "0.5072)"),
        "C_twin": "BA(CAL_trained) - BA(CAL_twin)",
        "S1": ("R - (B + 2) / 2, where R in {1, ..., B + 1} is the rank of the observed assignment among the B + 1 "
               "group assignments when they are ordered by (exact rational BA, recorded tie-break value), ascending; "
               "S1 > 0 exactly when R is one of the upper (B + 1) / 2 ranks (R >= 4 for B = 5); S1 is never 0"),
        "S2": "BA(A) - BA(T)",
        "raw_gap": "BA(A) - BA(R): descriptive linear-accessibility contrast only",
        "S1_margin": "BA(A) - median_b BA(P_b): descriptive only, never tested",
        "kappa": KAPPA,
        "published_reference_accuracy": PUBLISHED_ACCURACY,
        "published_reference_source": PUBLISHED_ACCURACY_SOURCE,
        "kappa_rationale": (
            "kappa is the minimum fraction of the published model's own above-chance balanced accuracy that the "
            "probe must recover. The model's head is a sigmoid of a linear function of this same layer, so a linear "
            "probe can represent the model's decision exactly and kappa = 1 is attainable in the population; a "
            "finite-sample probe loses part of it (the pipeline-faithful power check measures the loss). kappa = 0.5 "
            "requires that at least half of the known signal survives the probe pipeline, so a pipeline that "
            "recovers only a small fraction of a known effect cannot pass, while estimation loss alone does not "
            "fail a working pipeline; the power evidence reports kappa in {0, 0.25, 0.5, 0.75, 0.9}. kappa is a "
            "prospective judgment, fixed before any CE3 measurement"),
    },
    "statistics": {
        "unit": "the replicate; folds and permutations are nested within it",
        "estimand": "q(D) = P(D_i > 0) for a valid replicate i, separately for each endpoint",
        "null": "H0_D: q(D) <= 1/2",
        "alternative": "H1_D: q(D) > 1/2",
        "test": ("exact one-sided binomial test over all n valid replicates: N+ = #{i : D_i > 0}, p = P(X >= N+), "
                 "X ~ Bin(n, 1/2); reject when p < alpha (with n = 20: N+ >= 15); valid for every q <= 1/2 because "
                 "P(Bin(n, q) >= k) is increasing in q"),
        "zero_policy": ("an endpoint exactly 0.0 (including -0.0) is a non-positive outcome: it stays in the "
                        "denominator and counts against rejection; no zero is ever removed. S1 cannot be 0"),
        "no_valid_observations": "n = 0: no p-value, not rejected, reported NO_VALID_OBSERVATIONS",
        "non_finite": "a non-finite endpoint makes the replicate EXCLUDED; it is never treated as zero",
        "interval": "two-sided exact 95% Clopper-Pearson interval for q over all valid replicates",
        "descriptive": "mean, median with exact order-statistic 95% interval, quartiles, range and all values",
        "alpha": ALPHA,
        "gate_C": ("H0_C = H0_task OR H0_twin, with H0_task: q(C_task) <= 1/2 and H0_twin: q(C_twin) <= 1/2; C is "
                   "rejected only when both component tests reject at alpha (Berger's intersection-union test: "
                   "under H0_C at least one component null is true, and that component rejects with probability at "
                   "most alpha, so the gate has level alpha without adjustment)"),
        "fixed_sequence": list(FIXED_SEQUENCE),
        "multiplicity": ("fixed-sequence gatekeeping at full alpha: a step is tested only after every earlier step "
                         "rejected; the family-wise error rate is at most alpha under arbitrary dependence"),
        "s1_null": ("H0_S1: conditional on the trained representation matrix H, the label-independent folds F and "
                    "the class-count vector N = (N_0, ..., N_4) of the replicate's main-set labels, the label vector y "
                    "is exchangeable: every arrangement of labels with counts N is equally likely, whatever H is. It "
                    "holds in particular when the rows (h_j, y_j) are i.i.d. with y_j independent of h_j"),
        "s1_null_exactness": (
            "every element of the permutation group G = {id, sigma, ..., sigma^B} preserves N, and under H0_S1 y o "
            "sigma has the same conditional law as y given (H, F, N). Replacing y by y o sigma maps the assignment "
            "tuple (y, y o sigma, ..., y o sigma^B) to its cyclic shift, so, conditionally on (H, F, N), the score "
            "tuple s_j = f(H, y o sigma^j, F) of the deterministic probe has the same law as its cyclic shift, "
            "whatever ties the discrete balanced accuracy produces. The recorded tie-break is a uniform permutation "
            "independent of (H, F, N, y), so the pairs (s_j, t_j) are distinct and jointly shift-invariant in law; "
            "all B + 1 positions therefore have the same rank law, and since the ranks are a permutation of "
            "1 .. B + 1, P(R = r | H, F, N) = 1/(B + 1) for each r and P(S1 > 0 | H, F, N) = 1/2 exactly; the "
            "unconditional statement follows by averaging. The event E that coverage and convergence hold for all "
            "B + 1 assignments is a function of (H, F) and of the set of assignments, hence shift-symmetric, so the "
            "law also holds given E. Other validity checks (for example the T arm, which uses y) need not be "
            "symmetric, but {S1 rejects and all replicates VALID} is contained in {S1 rejects and E holds in every "
            "replicate}, so P(S1 rejects and all VALID) <= P(S1 rejects | E) <= alpha. Replicates are independent, "
            "so N+ ~ Bin(n, 1/2) under H0_S1 given E"),
        "s1_superseded_rule": (
            "the earlier rule S1 = BA(A) - median of B permuted BAs with zeros removed is not exact: with exchangeable "
            "two-valued scores P(s = 1) = 0.3 and B = 5 it gives P(S1 > 0 | S1 != 0) = 0.687 (and 0.313 for "
            "P(s = 1) = 0.7), so it was replaced"),
        "decision": {
            "PRECHECK_FAILED": "preflight, binding, provenance, pins, P1 or benchmark check failed",
            "INCONCLUSIVE": "fewer than 20 VALID replicates, an untestable step, or S1 not rejected",
            "POSITIVE_CONTROL_FAILED": "C not rejected; S1 and S2 are reported descriptively as not tested",
            "TARGET_DECODABLE_TWIN_UNRESOLVED": "C and S1 rejected, S2 not",
            "SUPPORTED": "C, S1 and S2 rejected in the frozen order",
        },
    },
    "failure_semantics": {
        "FAILED": ("a replicate whose gate E/F, class coverage, twin admissibility, head consistency or probe "
                   "convergence fails, or whose execution raises; retained with its partial artifacts"),
        "EXCLUDED": "a replicate with a non-finite endpoint",
        "VALID": "all checks passed and all endpoints finite",
        "replacement": "none",
    },
    "conduct": {
        "resume": ("the journal is authoritative; terminal states are never left; an artifact without a committed "
                   "transition is quarantined and its step re-executed; a committed artifact that is missing or "
                   "altered makes the replicate FAILED"),
        "smoke": "non-evidentiary, separate autond_ce3_smoke_* directory, never merged",
        "evidence_location": EVIDENCE_RELPATH,
    },
    "runtime": {
        "python": "3.11", "tensorflow": "2.15.1", "keras": "2.15.0", "numpy": "1.26.4", "pandas": "2.2.2",
        "scipy": "1.11.4", "scikit_learn": "1.9.1", "threadpoolctl": "3.7.0",
    },
    "dependencies": {"root": "audit/cryptography", "sha256": DEPENDENCIES},
    "validation_evidence": {
        "p1": {"relpath": P1_RECORD_RELPATH, "sha256": P1_RECORD_SHA256},
        "power": {"relpath": POWER_EVIDENCE_RELPATH, "sha256": POWER_EVIDENCE_SHA256},
        "benchmark": {"relpath": BENCHMARK_RELPATH, "sha256": BENCHMARK_SHA256},
        "representation_diagnostics": {"relpath": REPRESENTATION_DIAGNOSTICS_RELPATH,
                                       "sha256": REPRESENTATION_DIAGNOSTICS_SHA256},
    },
    "seeds": {
        "main_folds": FOLD_SEED_BASE, "calibration_folds": CALIBRATION_FOLD_SEED_BASE,
        "calibration_shuffle": CALIBRATION_SHUFFLE_SEED_BASE,
        "permutations": PERMUTATION_SEED_BASE, "benchmark_replicate_index": BENCHMARK_REPLICATE_INDEX,
        "s1_tiebreak": "os.urandom via random.SystemRandom; recorded in data.npz, not derived from a seed",
        "data": "os.urandom; the generated data are retained, not their seeds",
    },
    "evidence_schema": {
        "data": ("runs/<id>/data.npz: CE2 run-data arrays (key_words, pt0, pt1, c0, c1, dx, dy, weights, target), "
                 "classes (int8) and s1_tiebreak (int16, length B + 1)"),
        "calibration": ("runs/<id>/calibration.npz: genuine key_words, pt0, pt1, c0, c1, random_inputs_packed, order, "
                        "labels"),
        "representations": ("runs/<id>/representations.npz: trained_main, twin_main, trained_cal, twin_cal (float32), "
                            "score_main, score_cal (float32)"),
        "probes": ("runs/<id>/probes.npz: fold ids, permutation index vectors, and per arm and fold the scaler mean "
                   "and scale, coefficients, intercepts, iterations, gradient norms and held-out predictions"),
        "measurement": "runs/<id>/measurement.json: balanced accuracies, endpoints, coverage, convergence, timings",
        "journal": "journal.jsonl, core-journal-2",
        "certificate": CERTIFICATE_SCHEMA,
    },
    "verification": {
        "verifier": "audit/cryptography/experiments/autond/ce3/verify.py",
        "independence": (
            "imports nothing from the CE3 producer modules, targets/autond or core.evidence; it has its own journal "
            "replay, KFold, permutation-group and tie-break reconstruction, class assignment, edge selection, "
            "scaler, logistic-objective gradient, prediction, float and exact rational balanced accuracy, S1 rank, "
            "exact sign tests, Clopper-Pearson bisection and decision rule"),
        "shared_with_producer": {
            "ce2_verifier_primitives": (
                "experiments/autond/ce2/verify.py (pinned): SPECK trajectory (_trajectory, _ror), xdp+ weights "
                "(xdp_weight), word-to-bit conversion (_words_to_bits), sha256_file, canonical_sha256, content_hash, "
                "Findings, load_upstream, strict_matches_hdf5 and weights_hash; the CE3 producer uses CE2 r3 "
                "adapter.py, which the CE2 verifier was written to check independently, so these primitives are "
                "independent of the producer's cipher but are themselves CE2 code"),
            "model_building_and_binding": (
                "the pinned upstream dbitnet.make_model builder loaded through CE2 load_upstream; the producer loads "
                "the same builder through targets/autond/upstream.py and binds with targets/autond/model_binding.py, "
                "the verifier binds with the CE2 verifier's strict_matches_hdf5 and weights_hash"),
            "inference": "TensorFlow/Keras 2.15 predict for both representations and the model scores",
            "evidence_infrastructure": (
                "the verifier does not import core/evidence.py but reads the core-journal-2 journal, record caches, "
                "quarantine and staging layout that it writes, and the content-hash convention shared with CE2"),
            "optional": "--sklearn-refit also shares scikit-learn 1.9.1",
        },
        "numerical_policy": (
            "representations are re-extracted and compared bitwise; if they differ, the stored probe parameters are "
            "applied to the re-extracted representations and every held-out prediction, endpoint sign and the "
            "decision must be identical (DECISION_INVARIANT_NUMERICAL_DIFFERENCE); any sign or decision difference "
            "fails"),
    },
    "references": [
        "Bellini et al., A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input "
        "Differences, IACR ePrint 2022/1467; https://github.com/Crypto-TII/AutoND",
        "Hewitt and Liang, Designing and Interpreting Probes with Control Tasks, EMNLP 2019",
        "Maurer, Hothorn and Lehmacher, Multiple comparisons in drug clinical trials and preclinical assays: "
        "a-priori ordered hypotheses, 1995",
        "Berger, Multiparameter hypothesis testing and acceptance sampling, Technometrics 1982 (intersection-union)",
        "Hemerik and Goeman, Exact testing with random permutations, TEST 2018 (permutation subgroups)",
    ],
}

SMOKE = {"n_replicates": 2, "main_pairs_per_replicate": 6000, "calibration_per_class": 3000,
         "min_class_count_per_fold": 50, "permutations_per_replicate": 3}


class DesignError(RuntimeError):
    pass


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def design_sha256(design: dict | None = None) -> str:
    return hashlib.sha256(canonical_json(DESIGN if design is None else design)).hexdigest()


def design() -> dict:
    return copy.deepcopy(DESIGN)


def sizes(mode: str) -> dict:
    if mode == "smoke":
        return {"n_replicates": SMOKE["n_replicates"], "main_pairs": SMOKE["main_pairs_per_replicate"],
                "calibration_per_class": SMOKE["calibration_per_class"],
                "min_class_count_per_fold": SMOKE["min_class_count_per_fold"],
                "permutations": SMOKE["permutations_per_replicate"], "n_required_valid": SMOKE["n_replicates"]}
    if mode != "production":
        raise DesignError(f"unknown mode {mode!r}")
    return {"n_replicates": N_REPLICATES, "main_pairs": N_MAIN_PAIRS, "calibration_per_class": N_CALIBRATION_PER_CLASS,
            "min_class_count_per_fold": MIN_CLASS_COUNT_PER_FOLD, "permutations": N_PERMUTATIONS,
            "n_required_valid": N_REPLICATES}


def run_design(mode: str) -> dict:
    d = design()
    if mode == "smoke":
        d["smoke_overrides"] = copy.deepcopy(SMOKE)
    return d


def render_preregistration() -> dict:
    return {"schema": PREREGISTRATION_SCHEMA, "design_sha256": design_sha256(), "design": design(),
            "approval": {"status": "PENDING_REVIEW", "approved_design_sha256": None, "approved_by": None,
                         "approved_utc": None}}


def _reject_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise DesignError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def parse_json_strict(text: str):
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def load_preregistration(path: Path) -> dict:
    prereg = parse_json_strict(Path(path).read_text())
    if prereg.get("schema") != PREREGISTRATION_SCHEMA:
        raise DesignError(f"unexpected preregistration schema {prereg.get('schema')!r}")
    if prereg.get("design") != DESIGN:
        raise DesignError("preregistration design differs from design.DESIGN")
    if prereg.get("design_sha256") != design_sha256() or design_sha256(prereg["design"]) != design_sha256():
        raise DesignError("preregistration design hash does not match")
    return prereg


_WHOLE = object()


def pending_items(d=_WHOLE, path: str = "") -> list[str]:
    d = DESIGN if d is _WHOLE else d
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += pending_items(v, f"{path}.{k}" if path else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out += pending_items(v, f"{path}[{i}]")
    elif d in (PROPOSED, UNSET) or (d is None and path.endswith("target.edges")):
        out.append(path)
    return out


def production_authorisation(prereg: dict) -> tuple[bool, str]:
    d = prereg.get("design", {})
    if d.get("status") != FROZEN:
        return False, f"design status is {d.get('status')!r}, not {FROZEN!r}"
    pend = pending_items(d)
    if pend:
        return False, f"design items still pending: {pend}"
    a = prereg.get("approval") or {}
    if a.get("status") != "APPROVED":
        return False, f"preregistration approval status is {a.get('status')!r}, not 'APPROVED'"
    if a.get("approved_design_sha256") != design_sha256():
        return False, "approved_design_sha256 does not equal the design hash"
    if not a.get("approved_by") or not a.get("approved_utc"):
        return False, "approval lacks approved_by / approved_utc"
    return True, "approved"
