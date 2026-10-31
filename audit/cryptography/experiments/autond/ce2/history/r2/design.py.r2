"""
AutoND CE2 design: theory consistency of the published AutoND SPECK32/64 round-8 DBitNet.

This module is the single source of the CE2 design. ``preregistration/preregistration.json`` is
its deterministic serialisation plus an approval block; the design hash is the SHA-256 of the
canonical JSON of ``DESIGN`` (keys sorted, separators "," and ":", ASCII). Any change to the
design requires a new design version; an approved, hash-anchored preregistration is never edited.

Revision r2 (2026-10-31) supersedes r1 (design 7245d152..., never approved, never run in
production). The scientific parameters are unchanged; see ``DESIGN["revision"]["changes"]``.
The r1 design, preregistration and implementation are kept byte-identical under ``history/r1``.

The design is independent of the Gohr case study and of AutoND CE1. It imports neither.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

EXPERIMENT_ID = "AUTOND-CE2-THEORY-CONSISTENCY"
DESIGN_VERSION = "AutoND-CE2-frozen-design-2026-10-r2"
CERTIFICATE_SCHEMA = "autond-ce2-certificate-2"
PREREGISTRATION_SCHEMA = "autond-ce2-preregistration-1"
PREREGISTRATION_RELPATH = "experiments/autond/ce2/preregistration/preregistration.json"
R1_DESIGN_SHA256 = "7245d152087931cc5e0a65e05957b298a7c98e81cf4c275800f39b25579c06b1"

PUBLISHED_CHECKPOINT_RELPATH = "model_weights/speck3264_single-key_0x400000_round8_polished.h5"
PUBLISHED_CHECKPOINT_SHA256 = "a7be788dbbb760a7be9053748e6e0f37dedc5768db3a74a5c3ebd294dca4cc26"
UPSTREAM_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"

PROPOSED = "PROPOSED_PENDING_APPROVAL"

DESIGN: dict = {
    "experiment_id": EXPERIMENT_ID,
    "design_version": DESIGN_VERSION,
    "revision": {
        "supersedes": {"design_version": "AutoND-CE2-frozen-design-2026-10-r1", "design_sha256": R1_DESIGN_SHA256,
                       "status": "never approved; no production run; exploratory smoke runs only"},
        "scientific_parameters_changed": False,
        "changes": [
            "verification: inference is recomputed from verifier-recomputed model scores, not from stored "
            "scores; score agreement is assessed on ranks and on the signs that the confirmatory tests use, "
            "not by an elementwise tolerance alone",
            "evidence: the hash-chained journal is authoritative; run records are materialised from it; "
            "artifacts are committed by intent / staged write / hash / atomic promote / transition; "
            "uncommitted artifacts are quarantined and retained, never analysed",
            "hypothesis: recorded explicitly as TWO_SIDED; measured with core.theory_consistency instead of "
            "the legacy framework test that encodes a positive hypothesis and maps undefined to 0.0",
            "within-run p-values: explicit UNDERFLOW and DEGENERATE states instead of 0.0",
            "behavioural binding: the untrained twin's accuracy on the same set is recorded as a diagnostic "
            "(not a gate); an interrupted initialisation is abandoned and never resumed",
            "power: described as a scenario-based sensitivity analysis with Monte Carlo error, validated "
            "against a pipeline-faithful simulation; not a power guarantee",
            "paths: experiment relocated to audit/cryptography/experiments/autond/ce2/",
        ],
    },
    "hypothesis": {
        "direction": "TWO_SIDED",
        "h0": "the run-level population Spearman rho is distributed symmetrically about 0 (Pr[rho_i > 0] = 1/2)",
        "h1": "Pr[rho_i > 0] != 1/2; no direction is predicted",
        "measurement": "core.theory_consistency.TheoryConsistencyMeasurement(HypothesisDirection.TWO_SIDED)",
    },
    "research_question": (
        "Is the output of the published AutoND SPECK32/64 eight-round DBitNet statistically associated "
        "with a Markov-model single-characteristic probability proxy calculated from the realized "
        "eight-round XOR-difference trajectory of genuine difference-bearing SPECK plaintext pairs?"),
    "study_type": "observational association study; supports no causal claim",
    "estimand": (
        "the run-level population Spearman rank correlation between the published model's sigmoid score "
        "and the trajectory-derived log2 Markov characteristic proxy, over genuine pairs drawn from the "
        "frozen sampling distribution, for this one published checkpoint"),
    "reference": {
        "case_study": "AutoND (Bellini et al., IACR ePrint 2022/1467), Crypto-TII/AutoND",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_hash_file": "audit/cryptography/upstream_sha256.json",
        "checkpoint_relpath": PUBLISHED_CHECKPOINT_RELPATH,
        "checkpoint_sha256": PUBLISHED_CHECKPOINT_SHA256,
        "builder": "dbitnet.make_model(64)",
        "loading": ("Keras load_weights on the weights-only HDF5 file, then strict verification that every "
                    "weight array of every weight-bearing layer equals the corresponding HDF5 dataset "
                    "bit for bit, in order, with no unused dataset"),
        "observed_architecture": {
            "layers": 34, "weight_bearing_layers": 23, "hdf5_weight_arrays": 68,
            "params_total": 298113, "params_trainable": 296065, "params_non_trainable": 2048,
            "input_shape": [None, 64, 1], "output_shape": [None, 1]},
        "observed_architecture_note": (
            "298113 is the total parameter count; 2048 of these are BatchNormalization moving statistics"),
        "not_used": "no AutoND CE1 arm, warm start or checkpoint is used anywhere in CE2",
    },
    "score": {
        "definition": "sigmoid output of model.predict(X, batch_size=5000) in Keras inference mode, float32",
        "orientation": "label 1 = genuine pair (upstream convention); a larger score means 'more genuine'",
        "transformation": "none (identity); Spearman's coefficient is invariant to monotone transforms",
    },
    "cipher": {
        "name": "speck3264", "rounds": 8, "word_size": 16, "rotation_alpha": 7, "rotation_beta": 2,
        "round_function": "upstream ciphers/speck3264.py enc_one_round, key schedule expand_key",
        "input_difference": "0x400000",
        "input_difference_words": {"x": "0x0040", "y": "0x0000"},
        "input_difference_encoding": (
            "main.integer_to_binary_array(0x400000, 32), most significant bit first; convert_from_binary "
            "maps bits 0-15 to word x (left, rotated by ALPHA) and bits 16-31 to word y"),
        "model_input": ("np.hstack([convert_to_binary([x0, y0]), convert_to_binary([x1, y1])]): 64 uint8 bits "
                        "[C0 x, C0 y, C1 x, C1 y], each word most significant bit first"),
    },
    "sampling": {
        "source": "os.urandom, one byte per bit masked with & 1, exactly as upstream main.make_train_data",
        "key": "an independent uniform 64-bit master key per pair, shared by both members of the pair",
        "plaintext": "an independent uniform 32-bit plaintext P0 per pair; P1 = P0 XOR delta",
        "upstream_semantics": (
            "identical construction to the genuine (Y = 1) class of main.make_train_data: per-sample keys0, "
            "keys1 = keys0 ^ 0, pt1 = pt0 ^ delta_state, C = hstack(C0, C1)"),
        "reproducibility": (
            "datasets are not seed-replayable; reproducibility rests on the retained raw arrays and their "
            "hashes. Seeds govern only the untrained twin, the target permutation and the bootstrap"),
    },
    "population": "genuine difference-bearing pairs only; no random-class sample enters the CE2 correlations",
    "target": {
        "name": "trajectory-derived Markov characteristic probability estimate",
        "terminology": "Markov-model single-characteristic probability proxy",
        "per_round": ("alpha_r = ROR7(dx_{r-1}), beta_r = dy_{r-1}, gamma_r = dx_r, from the realized states of "
                      "both encryptions; the round key cancels in the XOR difference"),
        "factor": "xdp+(alpha_r, beta_r -> gamma_r) = 2^-k_r over uniform 16-bit operands",
        "value": "T = sum_{r=1..8} log2 xdp+ = -sum_r k_r (exact integer; larger = more probable)",
        "invalid": "any observed factor equal to zero is an implementation error: the run FAILS, no epsilon",
        "interpretation": (
            "a proxy relying on Markov/independence assumptions about intermediate propagation; NOT the exact "
            "fixed-key characteristic probability (round keys are schedule-derived, Beyne-Rijmen CRYPTO 2022), "
            "NOT the differential probability of the full cipher, NOT an unconditional bound, and NOT a "
            "measurement of what the network learned"),
    },
    "mathematical_gates": {
        "A": "Lipmaa-Moriai closed form with exact integer weights",
        "B": "carry-pair automaton with exact integer path counts, sharing no code with A",
        "C": {"word_sizes": [4, 6, 8], "coverage": "exhaustive over all (alpha, beta, gamma); A == B"},
        "D": {"word_sizes_exhaustive_literal": [4, 6, 8],
              "word16": "every round transition of the first pair of the preflight corpus, by literal 2^32 enumeration",
              "requirement": "literal operand-pair counts == A == B"},
        "E": "every production sample and round: A == B exactly and xdp+ > 0",
        "F": ("every production sample: trajectory from upstream enc_one_round reproduces upstream encrypt bit "
              "for bit; initial difference equals delta; per round the modular-addition output difference "
              "equals the next left-word difference and dy_r = ROL2(dy_{r-1}) ^ dx_r"),
        "preflight_corpus_pairs": 4096,
        "failure": "any gate failure makes the preflight PRECHECK_FAILED or the run FAILED",
    },
    "behavioural_binding": {
        "dataset": ("fresh balanced mixed set: n/2 genuine pairs (CE2 sampling) and n/2 random pairs whose 64 "
                    "input bits are uniform (upstream negative class), shuffled by PCG64(shuffle_seed)"),
        "n": 1000000, "shuffle_seed": 20261102,
        "accuracy_rule": "prediction 1 iff score > 0.5 (upstream eval_nets / keras binary_accuracy)",
        "interval": "one-sided 95% Wilson lower bound, z = 1.6448536269514722",
        "requirement": "lower bound > 0.5",
        "note": ("an independent CE2 model-binding validation, not the first measurement of this checkpoint "
                 "and not a matched replication of the published ~0.5144; it does not alter AutoND CE1"),
        "checks": ["exact label balance n/2 : n/2", "genuine ciphertexts recomputed by upstream encrypt",
                   "model input equals upstream hstack encoding", "all scores finite",
                   "shuffle reproduced from shuffle_seed"],
        "diagnostics": ("the untrained twin's accuracy and Wilson interval on the same shuffled set are "
                        "recorded; they are not a gate"),
        "scope": ("binds the checkpoint and the input pipeline behaviourally; it is not evidence that the "
                  "model learned differential cryptanalysis"),
        "interrupted_initialisation": ("an initialisation that does not complete (no run_manifest.json) is "
                                       "abandoned and never resumed; production restarts in a new directory "
                                       "and the abandoned directory is retained and reported"),
    },
    "runs": {
        "n_runs": 20, "pairs_per_run": 1000000, "min_valid_runs": 20, "reserve_runs": 0,
        "status": PROPOSED,
        "run_ids": "run_00 .. run_19",
    },
    "controls": {
        "permuted_target": {
            "definition": ("the same published-model scores against target[perm], perm = "
                           "numpy.random.Generator(PCG64(permutation_seed_base + run_index)).permutation(n)"),
            "permutation_seed_base": 7100000,
            "property": "exact marginal distribution and ties of the target are preserved"},
        "untrained_twin": {
            "definition": ("one untrained DBitNet from dbitnet.make_model(64) after "
                           "tf.keras.utils.set_random_seed(twin_seed), created once at initialisation, weights "
                           "saved and hashed, reloaded and hash-checked for every run, never trained"),
            "twin_seed": 20261031},
        "same_dataset": "all three conditions read one immutable run dataset; its content hash is checked per call",
        "role": "negative-control diagnostics; neither establishes causal use or an internal representation",
    },
    "statistics": {
        "unit": "independent evaluation run",
        "primary_endpoint": "rho_i = Spearman(score, T) per valid run, average ranks, no jitter",
        "primary_test": {"name": "exact two-sided binomial sign test over nonzero rho_i",
                         "h0": "Pr[rho_i > 0] = 1/2", "h1": "Pr[rho_i > 0] != 1/2",
                         "p_value": "min(1, 2 * smaller binomial tail)", "zeros": "ties: removed and counted"},
        "control_tests": {"statistic": "D_i = |rho_primary,i| - |rho_control,i|",
                          "test": "exact one-sided sign test, H1: Pr[D_i > 0] > 1/2; zeros removed and counted",
                          "multiplicity": "Holm across the two control tests at alpha",
                          "gate": "tested only if the primary test rejects (serial gatekeeping)"},
        "alpha": 0.05,
        "familywise": ("primary at alpha, then Holm over the two control tests at alpha; FWER of the procedure "
                       "<= alpha"),
        "effect_size": "median run-level rho, IQR, min, max, sign counts",
        "median_ci": "distribution-free order-statistic interval (ranks 6 and 15 for 20 runs; exact coverage reported)",
        "sensitivity": {"method": "percentile bootstrap of the median over whole runs",
                        "n_resamples": 10000, "seed": 20261101},
        "within_run_p": ("descriptive only, never confirmatory; t approximation with explicit UNDERFLOW "
                         "(log10 reported) and DEGENERATE (|rho| = 1) states, never 0.0"),
        "secondary": ["last-round factor log2 alone", "sum of rounds 1-7 log2 factors",
                      "per-round weight distributions"],
        "secondary_role": "descriptive; latent trajectory-derived quantities; not input-visible components",
        "power": {"report": "experiments/autond/ce2/docs/POWER_ANALYSIS.md", "target_joint_power": 0.90,
                  "nature": ("scenario-based sensitivity analysis of the complete decision rule under stated "
                             "Gaussian run-level models; Monte Carlo standard errors reported; the Gaussian "
                             "model is checked against a pipeline-faithful simulation with the real discrete "
                             "target; NOT a guarantee of power for the unknown true effect"),
                  "status": PROPOSED},
    },
    "decision": {
        "SUPPORTED": "primary rejects and both Holm-adjusted control tests reject",
        "ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED": "primary rejects; at least one control test does not",
        "INCONCLUSIVE": ("primary not significant, fewer than min_valid_runs valid runs, or an evidence check "
                         "fails; never evidence of zero association"),
        "PRECHECK_FAILED": "reference, runtime, behavioural or mathematical gate failed before analysis",
    },
    "exclusion": {
        "EXCLUDED": ("a run whose primary or either control coefficient is undefined (non-finite array, constant "
                     "score, constant target, non-finite rho); recorded with its reason, never scored as 0"),
        "FAILED": "a run whose gate fails or whose execution raises; retained with its partial artifacts",
        "replacement": "none; no reserve runs",
    },
    "conduct": {
        "stopping": "all planned runs are executed; no early stop",
        "interim": "no interim analysis",
        "resume": ("the journal is authoritative; terminal states (VALID, EXCLUDED, FAILED) are never left; "
                   "an interrupted run continues from its last committed transition; an artifact without a "
                   "committed transition is quarantined (retained, hashed, journalled, never analysed) and its "
                   "step re-executed; a committed artifact that is missing or altered makes the run FAILED"),
        "smoke": "non-evidentiary, separate autond_ce2_smoke_* directory, never merged",
    },
    "runtime": {
        "python": "3.11", "tensorflow": "2.15.1", "keras": "2.15.0", "numpy": "1.26.4",
        "pandas": "2.2.2", "scipy": "1.11.4",
        "note": "pandas is required because upstream train_nets (imported by main) imports it",
    },
    "verification": {
        "verifier": "audit/cryptography/experiments/autond/ce2/verify.py",
        "independence": ("imports nothing from the CE2 runner, analysis, theory, adapter or design modules, nor "
                         "from core.evidence; own journal replay and state machine, own word/bit conversion, "
                         "round function and key schedule, own transfer-matrix xdp+, own ranks, exact tests, "
                         "Holm and decision rule, own strict checkpoint comparison and weight hashing"),
        "shared_with_producer": ("the pinned upstream DBitNet builder and TensorFlow/Keras inference are shared: "
                                 "recomputed scores establish reproducibility of inference, not an independent "
                                 "implementation of the network"),
        "inference_from": "verifier-recomputed scores (stored scores are compared, never used for inference)",
        "numerical_policy": {
            "score_comparison": ("report bitwise equality, max absolute difference, rank-vector equality and "
                                 "tie-structure equality per run and model"),
            "decision_relevance": ("the confirmatory tests use only sign(rho_primary,i) and sign(D_c,i); the "
                                   "verifier recomputes both from its own scores"),
            "acceptance": ("PASS if every run's recomputed rho agrees with the stored-score rho to 1e-12, or if "
                           "it differs but every decision-relevant sign and the recomputed decision are "
                           "identical (reported as DECISION_INVARIANT_NUMERICAL_DIFFERENCE); FAIL if any sign or "
                           "the decision differs"),
            "score_report_threshold_abs": 1e-4,
        },
        "preflight_recomputation": ("gates C and D are independently re-executed by the verifier at word sizes "
                                    "4 and 6 (transfer matrix vs literal enumeration) and optionally 8; the "
                                    "16-bit literal checks and the encoding check are recorded, not re-executed"),
    },
    "evidence_schema": {
        "run_data": ("runs/<id>/data.npz: key_words (n,4) uint16, pt0/pt1 (n,2) uint16, c0/c1 (n,2) uint16, "
                     "dx/dy (n,9) uint16, weights (n,8) uint8, target (n,) int16"),
        "run_outputs": "runs/<id>/outputs.npz: score_published, score_twin (n,) float32, permutation (n,) int64",
        "run_record": ("runs/<id>/record.json, a cache materialised from the journal; measurement.json "
                       "(committed artifact)"),
        "journal": ("journal.jsonl, schema core-journal-2: continuous seq, experiment_id, prev_sha256; run "
                    "transitions carry from/to state and the committed fields"),
        "quarantine": "runs/<id>/quarantine/: uncommitted artifacts, each with an artifact_quarantined event",
        "certificate": CERTIFICATE_SCHEMA,
    },
    "references": [
        "Bellini et al., A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input "
        "Differences, IACR ePrint 2022/1467; https://github.com/Crypto-TII/AutoND",
        "Lipmaa and Moriai, Efficient Algorithms for Computing Differential Properties of Addition, FSE 2001, "
        "doi:10.1007/3-540-45473-X_28",
        "Beyne and Rijmen, Differential Cryptanalysis in the Fixed-Key Model, CRYPTO 2022, IACR ePrint 2022/837",
    ],
}

#: Reduced, non-evidentiary sizes for the smoke pipeline check. Never part of the design hash.
SMOKE = {"n_runs": 3, "pairs_per_run": 4096, "min_valid_runs": 3, "behavioural_n": 20000,
         "gate_c_word_sizes": [4, 6], "gate_d_word_sizes": [4, 6], "word16_transitions": 1,
         "preflight_corpus_pairs": 1024, "bootstrap_resamples": 1000}


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def design_sha256(design: dict | None = None) -> str:
    return hashlib.sha256(canonical_json(DESIGN if design is None else design)).hexdigest()


def design() -> dict:
    return copy.deepcopy(DESIGN)


def render_preregistration() -> dict:
    return {"schema": PREREGISTRATION_SCHEMA, "design_sha256": design_sha256(), "design": design(),
            "approval": {"status": "PENDING_REVIEW", "approved_design_sha256": None,
                         "approved_by": None, "approved_utc": None}}


class PreregistrationError(RuntimeError):
    pass


def load_preregistration(path: Path) -> dict:
    """Load and check a preregistration against this module. Raises on any divergence."""
    prereg = json.loads(Path(path).read_text())
    if prereg.get("schema") != PREREGISTRATION_SCHEMA:
        raise PreregistrationError(f"unexpected schema {prereg.get('schema')!r}")
    if prereg.get("design") != DESIGN:
        raise PreregistrationError("preregistration design differs from frozen_design_autond_ce2.DESIGN")
    if prereg.get("design_sha256") != design_sha256() or design_sha256(prereg["design"]) != design_sha256():
        raise PreregistrationError("preregistration design hash does not match")
    return prereg


def pending_items(d: dict | None = None, path: str = "") -> list[str]:
    """Design paths still marked PROPOSED_PENDING_APPROVAL."""
    d = DESIGN if d is None else d
    out = []
    for k, v in d.items():
        p = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            out += pending_items(v, p)
        elif v == PROPOSED:
            out.append(p)
    return out


def production_authorisation(prereg: dict) -> tuple[bool, str]:
    if pending_items(prereg.get("design", {})):
        return False, f"design items still pending approval: {pending_items(prereg['design'])}"
    a = prereg.get("approval") or {}
    if a.get("status") != "APPROVED":
        return False, f"preregistration approval status is {a.get('status')!r}, not 'APPROVED'"
    if a.get("approved_design_sha256") != design_sha256():
        return False, "approved_design_sha256 does not equal the frozen design hash"
    if not a.get("approved_by") or not a.get("approved_utc"):
        return False, "approval lacks approved_by / approved_utc"
    return True, "approved"
