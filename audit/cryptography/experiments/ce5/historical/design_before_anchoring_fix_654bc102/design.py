"""
G-CE5 design (r3): does the bound Gohr depth-10 checkpoint's output agree more,
within real pairs, with the average-key-rank score of Benamira et al. (P1,
Algorithms 1 and 2) than with the CE2 single-trail probability?

Fields equal to NOT_YET_RECORDED block approval. pending_items() lists them;
stage_problems() lists those that block a given pre-approval stage; and
production_authorisation() refuses while any remain, while the approval does
not match the design hash, without the confirm token, or without an upgraded
OpenTimestamps proof of the preregistration file.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

EXPERIMENT_ID = "GOHR-CE5-DISTRIBUTIONAL-COMPARATOR"
DESIGN_VERSION = "Gohr-CE5-design-2026-11-r3"
PREREGISTRATION_RELPATH = "experiments/ce5/preregistration/preregistration.json"
CONFIRM_TOKEN = "RUN_GCE5_PRODUCTION"

PROPOSED = "PROPOSED_PENDING_APPROVAL"
UNSET = "NOT_YET_RECORDED"

REFERENCE_CHECKPOINT_SHA256 = "256eb4a5ba93414f0a46ffd498d6121a03705698ac2a81764c2375207c5d6737"
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"

_DESIGN = {
    "experiment_id": EXPERIMENT_ID,
    "version": DESIGN_VERSION,
    "dimension": "CE",
    "claim": "G1",
    "question": (
        "Within real 5-round pairs, is the rank association between the bound network's output N and "
        "the average-key-rank score D greater than, smaller than, or practically equal to its rank "
        "association with the CE2 single-trail probability T?"),
    "why": (
        "G-CE2 (median Spearman 0.807 between N and T, 20/20 positive) excludes the registered "
        "alternatives of CE2 but not a distributional account: Benamira et al. build a key-averaged "
        "distinguisher from a masked difference distribution that closely matches the network's "
        "5-round accuracy. CE5 tests which registered account agrees better with the fixed network."),
    "hypotheses_wording": (
        "Both D and T are differential-cryptanalytic quantities. CE5 compares two cryptanalytic "
        "explanations of the network's score; it does not test whether the signal is cryptographic, a "
        "result favouring one account does not disprove the other, and no outcome establishes a causal "
        "mechanism."),
    "model": {
        "checkpoint": "Archive/best5depth10.h5",
        "sha256": REFERENCE_CHECKPOINT_SHA256,
        "binding": "audit.cryptography.ce234.production.verify_reference_model(behavioural=True), unmodified",
        "provenance": ("output of the authors' re-execution of Gohr's train_nets.py (depth 10, 5 rounds, "
                       "200 epochs), commit 2b54425; not Gohr's released weights"),
        "retraining": "none",
    },
    "cipher": {"name": "SPECK32/64", "rounds": 5, "input_difference": ["0x0040", "0x0000"],
               "conventions": "audit.cryptography.gohr.speck (Gohr's code); word order (l, r); MSB-first encoding"},
    "trail_proxy_T": {
        "source": ("audit.cryptography.ce234.production.generate_theory_data(n, with_factors=True), "
                   "imported unmodified: product of the per-round xdp+ factors of the realised trail"),
        "domain": ("real pairs only: T needs the realised intermediate differences, which a random pair "
                   "does not have; the primary comparison is therefore conditional on real pairs"),
        "persisted": "T and the per-round factors F; the verifier checks T = prod(F) for feasible pairs",
    },
    "baseline_D": {
        "construction": ("average-key-rank differential distinguisher, Benamira et al. 2021, Sect. 4.5, Supplementary "
                         "Algorithm 1 (aDDT construction) and Algorithm 2 (distinguisher), Table 7"),
        "algorithm_source_text": "docs/SOURCE_VERIFICATION.md section 2 (verbatim, P1 p. 32)",
        "primary_source_pdf_sha256": "a0f48abeefb48358ac629d92bc653dd9d141a8d83c1116be9491bb3b22d03cae",
        "algorithm_source_text_rule": (
            "Algorithms 1 and 2 transcribed verbatim with page numbers into docs/SOURCE_VERIFICATION.md; "
            "every step of akr.py and reference.py is mapped to a line of that transcription"),
        "source_location": ("P1 Sect. 4.5 pp. 17-18 (method, n = 10^7, mask, complexity); Supplementary A "
                            "p. 32 (Algorithms 1 and 2); Table 7 p. 19"),
        "mask": ["0xff8f", "0xff8f"],
        "mask_applies_to": "(dl || dv): 0xff8f on dl, 0xff8f on dv (Algorithm 1 L11)",
        "mask_weight": 26,
        "table_definition": ("Algorithm 1: dl = l xor l', dv = l xor r xor l' xor r' at nr - 1 rounds; "
                             "aDDT[(dl || dv) & m] += 1/n; stored as integer counts over the 2^26 masked cells"),
        "table_rounds": 4,
        "table_pairs": 10_000_000,
        "candidates": ("Algorithm 2 L4: key = 0 .. 2^12 - 1, literal; one-round decryption of both "
                       "ciphertexts per candidate"),
        "lookup": ("aDDT[(dl || dv) & m]; Algorithm 2 L9 omits '& m', read as the masked cell written by "
                   "Algorithm 1 L11 (interpretation recorded in docs/SOURCE_VERIFICATION.md)"),
        "score": "score / 2^12 (Algorithm 2 L11), continuous, pre-threshold, kept for the correlations",
        "decision": "real iff score / 2^12 > 2^-HW(m) = 2^-26, strict (Algorithm 2 L11); used by the gate",
        "smoothing": "none: Algorithm 1 L2 initialises to zero and unseen cells contribute 0",
        "table_check": (
            "an independent 10^5-pair sample from reference.py (own SPECK, own random stream) is compared "
            "with the recorded table over its 200 most frequent cells with expected count >= 5 plus a pooled "
            "remainder; PASS iff chi-square p >= 0.001 at the registered rounds and p < 1e-12 at one round "
            "fewer and one more"),
        "implementation": "experiments/ce5/akr.py; independent scalar reference experiments/ce5/reference.py",
        "table_sha256": "09ff9388871fc6e41d2099d3a41619430528e49ce78c6ee5e919531f2cf54be6",
        "table_record": "experiments/ce5/tables/tables_manifest.json (r3, Algorithm 1; table check PASS)",
        "table_validation": {"record": "validation/r3_final_20261105_tablecheck_v2/table_validation.json",
                             "sha256": "28808b8a5f34831d4a075b26019b84c4ba9eb2c051f1c6de77abc3dcc72099cf",
                             "rule": ("output of the tablecheck stage with overall decision PASS (every registered "
                                      "table PASS, n_pairs registered, (dl, dr) control rejected, compaction "
                                      "bijective); calibration and production refuse without it")},
    },
    "secondary_comparator_LR": {
        "construction": ("Algorithm 1 table at nr = 5 rounds (10^7 pairs, independent of the aKR table), "
                         "read at the masked (dl || dv) of the ciphertexts themselves, no key guessing; "
                         "secondary, not a published distinguisher"),
        "table_sha256": "2ba703dbfaf4e5d5bfb0f1c83135dbae890dd816725ce1a791187a97f9d91e48",
    },
    "instrument_order": (
        "tables are built once, hashed and recorded here before calibration; calibration, gate and "
        "production all use the recorded tables"),
    "previous_design_sha256": "edb4bb88bd6dbef74c7710e4ef9b5b6f3344e80e7e4f85083e1289b34366e33f",
    "design_history": ["edb4bb88bd6dbef74c7710e4ef9b5b6f3344e80e7e4f85083e1289b34366e33f (before margin and tolerance)",
                       "b5c4ac4fccc804563ff406a5698d489f6b38edc728b0ea7d73f7ed1282096aa2 (before n_replicates; calibrated under this design)",
                       "f6f0ceb994ed02473bbe4e44ce2c82374f453ec0f2f4bc3ed8471082e467619f (before the tolerance_rule wording correction)"],
    "superseded": ("design r2 (2^13-class average of all 2^16 subkeys, (delta_l, delta_r) table) is kept in "
                   "historical/r2_superseded_20261105 and is not evidence for this design"),
    "not_run": ["M-ODT/LightGBM (its masks are derived from the network, so it is not an independent baseline)"],
    "gate": {
        "purpose": ("the baseline must reproduce the published aKR result, not merely beat the DDT, before "
                    "any confirmatory data exist"),
        "published_akr": {"accuracy": 0.9298, "tpr": 0.9076, "tnr": 0.9522,
                          "location": "P1 Table 7, p. 19, row 5 (92.98%, 90.76%, 95.22%)"},
        "published_ddt_accuracy": 0.911,
        "published_ddt_location": ("P1 Table 1, p. 9, row D5 (0.911, 0.877, 0.947; taken from Table 2 of Gohr "
                                   "2019); also P1 Table 10, p. 27"),
        "published_evaluation_size": ("not stated for Table 7; Table 8 (p. 19) uses 10^5 pairs, about 50,000 "
                                      "per class"),
        "pairs_per_class": 100_000,
        "table_variability": (
            "before approval, non-confirmatory: 5 independent tables built from fresh data, each scored on "
            "its own fresh 10^5 + 10^5 pairs; sigma_t = SD of the 5 accuracies (and of TPR, TNR)"),
        "tolerance_rule": ("tau = max(0.005, 3 * sigma_t(accuracy)) and tau_rate = max(0.010, 3 * "
                           "max(sigma_t(TPR), sigma_t(TNR))), fixed from table_variability before approval; the "
                           "floors 0.005 and 0.010 were proposed before either variability run and confirmed on "
                           "2026-11-05 (tolerance_decision)"),
        "tolerance": {"accuracy": 0.005, "rates": 0.010},
        "tolerance_decision": ("set on 2026-11-05 by operator authorisation; equals the registered tolerance_rule "
                               "applied to the variability records: 3 SD of accuracy 0.0012 (run 1) and 0.0015 (run 2) "
                               "and of the rates 0.0024 and 0.0026 are below the floors, so tau = 0.005 and "
                               "tau_rate = 0.010; floors written before either variability run; units: absolute "
                               "proportions; no gate or production data existed"),
        "tolerance_source": {"variability_v2": "validation/r3_final_20261105_variability_v2/table_variability.json",
                             "variability_v2_sha256": "ecda5bd5d8a55960a7cd5807f8599600d834ab87555c169c2fa93863425530f5",
                             "variability_v1": "validation/r3_final_20261105/table_variability.json",
                             "variability_v1_sha256": "7f2433a64b2b060eb228158e7c14761be7bd75c73e0249203d78477f5c8fd843"},
        "rule": (
            "REPRODUCED iff the Clopper-Pearson 95% interval of accuracy lies inside [0.9298 - tau, 0.9298 + "
            "tau] and the CP intervals of TPR and TNR lie inside their published values +- tau_rate. "
            "SEPARATED (reported) iff the accuracy CP lower limit exceeds (0.9298 + 0.911) / 2. CE5 proceeds "
            "only on REPRODUCED; SEPARATED without REPRODUCED is reported as an adequacy result and CE5 is "
            "UNADJUDICATED"),
        "data": "fresh pairs from os.urandom after approval, independent of the tables and of every replicate",
        "reported": "accuracy, TPR, TNR, full confusion matrix, both gate components",
        "on_failure": ("CE5 is UNADJUDICATED and the failure is reported; no replicate is run and no "
                       "substitute baseline is run under this design"),
    },
    "replicate": {
        "unit": "replicate (independent fresh data); never the pair",
        "real_pairs_per_replicate": 1_000_000,
        "real_pairs_record": ("validation/benchmark.json (2026-11-05T11:16:06Z) and validation/r3_final_20261105/"
                              "benchmark.json: 0.109 and 0.110 ms per real pair, where one real pair stands for one "
                              "real and one random pair (generation, inference, aKR and LR scoring); 36.3 bytes per "
                              "real pair; at 10^6 pairs and R = 60, 1.84 h and 2.2 GB; peak footprint 850-885 MB"),
        "random_pairs_per_replicate": "equal to real_pairs_per_replicate",
        "n_replicates": 20,
        "n_replicates_record": {"calibration": "validation/calibration_20261105T164019Z/calibration.json", "calibration_sha256": "8ac4e70810ae5de3218a1d024612e1230c1a1444a61fffc9f0365982991ad960",
                                "verification": "validation/calibration_20261105T164019Z/calibration_verification.json", "verification_sha256": "0c6f65889c4891ca05ae6d777abbe308df47d2688d314e5cf30d0a35200e639c",
                                "basis": ("registered rule applied to the blinded calibration: sigma_plan = 0.0016756; "
                                          "at R = 20 all three Bonferroni one-sided Clopper-Pearson lower bounds are "
                                          "0.9999 >= 0.80; 20 is the smallest value of the registered grid; "
                                          "independently verified")},
        "minimum_valid_replicates": "all registered replicates",
        "rules": {
            "real_pairs_per_replicate": (
                "largest of 10^4, 3x10^4, 10^5, 3x10^5, 10^6 whose projected production time (trail "
                "generation, network inference and aKR scoring of real and random pairs, each measured by the "
                "benchmark) at the worst-case R = 60 is at most 12 hours on the registered machine and whose "
                "raw files at R = 60 fit in 10 GB; fixed before calibration"),
            "n_replicates": (
                "smallest R in {20, 25, ..., 60} for which, under a normal planning model with SD sigma_plan "
                "(analysis.planning_sd), the 99% Clopper-Pearson lower bounds of the Monte Carlo estimates "
                "(50,000 simulated sets, common random numbers) of P(EQUIVALENT | Delta = 0), "
                "P(DISTRIBUTION_GREATER | Delta = 2m) and P(TRAIL_GREATER | Delta = -2m) are all >= 0.80; "
                "60 if none, with the shortfall reported. sigma_plan = max(s, 1.4826 * MAD) * sqrt(df / "
                "chi2_{0.20, df}); since it is never below the exact 80% upper confidence bound for sigma under "
                "normality, its coverage is at least 80% under normality; no other coverage is claimed"),
            "calibration": ("10 blinded replicates at the chosen pair count on fresh data, after the margin "
                            "is fixed: only the spread of Delta is kept, its location and all correlations are "
                            "discarded; never evidence. The planning spread is sigma_plan (n_replicates rule)"),
        },
        "data": "fresh per replicate from os.urandom; raw arrays persisted before analysis",
    },
    "primary": {
        "statistic": (
            "Delta_r = atanh(rho_s(N, D | real)) - atanh(rho_s(N, T | real)) on the same real pairs of "
            "replicate r, rho_s = Spearman with average ranks for ties"),
        "dependence": ("the two correlations share N and the pairs; their dependence is absorbed by taking "
                       "the paired difference within each replicate and inferring across independent replicates"),
        "interval": ("distribution-free interval for the median of Delta_r from binomial order statistics, "
                     "coverage >= 95% (95.9% at R = 20)"),
        "margin": 0.10,
        "margin_proposal": 0.10,
        "margin_decision": ("set to 0.10 on 2026-11-05 by operator authorisation, equal to the proposal written in "
                            "design r1 before any CE5 data existed; units: Fisher-z difference of Delta_r; not "
                            "informed by calibration, variability or any confirmatory data"),
        "margin_rationale": (
            "proposal only: on the Fisher-z scale, which approximately stabilises the sampling variance of a "
            "correlation; at the G-CE2 level (rho_s = 0.807) 0.10 corresponds to a change of about 0.032 in "
            "rho_s (rho_s^2 changes by about 0.053; for a rank correlation this is not a proportion of variance "
            "explained in the regression sense). Under the revised outcomes a DISTRIBUTION_GREATER or TRAIL_GREATER verdict needs the "
            "whole interval beyond the margin, so the margin is the smallest difference the experiment can "
            "certify. The operator sets the value before calibration; no data have informed it."),
        "margin_sensitivity": [0.05, 0.20],
        "outcomes": {
            "DISTRIBUTION_GREATER": "lower limit >= margin (a practically meaningful difference is demonstrated)",
            "TRAIL_GREATER": "upper limit <= -margin",
            "EQUIVALENT": "-margin < lower limit and upper limit < margin",
            "INCONCLUSIVE": ("otherwise; a non-significant difference is never read as equivalence, and a "
                             "difference whose interval excludes 0 but crosses the margin is reported only as "
                             "a direction (qualifier DIRECTION_D or DIRECTION_T), never as practically "
                             "meaningful"),
        },
        "alpha": 0.05,
    },
    "secondary": {
        "resolution_matched": (
            "T takes few distinct values and D is nearly continuous, which caps rho(N, T) below rho(N, D) for "
            "reasons of resolution alone. Delta_matched_r uses D re-ranked with T's tie structure (D's order, "
            "T's tie groups) in place of D; same interval and outcomes, descriptive"),
        "margin_sensitivity": "the primary outcome at each margin in primary.margin_sensitivity, descriptive",
        "class_conditional": (
            "rho_s(N, D | real) and rho_s(N, D | random), descriptive; T is undefined for random pairs"),
        "lr_comparator": "Delta_LR_r = atanh(rho_s(N, LR)) - atanh(rho_s(N, T)), same interval, descriptive",
        "incremental_value": {
            "target": "van der Waerden normal scores of N",
            "features": "normal scores of T and D, each with degree-3 polynomial terms; M_TD adds T*D",
            "fit": "ordinary least squares on the first half of each replicate's real pairs",
            "loss": "mean squared error on the second half; the same evaluation pairs for all three models",
            "statistics": "Delta_D = L(M_T) - L(M_TD); Delta_T = L(M_D) - L(M_TD)",
            "test": "exact two-sided sign test over replicates for each; Holm over the two",
            "interpretation": ("predictive information about N beyond the competing predictor; not causal "
                               "reliance or mechanism"),
        },
    },
    "anchoring": ("preregistration.json must have an OpenTimestamps proof preregistration.json.ots whose "
                  "embedded SHA-256 equals the file's and which contains a Bitcoin attestation tag, and a run of "
                  "'ots -v verify' recorded by runner.record_ots_verification as "
                  "preregistration.json.ots.verification.json with exit code 0, a 'Success! Bitcoin block' line, "
                  "no failure line, a 'Got digest' line equal to the file's SHA-256, and file and proof hashes "
                  "equal to the current bytes; production refuses otherwise. The digest and tag checks are "
                  "byte-level consistency checks only; only 'ots verify', which needs a Bitcoin node, checks the "
                  "commitment path against a block header"),
    "verification": (
        "verify.py imports only design.py and reference.py and consumes the raw files and tables. It checks "
        "the preregistration, approval and timestamp records; recomputes table hashes and the independent "
        "table check; the gate's confusion matrix, CP intervals, components and decision; D for 32 random "
        "pairs per class per replicate with the literal scalar Algorithm 2 (2^12 candidates); a class-label "
        "sanity check; T = prod(F); every correlation, Fisher transform, paired difference, interval, "
        "secondary outcome, qualifier, incremental value, Holm adjustment and the decision; and optionally N "
        "for 256 pairs per class per replicate from the checkpoint. Any mismatch makes the result "
        "NOT_VERIFIED"),
    "exclusions": "a replicate is re-drawn only after an infrastructure failure, kept in the log",
    "environment_requirement": "Python 3.11, TensorFlow 2.21, Keras 3, numpy 2.x, scipy 1.17, CPU",
    "not_claimed": [
        "that the network computes D or T",
        "that rejecting the single-trail account rejects cryptographic learning",
        "anything about other checkpoints, rounds or input differences",
    ],
}


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def design() -> dict:
    return copy.deepcopy(_DESIGN)


def design_sha256(d: dict | None = None) -> str:
    return hashlib.sha256(canonical_json(design() if d is None else d)).hexdigest()


def pending_items(d=None, path: str = "") -> list[str]:
    d = design() if d is None else d
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += pending_items(v, f"{path}.{k}" if path else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out += pending_items(v, f"{path}[{i}]")
    elif d == UNSET:
        out.append(path)
    return out


def render_preregistration(d: dict | None = None) -> dict:
    d = design() if d is None else d
    return {"design": d, "approval": {"status": PROPOSED, "design_sha256": design_sha256(d),
                                      "approved_by": None, "approved_utc": None}}


def load_preregistration(path: Path) -> dict:
    return json.loads(Path(path).read_text())


OTS_BITCOIN_TAG = bytes.fromhex("0588960d73d71901")
OTS_PENDING_TAG = bytes.fromhex("83dfe30d2ef90c8e")

ALGORITHM_FIELDS = ("baseline_D.algorithm_source_text", "baseline_D.source_location", "baseline_D.table_definition",
                    "baseline_D.table_rounds", "baseline_D.table_pairs", "baseline_D.candidates",
                    "baseline_D.smoothing")
STAGE_REQUIREMENTS = {
    "tables": ALGORITHM_FIELDS,
    "benchmark": ALGORITHM_FIELDS + ("baseline_D.table_sha256", "secondary_comparator_LR.table_sha256"),
    "table_variability": ALGORITHM_FIELDS,
    "calibrate": ALGORITHM_FIELDS + ("baseline_D.table_sha256", "secondary_comparator_LR.table_sha256",
                                     "baseline_D.table_validation.record", "baseline_D.table_validation.sha256",
                                     "primary.margin", "replicate.real_pairs_per_replicate"),
}


def stage_problems(stage: str, d: dict | None = None) -> list[str]:
    """Unset fields that block a pre-approval stage; production uses
    production_authorisation(), which requires every field."""
    pend = set(pending_items(d))
    return [f for f in STAGE_REQUIREMENTS[stage] if f in pend]


def ots_digest(path: Path) -> str | None:
    """SHA-256 embedded in an OpenTimestamps detached proof, or None."""
    p = Path(path)
    if not p.exists():
        return None
    b = p.read_bytes()
    n = len(OTS_MAGIC)
    if b[:n] != OTS_MAGIC or b[n:n + 1] != b"\x01" or b[n + 1:n + 2] != b"\x08":
        return None
    return b[n + 2:n + 34].hex()


def ots_attestations(path: Path) -> dict:
    b = Path(path).read_bytes() if Path(path).exists() else b""
    return {"bitcoin": b.count(OTS_BITCOIN_TAG), "pending": b.count(OTS_PENDING_TAG)}


def production_authorisation(prereg: dict, token: str | None, *, prereg_path: Path | None = None) -> tuple[bool, str]:
    d = prereg.get("design", {})
    a = prereg.get("approval", {})
    if pending_items(d):
        return False, f"design has unset items: {pending_items(d)}"
    if a.get("status") != "APPROVED":
        return False, f"approval status is {a.get('status')!r}, not APPROVED"
    if a.get("design_sha256") != design_sha256(d):
        return False, "approval hash does not match the design"
    if not a.get("approved_by") or not a.get("approved_utc"):
        return False, "approval has no approver or time"
    if token != CONFIRM_TOKEN:
        return False, "confirm token missing or wrong"
    if prereg_path is None:
        return False, "preregistration file path required for the timestamp check"
    proof = Path(str(prereg_path) + ".ots")
    digest = hashlib.sha256(Path(prereg_path).read_bytes()).hexdigest()
    if ots_digest(proof) != digest:
        return False, "no OpenTimestamps proof matching the preregistration file"
    if ots_attestations(proof)["bitcoin"] < 1:
        return False, "OpenTimestamps proof has no Bitcoin attestation yet (run ots upgrade, then ots verify)"
    ok, why = ots_verification_ok(Path(str(proof) + ".verification.json"), Path(prereg_path))
    if not ok:
        return False, why
    return True, "approved"


OTS_FAILURE_MARKERS = ("File does not match original", "Could not connect", "verification failed",
                       "not found", "Pending confirmation", "Error", "Bitcoin disabled")


def ots_verification_ok(record: Path, prereg_path: Path) -> tuple[bool, str]:
    """A recorded `ots -v verify` run (runner.record_ots_verification) counts
    only if it exited 0, reports success against a Bitcoin block, reports no
    failure, hashed the current preregistration (its 'Got digest' line), and
    was made on the current file and proof bytes."""
    if not record.exists():
        return False, f"no recorded 'ots verify' run ({record.name}); an attestation tag in the proof is not verification"
    r = json.loads(record.read_text())
    out = r.get("stdout", "") + r.get("stderr", "")
    file_sha = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
    proof = Path(str(prereg_path) + ".ots")
    proof_sha = hashlib.sha256(proof.read_bytes()).hexdigest() if proof.exists() else None
    if r.get("exit_code") != 0:
        return False, f"recorded 'ots verify' exited {r.get('exit_code')}"
    if "Success! Bitcoin block" not in out:
        return False, "recorded 'ots verify' output has no Bitcoin success line"
    bad = [m for m in OTS_FAILURE_MARKERS if m in out]
    if bad:
        return False, f"recorded 'ots verify' output reports {bad}"
    if f"Got digest {file_sha}" not in out:
        return False, "recorded 'ots verify' did not hash the current preregistration file"
    if r.get("file_sha256") != file_sha or r.get("proof_sha256") != proof_sha:
        return False, "recorded 'ots verify' was made on different file or proof bytes"
    return True, "verified"


def gate_spec(d: dict) -> dict:
    g = d["gate"]
    pub = g["published_akr"]
    tol = g["tolerance"]
    a, ddt = float(pub["accuracy"]), float(g["published_ddt_accuracy"])
    if a <= ddt:
        raise ValueError("published aKR accuracy must exceed the DDT accuracy")
    tau, tau_r = float(tol["accuracy"]), float(tol["rates"])
    return {"accuracy": (a - tau, a + tau), "tpr": (pub["tpr"] - tau_r, pub["tpr"] + tau_r),
            "tnr": (pub["tnr"] - tau_r, pub["tnr"] + tau_r), "separation_lower": (a + ddt) / 2}
