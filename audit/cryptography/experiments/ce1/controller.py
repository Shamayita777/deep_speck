"""
CE1 production controller (resumable).

This module is the ONLY production path for CE1. It replaces the earlier
closure pair `data_fn` / `train_eval_fn`, whose training-time validation
split travelled through the mutable attribute `data_fn.last_validation`
and whose training data existed only in memory (os.urandom, not
seed-replayable), so no interrupted arm could ever be resumed.

    controller
      -> prepare / load immutable block dataset (content-hashed, fail closed)
      -> validate hashes + provenance
      -> inspect arm state
      -> SKIP | EVALUATE | RESUME | RERUN | TRAIN | PAUSE (needs operator)
      -> GohrTrainer.train(initial_epoch=N, terminal_epoch=200,
                           extra_callbacks=[CE1EpochCheckpoint])
      -> per-epoch checkpoint (weights + optimizer) + state + ledger
      -> TRAINING_COMPLETE
      -> sealed-set evaluation of the persisted terminal model
      -> EVALUATED
      -> block VALID only when BOTH arms are EVALUATED and paired

FROZEN DESIGN - untouched here. n_blocks, min_valid_blocks, seeds and the
randomized assignment (seed_manifest), sample counts, terminal epoch,
FINAL_EPOCH rule, sealed set, test and analysis are all consumed, never
chosen. Resume semantics: checkpoint/state resume infrastructure =
IMPLEMENTED; production checkpoint/state resume = UNPROVEN until the real
interruption integration test passes; bit-exact trajectory = NOT CLAIMED; shuffle_stream_continuity = false for
every resumed arm. shuffle is NOT forced off.

SUBORDINATION TO THE FROZEN PROTOCOL. The frozen counting rule is
authoritative: 8 predeclared blocks, >= 6 VALID required, <= 2 failures
tolerated, ALL valid blocks analysed, no early stop when 6 valid blocks
appear first. This controller adds no stopping rule and no discretionary
action: there is NO operator abandonment. Every engineering decision below
is a function ONLY of artifact/state validity and interruption/failure
status (BLOCK_RESOLUTION_RULES, recorded in every run manifest before any
training); none reads accuracy, deltas, p-values or validation metrics.
  * an arm is evaluated only when BOTH arms of its block are trained;
  * an EVALUATED arm is never retrained, whatever happens to its artifacts;
  * interruptions (SIGKILL, KeyboardInterrupt, SystemExit) are resumed and
    never count as failures; exceptions count toward a predeclared cap;
  * a block's resolution (VALID / FAILED / PENDING) is re-derived from the
    evidence on every pass by `block_validity_gate`, never set by hand.

TensorFlow is imported lazily, only inside training/evaluation, so the
parent process of the 2-GPU launcher never initialises CUDA.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit.cryptography.audit_config import REFERENCE
from audit.cryptography.experiments.ce1 import ce1_datasets as DS
from audit.cryptography.experiments.ce1 import resume as R
from audit.cryptography.experiments.ce1.ce1_ledger import CE1Ledger, RESUME_SEMANTICS
from audit.cryptography.experiments.ce1.design import build_ce1_block
from audit.cryptography.frozen_design import CE1 as CE1_SPEC, DESIGN_SPECIFICATION_VERSION

EXPERIMENT_ID = "CE1-SIGNAL-DESTRUCTION"
#: Bumped to 2: evaluation record schema (exact n_correct/n, bound predictions,
#: ledger-bound evaluation hash), PAUSED semantics, path containment on load.
CONTROLLER_PROTOCOL_VERSION = "ce1-controller-2"
#: A run directory may only be reopened by a controller whose protocol is in
#: this set. Anything else is refused rather than silently reinterpreted.
COMPATIBLE_CONTROLLER_PROTOCOLS = frozenset({CONTROLLER_PROTOCOL_VERSION})
RUN_MANIFEST = "run_manifest.json"
LEDGER_NAME = "ledger.jsonl"

#: The ONE sealed evaluation set of CE1. Generated once (production_20260929),
#: verified by content hash. Production refuses any other set and never
#: generates a new one.
PINNED_SEALED_EVALUATION_SHA256 = (
    "33df1f95a5df5eb38bfba82ede7aa7d8c260b52b246827d677c56a7876e2a3e5")

#: Canonical location of the preserved legacy run (read-only for this module).
LEGACY_RUN_RELATIVE = "audit/cryptography/evidence_current/ce1/production_20260929"

ARMS = R.ARMS                       # ("baseline", "destroyed")

#: OPERATIONAL SAFETY CAPS - not scientific criteria. Reaching one PAUSES the
#: arm (NEEDS_OPERATOR); it never fails a block and never consumes the frozen
#: failure tolerance. Counted from failure STATUS only.
MAX_ARM_FAILURES = 3        # exceptions raised by training/evaluation code, per arm
MAX_ARM_RERUNS = 2          # from-scratch restarts forced by an unusable checkpoint

#: Predeclared block-resolution rules. Written into every run manifest at
#: creation (before any training or evaluation) and never changed by a run.
BLOCK_RESOLUTION_RULES = {
    "version": "ce1-block-resolution-2",
    "frozen_counting_rule": ("8 predeclared blocks; a result requires >= 6 VALID blocks; at "
                             "most 2 non-valid blocks tolerated; ALL valid blocks are "
                             "analysed; no early stop when 6 valid blocks appear first"),
    "VALID": ("both arms EVALUATED and every required invariant EVIDENCED by artifacts: "
              "evaluation_bound, terminal_epoch, assignment, shared_training_inputs, "
              "label_construction, shared_validation"),
    "NOT_COUNTED (final gate only; permanent, evidence-based)": {
        "N-INSUFFICIENT-EVIDENCE": "a required invariant is permanently UNVERIFIABLE for a "
                                   "completed block (legacy block whose training/validation "
                                   "data were never persisted)",
        "N-FINAL-EPOCH": "the evaluated model is provably not the terminal-epoch model "
                         "(optimizer step counter != terminal_epoch x steps_per_epoch)",
        "N-PAIRING": "the evaluated arms provably did not share one committed dataset with "
                     "the frozen destroyed-label construction",
    },
    "NEEDS_OPERATOR (unresolved; finalize refuses)": [
        "operational retry cap reached (MAX_ARM_FAILURES / MAX_ARM_RERUNS)",
        "evaluation record / predictions / ledger binding fail verification",
        "terminal model, state file or committed dataset missing or altered",
        "arm bound to a different configuration, seed, assignment or sealed set",
        "training refused because it would break pairing",
    ],
    "PENDING": "normal progress; finalize refuses",
    "interruptions": "SIGKILL, KeyboardInterrupt and SystemExit are interruptions: resumed, "
                     "never counted as failures",
    "operator_actions": {
        "GRANT_RETRY": "re-arm the retry caps of a capped arm (recorded)",
        "REEVALUATE": "deterministically re-score the hash-verified terminal model after an "
                      "evaluation-integrity failure; results must agree with every earlier "
                      "ledger evaluation of the same model or the arm stays paused",
        "RESTART_BLOCK": "archive and restart a controller block only while NO arm of it has "
                         "ever been evaluated (ledger-checked)",
        "forbidden": "no operator action can fail, exclude, abandon or select a block",
    },
    "never_used": ["accuracy", "paired difference", "p-value", "validation loss/accuracy",
                   "any function of an evaluation result"],
    "caps_are_not_scientific_failures": True,
}

# arm actions (PAUSE = NEEDS_OPERATOR: an engineering stop, never a failure)
TRAIN, RESUME, RERUN, EVALUATE, SKIP, PAUSE = (
    "TRAIN", "RESUME", "RERUN", "EVALUATE", "SKIP", "PAUSE")


class ControllerError(RuntimeError):
    pass


class RunDirectoryError(ControllerError):
    """The run directory is missing, degenerate, misplaced, or not a CE1 run."""


#: Production CE1 run directories are direct children of this directory,
#: relative to the repository root (e.g. .../ce1/production_20260930).
CE1_RUNS_RELATIVE = Path("audit/cryptography/evidence_current/ce1")

_DEGENERATE_RUN_DIRS = ("", ".", "./")


def _reject_degenerate(run_dir) -> Path:
    """
    `Path("")` silently equals `Path(".")` - the working directory. A run
    directory must always be explicit; an empty value or '.' is never allowed
    to fall back to the working directory (or the repository root).
    """
    raw = "" if run_dir is None else str(run_dir)
    if raw.strip() in _DEGENERATE_RUN_DIRS:
        raise RunDirectoryError(
            f"an explicit CE1 run directory is required; got {raw!r}, which would resolve "
            "to the working directory. Refusing to fall back to '.'.")
    return Path(run_dir)


def validate_run_dir(run_dir, config: "CE1RunConfig", *, repo_root=None,
                     must_exist: bool = False) -> Path:
    """
    Fail-closed validation of a run directory, applied BEFORE anything is
    written (in addition to, never instead of, output_policy):

      * never empty / '.' (no silent fallback to the working directory);
      * production: must be a DIRECT CHILD of <repo>/audit/cryptography/
        evidence_current/ce1/ (so it can be neither the evidence root, nor ce1/
        itself, nor nested inside another run such as the legacy run);
      * an existing directory is adopted only if it is a CE1 controller run
        (has run_manifest.json) or is empty - a non-empty directory without a
        manifest (e.g. the preserved legacy run) is refused.
    """
    p = _reject_degenerate(run_dir)
    target = p.resolve()
    if config.production:
        root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
        runs_root = (root / CE1_RUNS_RELATIVE).resolve()
        if target.parent != runs_root:
            raise RunDirectoryError(
                f"production run directory {str(run_dir)!r} resolves to {target}; it must be a "
                f"direct child of {runs_root} (e.g. {runs_root / 'production_20260930'}).")
    if target.exists():
        if not target.is_dir():
            raise RunDirectoryError(f"{target} exists and is not a directory")
        if not (target / RUN_MANIFEST).exists() and any(target.iterdir()):
            raise RunDirectoryError(
                f"{target} exists, is not empty and is not a CE1 controller run (no "
                f"{RUN_MANIFEST}). Refusing to adopt it: it may be a preserved legacy run.")
    elif must_exist:
        raise RunDirectoryError(f"run directory {target} does not exist")
    if must_exist and not (target / RUN_MANIFEST).exists():
        raise RunDirectoryError(f"{target} is not a CE1 controller run (no {RUN_MANIFEST})")
    return p


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class CE1RunConfig:
    """
    Every value that defines what an arm IS. Production values are taken
    from the frozen design and the reference configuration; they are
    validated, never chosen. Non-production configs exist only so the real
    controller path can be exercised at toy scale in tests.
    """
    production: bool
    n_blocks: int
    min_valid_blocks: int
    terminal_epoch: int
    train_samples: int
    validation_samples: int
    evaluation_samples: int
    batch_size: int
    depth: int
    l2_reg: float
    rounds: int
    differential: tuple
    high_learning_rate: float
    low_learning_rate: float
    lr_cycle_length: int
    optimizer: str
    loss: str
    save_best_only_debug: bool
    seed_base: int
    assignment_seed: int
    design_version: str
    expected_sealed_sha256: str | None
    keep_last_checkpoints: int = 2          # operational; not in the fingerprint

    @classmethod
    def production_config(cls) -> "CE1RunConfig":
        from audit.cryptography.experiments.ce1.gohr_signal_destruction import (
            PRODUCTION_TRAINING)
        return cls(
            production=True,
            n_blocks=CE1_SPEC.n_blocks, min_valid_blocks=CE1_SPEC.min_valid_blocks,
            terminal_epoch=CE1_SPEC.terminal_epoch,
            train_samples=CE1_SPEC.train_samples,
            validation_samples=CE1_SPEC.validation_samples,
            evaluation_samples=CE1_SPEC.evaluation_samples,
            batch_size=PRODUCTION_TRAINING["batch_size"],
            depth=REFERENCE.depth, l2_reg=REFERENCE.l2_reg, rounds=REFERENCE.rounds,
            differential=tuple(REFERENCE.differential),
            high_learning_rate=PRODUCTION_TRAINING["high_learning_rate"],
            low_learning_rate=PRODUCTION_TRAINING["low_learning_rate"],
            lr_cycle_length=10,
            optimizer=PRODUCTION_TRAINING["optimizer"], loss=PRODUCTION_TRAINING["loss"],
            save_best_only_debug=PRODUCTION_TRAINING["save_best_only"],
            seed_base=CE1_SPEC.seed_base, assignment_seed=CE1_SPEC.assignment_seed,
            design_version=DESIGN_SPECIFICATION_VERSION,
            expected_sealed_sha256=PINNED_SEALED_EVALUATION_SHA256,
        )

    @classmethod
    def testing_config(cls, **overrides) -> "CE1RunConfig":
        base = asdict(cls.production_config())
        base.update(production=False, expected_sealed_sha256=None)
        base.update(overrides)
        base["differential"] = tuple(base["differential"])
        return cls(**base)

    @classmethod
    def from_dict(cls, d: dict) -> "CE1RunConfig":
        d = dict(d)
        d["differential"] = tuple(d["differential"])
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["differential"] = list(self.differential)
        return d

    def scientific_dict(self) -> dict:
        d = self.to_dict()
        d.pop("keep_last_checkpoints")
        return d

    def fingerprint(self) -> str:
        # Scientific configuration only: a controller bug fix must not make a
        # half-trained arm un-resumable, but any change to what the arm IS must.
        return R.config_fingerprint(self.scientific_dict(), REFERENCE.to_dict(), None)

    def steps_per_epoch(self) -> int:
        return -(-self.train_samples // self.batch_size)

    def validate_frozen(self) -> None:
        """Production refuses any deviation from the frozen design."""
        if not self.production:
            return
        frozen = CE1RunConfig.production_config()
        diffs = {k: (v, getattr(frozen, k)) for k, v in self.scientific_dict().items()
                 if getattr(frozen, k) != (tuple(v) if k == "differential" else v)}
        if diffs:
            raise ControllerError(f"REFUSING PRODUCTION: frozen parameters changed: {diffs}")


# =====================================================================
# Run directory + manifest
# =====================================================================

def seeds_for(config: CE1RunConfig) -> dict:
    from audit.cryptography.experiments.ce1.gohr_signal_destruction import seed_manifest
    return seed_manifest(config.n_blocks, config.seed_base, config.assignment_seed)


def open_run(run_dir, config: CE1RunConfig, *, repo_root=None,
             legacy_block0_decision: str | None = None) -> dict:
    """
    Create or re-open a run directory. A re-opened run must carry the SAME
    configuration fingerprint; anything else is a different experiment and
    is refused rather than mixed.
    """
    config.validate_frozen()
    run_dir = validate_run_dir(run_dir, config, repo_root=repo_root)
    if legacy_block0_decision not in (None, "RECOVER", "RETRAIN"):
        raise ControllerError(f"legacy_block0_decision must be RECOVER or RETRAIN")
    if config.production:
        from audit.cryptography.output_policy import assert_audit_output_path
        assert_audit_output_path(run_dir / RUN_MANIFEST, repo_root=repo_root)
    mpath = run_dir / RUN_MANIFEST
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text())
        except json.JSONDecodeError as exc:
            raise ControllerError(f"{mpath} unreadable: {exc}") from exc
        proto = manifest.get("controller_protocol_version")
        if proto not in COMPATIBLE_CONTROLLER_PROTOCOLS:
            raise R.ResumeRefused(
                f"{run_dir}: run was written by controller protocol {proto!r}; this "
                f"controller accepts {sorted(COMPATIBLE_CONTROLLER_PROTOCOLS)}. Refusing to "
                "reopen it under an incompatible protocol.")
        if manifest.get("block_resolution_rules") != BLOCK_RESOLUTION_RULES:
            raise R.ResumeRefused(
                f"{run_dir}: the run was created under different block-resolution rules; "
                "rules may not change during a run")
        if manifest.get("config_fingerprint") != config.fingerprint():
            raise R.ResumeRefused(
                f"{run_dir}: run was created with configuration fingerprint "
                f"{manifest.get('config_fingerprint')}, current is {config.fingerprint()}. "
                "Refusing to mix experiments.")
        if legacy_block0_decision and manifest.get("legacy_block0_decision") not in (
                None, legacy_block0_decision):
            raise ControllerError(
                f"legacy block0 decision is already recorded as "
                f"{manifest['legacy_block0_decision']!r}; it cannot be changed.")
        if legacy_block0_decision and manifest.get("legacy_block0_decision") is None:
            manifest["legacy_block0_decision"] = legacy_block0_decision
            R.atomic_write_json(mpath, manifest)
        return manifest
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "controller_protocol_version": CONTROLLER_PROTOCOL_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "seed_manifest": seeds_for(config),
        "resume_semantics": RESUME_SEMANTICS,
        "block_resolution_rules": BLOCK_RESOLUTION_RULES,
        "legacy_block0_decision": legacy_block0_decision,
        "created_utc": utc(),
    }
    R.atomic_write_json(mpath, manifest)
    CE1Ledger(run_dir / LEDGER_NAME).record(
        "RUN_START", block_id=None, arm=None, config_hash=config.fingerprint(),
        production=config.production)
    return manifest


def load_run_config(run_dir) -> CE1RunConfig:
    run_dir = _reject_degenerate(run_dir)
    mpath = run_dir / RUN_MANIFEST
    if not mpath.exists():
        raise RunDirectoryError(f"{run_dir.resolve()} is not a CE1 controller run "
                                f"(no {RUN_MANIFEST})")
    manifest = json.loads(mpath.read_text())
    cfg = CE1RunConfig.from_dict(manifest["config"])
    if cfg.fingerprint() != manifest["config_fingerprint"]:
        raise ControllerError("run manifest config does not reproduce its own fingerprint")
    return cfg


def ledger_for(run_dir) -> CE1Ledger:
    return CE1Ledger(Path(run_dir) / LEDGER_NAME)


# =====================================================================
# Sealed evaluation set
# =====================================================================

def _refuse_generation(_n):
    raise ControllerError("the sealed evaluation set may not be generated here")


def ensure_sealed(run_dir, config: CE1RunConfig, *, sealed_source=None,
                  generate_fn=None) -> dict:
    """
    Attach THE sealed set to this run, never a new one in production.

      * run_dir/sealed exists      -> reload + content-hash verify
      * sealed_source given        -> byte copy + content-hash verify
      * non-production + generator -> generate (tests only)
    The hash must equal config.expected_sealed_sha256 when that is set
    (always, in production: the pinned 33df1f95...).
    """
    from audit.cryptography.sealed_dataset import (
        _content_hash, data_path, manifest_path, prepare_sealed_evaluation_set)

    sdir = Path(run_dir) / "sealed"
    want = config.expected_sealed_sha256
    if config.production and want != PINNED_SEALED_EVALUATION_SHA256:
        raise ControllerError("production must pin the existing sealed evaluation set")

    if not manifest_path(sdir).exists() and sealed_source is not None:
        src = Path(sealed_source)
        src_npz = data_path(src) if src.is_dir() else src
        with np.load(src_npz) as z:
            X, Y = z["X"], z["Y"]
        digest = _content_hash(X, Y)
        if want and digest != want:
            raise ControllerError(f"sealed source {src_npz} has hash {digest}, expected {want}")
        sdir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(sdir), suffix=".npz.tmp")
        os.close(fd)
        shutil.copyfile(src_npz, tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, data_path(sdir))
        R.atomic_write_json(manifest_path(sdir), {
            "sha256": digest, "n_samples": int(X.shape[0]), "rounds": config.rounds,
            "differential": list(config.differential), "path": data_path(sdir).name,
            "role": "sealed_evaluation_set", "copied_from": str(src_npz),
            "policy": "byte copy of THE sealed set; never regenerated",
            "reused": True})

    if manifest_path(sdir).exists():
        gen = _refuse_generation
    elif not config.production and generate_fn is not None:
        gen = generate_fn
    else:
        raise ControllerError(
            f"no sealed evaluation set in {sdir} and no verified source supplied. "
            "Production never generates a new sealed set.")
    sealed = prepare_sealed_evaluation_set(
        sdir, generate_fn=gen, rounds=config.rounds, differential=config.differential,
        n_samples=config.evaluation_samples)
    if want and sealed["sha256"] != want:
        raise ControllerError(f"sealed set hash {sealed['sha256']} != pinned {want}")
    return sealed


# =====================================================================
# Block context (replaces data_fn.last_validation)
# =====================================================================

@dataclass
class BlockContext:
    """
    The complete, explicit, persisted data of one CE1 block. Both arms train
    on THIS object: the same X_train array, the same validation split (intact
    labels in both arms, per the frozen protocol note), and are scored on the
    same sealed set. Only the training LABELS differ.
    """
    block_id: str
    permutation_seed: int
    seeds: dict
    X_train: np.ndarray
    Y_train_baseline: np.ndarray
    Y_train_destroyed: np.ndarray
    X_validation: np.ndarray
    Y_validation: np.ndarray
    X_evaluation: np.ndarray
    Y_evaluation: np.ndarray
    hashes: dict
    commit: dict = field(default_factory=dict)

    def training_pair(self, arm: str):
        if arm == "baseline":
            return self.X_train, self.Y_train_baseline
        if arm == "destroyed":
            return self.X_train, self.Y_train_destroyed
        raise ValueError(arm)

    def validation_pair(self, arm: str):
        # identical object for both arms (intact labels) - by construction
        return self.X_validation, self.Y_validation

    def labels_hash(self, arm: str) -> str:
        return self.hashes["baseline_labels" if arm == "baseline" else "destroyed_labels"]

    def verify(self, config: CE1RunConfig) -> dict:
        """Recompute every design invariant from the arrays themselves."""
        problems = []
        n_tr, n_va = config.train_samples, config.validation_samples
        if self.X_train.shape != (n_tr, 64) or self.Y_train_baseline.shape != (n_tr,):
            problems.append(f"training shape {self.X_train.shape}/{self.Y_train_baseline.shape}")
        if self.X_validation.shape != (n_va, 64) or self.Y_validation.shape != (n_va,):
            problems.append(f"validation shape {self.X_validation.shape}")
        for name in ("X_train", "Y_train_baseline", "Y_train_destroyed",
                     "X_validation", "Y_validation"):
            if getattr(self, name).dtype != np.uint8:
                problems.append(f"{name} dtype {getattr(self, name).dtype} != uint8")
        # destroyed labels must be EXACTLY the frozen permutation of the baseline labels
        rebuilt = build_ce1_block(self.X_train, self.Y_train_baseline, self.X_evaluation[:1],
                                  self.Y_evaluation[:1], block_id=self.block_id,
                                  permutation_seed=self.permutation_seed)
        if not np.array_equal(rebuilt.Y_train_destroyed, self.Y_train_destroyed):
            problems.append("destroyed labels are not the frozen permutation "
                            f"(seed {self.permutation_seed}) of the baseline labels")
        if DS.array_hash(self.X_evaluation, self.Y_evaluation) != self.hashes["sealed"]:
            problems.append("evaluation arrays do not match the sealed-set hash")
        # the two arms' training inputs are one array; validation one pair
        bx, _ = self.training_pair("baseline"); dx, _ = self.training_pair("destroyed")
        vb, vd = self.validation_pair("baseline"), self.validation_pair("destroyed")
        if bx is not dx or vb[0] is not vd[0] or vb[1] is not vd[1]:
            problems.append("arms do not share training inputs / validation split")
        if problems:
            raise DS.DatasetStateError(f"{self.block_id}: " + "; ".join(problems))
        return {"block_id": self.block_id, "verified": True,
                "baseline_x_train_equals_destroyed_x_train": True,
                "validation_shared": True, "destroyed_labels_frozen_permutation": True}


def _block_dir(run_dir, block_id) -> Path:
    return Path(run_dir) / "blocks" / block_id


def _data_dir(run_dir, block_id) -> Path:
    return _block_dir(run_dir, block_id) / "data"


def _commit_path(run_dir, block_id) -> Path:
    return _data_dir(run_dir, block_id) / "dataset_commit.json"


def default_generate_fn(config: CE1RunConfig):
    """The real Gohr generator at the reference configuration (explicit, not defaults)."""
    def generate(n_train, n_val):
        from audit.cryptography.gohr.dataset import GohrDataset
        gen = GohrDataset(rounds=config.rounds, differential=tuple(config.differential),
                          train_samples=n_train, validation_samples=n_val)
        b = gen.generate_baseline_dataset()
        return b.train, b.validation
    return generate


def prepare_block(run_dir, config: CE1RunConfig, block_index: int, sealed: dict, *,
                  generate_fn=None, ledger: CE1Ledger | None = None) -> BlockContext:
    """
    Load the block's committed dataset, or create and commit it exactly once.

    Commit protocol: train, validation and destroyed-label partitions are
    persisted (content-hashed), then `dataset_commit.json` is written
    atomically. No arm may start before the commit. A committed dataset that
    is missing or altered is a hard error: training data are os.urandom and
    NOT seed-replayable, so they are never silently regenerated.
    """
    ledger = ledger or ledger_for(run_dir)
    bid = R.block_id_for(block_index)
    seeds = seeds_for(config)["seeds"][bid]
    ddir, cpath = _data_dir(run_dir, bid), _commit_path(run_dir, bid)
    fp = config.fingerprint()

    if not cpath.exists():
        started = [a for a in ARMS if R.load_arm_state(run_dir, bid, a) is not None]
        if started:
            raise DS.DatasetStateError(
                f"{bid}: arm state exists for {started} but the block dataset was never "
                "committed. Refusing to generate data under existing arm state.")
        if ddir.exists() and any(ddir.iterdir()):
            # interrupted preparation BEFORE commit: no arm ever saw these arrays.
            quarantine = _block_dir(run_dir, bid) / f"_uncommitted_{int(datetime.now().timestamp())}"
            shutil.move(str(ddir), str(quarantine))
            ledger.record("DATASET_PREPARE_RESTARTED", block_id=bid, arm=None,
                          config_hash=fp, quarantined_to=quarantine.name,
                          reason="dataset preparation interrupted before commit; "
                                 "no arm had started")
        generate_fn = generate_fn or default_generate_fn(config)
        (X_tr, Y_tr), (X_va, Y_va) = generate_fn(config.train_samples,
                                                 config.validation_samples)
        X_tr, Y_tr = np.asarray(X_tr), np.asarray(Y_tr)
        X_va, Y_va = np.asarray(X_va), np.asarray(Y_va)
        block = build_ce1_block(X_tr, Y_tr, sealed["X"], sealed["Y"], block_id=bid,
                                permutation_seed=seeds["permutation"])
        m_tr = DS.persist_block_dataset(ddir, block_id=bid, role="train", X=X_tr,
                                        Y=block.Y_train_baseline, rounds=config.rounds,
                                        differential=config.differential)
        m_va = DS.persist_block_dataset(ddir, block_id=bid, role="validation", X=X_va,
                                        Y=Y_va, rounds=config.rounds,
                                        differential=config.differential)
        m_de = DS.persist_block_labels(ddir, block_id=bid, role="train_destroyed_labels",
                                       Y=block.Y_train_destroyed, rounds=config.rounds,
                                       differential=config.differential)
        commit = {
            "schema": "ce1-block-dataset-commit-1", "block_id": bid,
            "config_fingerprint": fp, "rounds": config.rounds,
            "differential": list(config.differential),
            "permutation_seed": seeds["permutation"],
            "n_train": config.train_samples, "n_validation": config.validation_samples,
            "train_sha256": m_tr["sha256"], "validation_sha256": m_va["sha256"],
            "destroyed_labels_sha256": m_de["sha256"],
            "x_train_sha256": DS.array_hash(X_tr),
            "baseline_labels_sha256": DS.array_hash(block.Y_train_baseline),
            "sealed_evaluation_sha256": sealed["sha256"],
            "generator": "GohrDataset.generate_baseline_dataset (speck.make_train_data, "
                         "os.urandom) - NOT seed-replayable; these files are authoritative",
            "destroyed_label_rule": "design.build_ce1_block: Y[default_rng(permutation_seed)"
                                    ".permutation(n)]",
            "committed_utc": utc(),
        }
        R.atomic_write_json(cpath, commit)
        ledger.record("DATASET_PREPARED", block_id=bid, arm=None, config_hash=fp,
                      train_sha256=commit["train_sha256"],
                      validation_sha256=commit["validation_sha256"],
                      destroyed_labels_sha256=commit["destroyed_labels_sha256"])
        del X_tr, Y_tr, X_va, Y_va, block      # train ONLY on what was read back from disk

    ctx = load_block_context(run_dir, config, block_index, sealed)
    ledger.record("DATASET_LOADED", block_id=bid, arm=None, config_hash=fp,
                  train_sha256=ctx.hashes["train"])
    return ctx


def load_block_context(run_dir, config: CE1RunConfig, block_index: int,
                       sealed: dict) -> BlockContext:
    """Reload a committed block dataset and re-verify EVERYTHING before use."""
    bid = R.block_id_for(block_index)
    seeds = seeds_for(config)["seeds"][bid]
    cpath = _commit_path(run_dir, bid)
    if not cpath.exists():
        raise DS.DatasetStateError(f"{bid}: no committed dataset")
    try:
        commit = json.loads(cpath.read_text())
    except json.JSONDecodeError as exc:
        raise DS.DatasetStateError(f"{bid}: dataset commit unreadable: {exc}") from exc
    checks = {"block_id": bid, "config_fingerprint": config.fingerprint(),
              "rounds": config.rounds, "differential": list(config.differential),
              "permutation_seed": seeds["permutation"], "n_train": config.train_samples,
              "n_validation": config.validation_samples,
              "sealed_evaluation_sha256": sealed["sha256"]}
    bad = {k: (commit.get(k), v) for k, v in checks.items() if commit.get(k) != v}
    if bad:
        raise DS.DatasetStateError(f"{bid}: dataset commit does not match this run: {bad}")
    ddir = _data_dir(run_dir, bid)
    X_tr, Y_b, _ = DS.load_block_dataset(ddir, block_id=bid, role="train",
                                         expected_sha256=commit["train_sha256"],
                                         rounds=config.rounds,
                                         differential=config.differential)
    X_va, Y_va, _ = DS.load_block_dataset(ddir, block_id=bid, role="validation",
                                          expected_sha256=commit["validation_sha256"],
                                          rounds=config.rounds,
                                          differential=config.differential)
    Y_d, _ = DS.load_block_labels(ddir, block_id=bid, role="train_destroyed_labels",
                                  expected_sha256=commit["destroyed_labels_sha256"])
    x_hash = DS.array_hash(X_tr)
    if x_hash != commit["x_train_sha256"]:
        raise DS.DatasetStateError(f"{bid}: X_train hash mismatch")
    ctx = BlockContext(
        block_id=bid, permutation_seed=seeds["permutation"], seeds=seeds,
        X_train=X_tr, Y_train_baseline=Y_b, Y_train_destroyed=Y_d,
        X_validation=X_va, Y_validation=Y_va,
        X_evaluation=sealed["X"], Y_evaluation=sealed["Y"],
        hashes={"train": commit["train_sha256"], "validation": commit["validation_sha256"],
                "x_train": x_hash, "baseline_labels": commit["baseline_labels_sha256"],
                "destroyed_labels": commit["destroyed_labels_sha256"],
                "sealed": sealed["sha256"]},
        commit=commit)
    ctx.verify(config)
    return ctx


# =====================================================================
# Arm state + decisions
# =====================================================================

def _expected(config: CE1RunConfig, seeds: dict, arm: str, sealed_sha: str) -> dict:
    return {"config_fingerprint": config.fingerprint(),
            "design_version": config.design_version,
            "assignment_seed": config.assignment_seed,
            "assignment_flip": seeds["assignment_flip"],
            "model_seed": seeds[arm], "permutation_seed": seeds["permutation"],
            "sealed_evaluation_sha256": sealed_sha,
            "terminal_epoch": config.terminal_epoch}


def _identity_problems(state: R.ArmState, expected: dict) -> list:
    return [f"{k}: artifact {getattr(state, k)!r} != current {v!r}"
            for k, v in expected.items() if v is not None and getattr(state, k) != v]


def new_arm_state(config: CE1RunConfig, ctx: BlockContext, arm: str) -> R.ArmState:
    return R.ArmState(
        experiment_id=EXPERIMENT_ID, block_id=ctx.block_id, arm=arm,
        status=R.NOT_STARTED, last_completed_epoch=0,
        terminal_epoch=config.terminal_epoch, model_seed=ctx.seeds[arm],
        assignment_seed=config.assignment_seed,
        assignment_flip=ctx.seeds["assignment_flip"],
        permutation_seed=ctx.permutation_seed, config_fingerprint=config.fingerprint(),
        design_version=config.design_version,
        train_data_sha256=ctx.hashes["x_train"],
        validation_data_sha256=ctx.hashes["validation"],
        sealed_evaluation_sha256=ctx.hashes["sealed"],
        destroyed_labels_sha256=ctx.hashes["destroyed_labels"],
        train_labels_sha256=ctx.labels_hash(arm))


def _sealed_dict(run_dir, config, sealed) -> dict:
    """Accept the sealed dict, or its sha (then the verified set is loaded)."""
    if isinstance(sealed, dict):
        return sealed
    full = ensure_sealed(run_dir, config)
    if full["sha256"] != sealed:
        raise ControllerError(f"sealed hash {sealed} is not this run's sealed set")
    return full


def _caps(st: R.ArmState) -> tuple:
    g = 1 + st.operator_retry_grants
    return MAX_ARM_FAILURES * g, MAX_ARM_RERUNS * g


def decide_arm_detail(run_dir, config: CE1RunConfig, block_id: str, arm: str,
                      sealed) -> dict:
    """
    Pure inspection -> {"action", "reasons", "code"}. Never writes. Uses only
    artifact/state validity and failure STATUS (never an evaluation result).

        no state / epoch 0           -> TRAIN
        EVALUATED, fully verified    -> SKIP
        EVALUATED, record/predictions/
          ledger binding broken      -> PAUSE E-EVALUATION-INTEGRITY
        EVALUATED, model damaged     -> PAUSE E-TERMINAL-ARTIFACT (never retrained)
        TRAINING_COMPLETE, intact    -> EVALUATE (no retraining)
        TRAINING_COMPLETE, damaged   -> RERUN (controller; nothing evaluated yet)
                                        PAUSE E-TERMINAL-ARTIFACT (legacy: no data)
        IN_PROGRESS/FAILED, valid    -> RESUME at last_completed_epoch + 1
        IN_PROGRESS/FAILED, bad      -> RERUN
        identity mismatch            -> PAUSE E-IDENTITY
        state unreadable             -> PAUSE E-STATE-UNREADABLE
        retry cap reached            -> PAUSE E-RETRY-CAP   (operational, NOT a failure)
    """
    def out(action, reasons, code=None):
        return {"action": action, "reasons": list(reasons), "code": code}

    sealed = _sealed_dict(run_dir, config, sealed)
    seeds = seeds_for(config)["seeds"][block_id]
    try:
        st = R.load_arm_state(run_dir, block_id, arm)
    except R.ResumeRefused as exc:
        return out(PAUSE, [str(exc)], "E-STATE-UNREADABLE")
    if st is None:
        return out(TRAIN, ["no state"])
    exp = _expected(config, seeds, arm, sealed["sha256"])
    ident = _identity_problems(st, exp)
    if ident:
        return out(PAUSE, ident, "E-IDENTITY")
    if st.status == R.EVALUATED:
        term = R.validate_terminal_artifact(run_dir, st)
        if term:
            return out(PAUSE, term + ["an EVALUATED arm is never retrained"],
                       "E-TERMINAL-ARTIFACT")
        ev = verify_evaluation(run_dir, config, st, sealed)
        return out(PAUSE, ev, "E-EVALUATION-INTEGRITY") if ev else out(SKIP, ["EVALUATED"])
    fail_cap, rerun_cap = _caps(st)
    if st.failure_count >= fail_cap:
        return out(PAUSE, [f"{st.failure_count} recorded exceptions reached the operational "
                           f"cap {fail_cap}; operator GRANT_RETRY required"], "E-RETRY-CAP")
    if st.status == R.TRAINING_COMPLETE:
        term = R.validate_terminal_artifact(run_dir, st)
        if not term:
            return out(EVALUATE, ["TRAINING_COMPLETE"])
        if st.provenance != "controller":
            return out(PAUSE, term + ["legacy arm cannot be retrained"], "E-TERMINAL-ARTIFACT")
        problems = term
    elif st.last_completed_epoch == 0:
        return out(TRAIN, [f"status {st.status}, no completed epoch"])
    else:
        problems = R.validate_arm_for_resume(run_dir, st, expected=exp)
        if not problems:
            return out(RESUME, [f"resume at epoch {st.next_epoch}"])
    if st.rerun_count >= rerun_cap:
        return out(PAUSE, problems + [f"rerun cap {rerun_cap} reached; operator GRANT_RETRY "
                                      "required"], "E-RETRY-CAP")
    return out(RERUN, problems)


def decide_arm(run_dir, config: CE1RunConfig, block_id: str, arm: str, sealed) -> tuple:
    d = decide_arm_detail(run_dir, config, block_id, arm, sealed)
    return d["action"], d["reasons"]


# =====================================================================
# Training (real GohrTrainer, real model.fit, real per-epoch callback)
# =====================================================================

def _make_epoch_callback(run_dir, state_box: dict, ledger: CE1Ledger, lr_fn,
                         keep_last: int, config_hash: str):
    import keras

    class CE1EpochCheckpoint(keras.callbacks.Callback):
        """
        Durable per-epoch completion. Keras passes the 0-based epoch index;
        after index i the number of COMPLETED epochs is i + 1. Artifact
        (weights + optimizer) is written and hashed first, then the state
        marker is atomically replaced, then the ledger line is appended.
        """

        def on_epoch_end(self, epoch, logs=None):
            completed = epoch + 1
            st = state_box["state"]
            if completed != st.last_completed_epoch + 1:
                raise ControllerError(
                    f"{st.block_id}/{st.arm}: epoch sequence broken "
                    f"({st.last_completed_epoch} -> {completed})")
            applied = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
            scheduled = float(lr_fn(epoch))
            if abs(applied - scheduled) > 1e-6 * abs(scheduled):
                raise ControllerError(f"epoch {completed}: applied LR {applied} != frozen "
                                      f"schedule {scheduled}")
            st = R.record_completed_epoch(run_dir, st, epoch=completed, model=self.model,
                                          keep_last=keep_last, logs=logs or {},
                                          learning_rate=applied)
            state_box["state"] = st
            ledger.record("CHECKPOINT", block_id=st.block_id, arm=st.arm,
                          config_hash=config_hash, epoch=completed,
                          model_sha256=st.model_sha256, status=st.status)

    return CE1EpochCheckpoint()


def keras_file_optimizer_iterations(path) -> int | None:
    """
    Read the optimizer step counter straight from a Keras-3 `.keras` file with
    h5py (no TensorFlow import - safe in the TF-free 2-GPU parent). Keras 3
    stores it as optimizer/vars/0 inside model.weights.h5.
    """
    import io
    import zipfile

    import h5py
    try:
        with zipfile.ZipFile(path) as z:
            if "model.weights.h5" not in z.namelist():
                return None
            with h5py.File(io.BytesIO(z.read("model.weights.h5")), "r") as h:
                return int(h["optimizer"]["vars"]["0"][()])
    except (zipfile.BadZipFile, OSError, KeyError, TypeError, ValueError):
        return None                                   # unknown -> never counted as evidence


def build_model(config: CE1RunConfig):
    from audit.cryptography.gohr.model import GohrModel
    return GohrModel(depth=config.depth, regularization=config.l2_reg,
                     optimizer=config.optimizer, loss=config.loss).build()


def _archive_arm(run_dir, block_id, arm) -> str | None:
    d = R.arm_dir(run_dir, block_id, arm)
    if not d.exists():
        return None
    dest = _block_dir(run_dir, block_id) / "_rerun_archive" / f"{arm}_{int(datetime.now().timestamp())}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest))
    return str(dest.relative_to(Path(run_dir)))


def train_arm(run_dir, config: CE1RunConfig, ctx: BlockContext, arm: str, action: str, *,
              ledger: CE1Ledger | None = None, extra_callbacks=None) -> R.ArmState:
    """Execute TRAIN / RESUME / RERUN for one arm through the real GohrTrainer."""
    import keras

    from audit.cryptography.gohr.trainer import GohrTrainer

    ledger = ledger or ledger_for(run_dir)
    fp = config.fingerprint()
    adir = R.arm_dir(run_dir, ctx.block_id, arm)
    prior = R.load_arm_state(run_dir, ctx.block_id, arm)

    if action == RESUME:
        st = prior
        path = R.resolve_artifact(run_dir, st.model_artifact)
        if R.sha256_file(path) != st.model_sha256:
            raise R.ResumeRefused(f"checkpoint hash changed since recorded: {path}")
        model = keras.models.load_model(path)          # weights + optimizer state
        iters = int(model.optimizer.iterations.numpy())
        want = st.last_completed_epoch * config.steps_per_epoch()
        if iters != want:
            raise R.ResumeRefused(
                f"{ctx.block_id}/{arm}: restored optimizer has {iters} iterations, expected "
                f"{want} for {st.last_completed_epoch} completed epochs")
        initial_epoch = st.last_completed_epoch
        ledger.record_resume(block_id=ctx.block_id, arm=arm,
                             previous_config_hash=st.config_fingerprint,
                             resumed_config_hash=fp, from_epoch=initial_epoch + 1)
        st = R.mark_resumed(st)
        st.resume_count += 1
        ledger.record("RESUME", block_id=ctx.block_id, arm=arm, config_hash=fp,
                      initial_epoch=initial_epoch, next_epoch=initial_epoch + 1,
                      optimizer_iterations=iters, shuffle_stream_continuity=False)
    else:
        rerun_count = 0
        if action == RERUN:
            archived = _archive_arm(run_dir, ctx.block_id, arm)
            rerun_count = (prior.rerun_count + 1) if prior else 1
            ledger.record("RERUN", block_id=ctx.block_id, arm=arm, config_hash=fp,
                          archived_to=archived,
                          previous_epoch=prior.last_completed_epoch if prior else None)
        st = new_arm_state(config, ctx, arm)
        st.rerun_count = rerun_count
        st.failure_count = prior.failure_count if prior else 0
        # Seed BEFORE construction so the declared seed governs initialisation.
        GohrTrainer.set_seed(st.model_seed)
        model = build_model(config)
        initial_epoch = 0

    # dataset binding: the arm trains on the committed arrays, nothing else
    if st.train_data_sha256 != ctx.hashes["x_train"] or \
            st.validation_data_sha256 != ctx.hashes["validation"] or \
            st.train_labels_sha256 != ctx.labels_hash(arm):
        raise R.ResumeRefused(f"{ctx.block_id}/{arm}: arm state is bound to different data")

    st.status = R.IN_PROGRESS
    R.save_arm_state(run_dir, st)
    if action != RESUME:
        ledger.record("TRAIN_START", block_id=ctx.block_id, arm=arm, config_hash=fp,
                      model_seed=st.model_seed, initial_epoch=0,
                      train_sha256=ctx.hashes["train"], labels_sha256=st.train_labels_sha256)

    trainer = GohrTrainer(batch_size=config.batch_size, epochs=config.terminal_epoch,
                          checkpoint_dir=adir, save_best_only=config.save_best_only_debug,
                          high_learning_rate=config.high_learning_rate,
                          low_learning_rate=config.low_learning_rate,
                          lr_cycle_length=config.lr_cycle_length)
    box = {"state": st}
    cb = _make_epoch_callback(run_dir, box, ledger, trainer.learning_rate,
                              config.keep_last_checkpoints, fp)
    try:
        trainer.train(model, ctx.training_pair(arm), ctx.validation_pair(arm),
                      seed=st.model_seed, checkpoint_name="bestval_DEBUG_ONLY.keras",
                      initial_epoch=initial_epoch, terminal_epoch=config.terminal_epoch,
                      extra_callbacks=[cb] + list(extra_callbacks or []),
                      initial_best_val_loss=st.best_val_loss)
    except Exception as exc:
        # a FAILURE: counts toward the predeclared cap; completed epochs are kept
        R.record_arm_failure(run_dir, box["state"], exc)
        ledger.record("FAILURE", block_id=ctx.block_id, arm=arm, config_hash=fp,
                      phase="training", reason=type(exc).__name__, detail=str(exc)[:500],
                      failed_after_epoch=box["state"].last_completed_epoch,
                      failure_count=box["state"].failure_count)
        raise
    except BaseException as exc:
        # an INTERRUPTION (KeyboardInterrupt / SystemExit): never a failure; the
        # durable state already says IN_PROGRESS at the last completed epoch.
        ledger.record("INTERRUPTED", block_id=ctx.block_id, arm=arm, config_hash=fp,
                      reason=type(exc).__name__,
                      last_completed_epoch=box["state"].last_completed_epoch)
        raise
    st = box["state"]
    if st.status != R.TRAINING_COMPLETE or st.last_completed_epoch != config.terminal_epoch:
        raise ControllerError(f"{ctx.block_id}/{arm}: fit returned at epoch "
                              f"{st.last_completed_epoch}, status {st.status}")
    # Terminal (FINAL_EPOCH) model = the epoch-200 checkpoint, byte for byte.
    src = R.resolve_artifact(run_dir, st.model_artifact)
    final = adir / "FINAL_EPOCH.keras"
    fd, tmp = tempfile.mkstemp(dir=str(adir), suffix=".partial.keras")
    os.close(fd)
    shutil.copyfile(src, tmp)
    os.replace(tmp, final)
    st.terminal_model_artifact = R.rel_to_run(run_dir, final)
    st.terminal_model_sha256 = R.sha256_file(final)
    if st.terminal_model_sha256 != st.model_sha256:
        raise ControllerError("terminal copy differs from the epoch checkpoint")
    st.terminal_optimizer_iterations = keras_file_optimizer_iterations(final)
    if st.terminal_optimizer_iterations != config.terminal_epoch * config.steps_per_epoch():
        raise ControllerError(
            f"terminal model took {st.terminal_optimizer_iterations} optimizer steps, expected "
            f"{config.terminal_epoch} x {config.steps_per_epoch()}")
    R.save_arm_state(run_dir, st)
    ledger.record("TRAINING_COMPLETE", block_id=ctx.block_id, arm=arm, config_hash=fp,
                  terminal_epoch=st.last_completed_epoch,
                  terminal_model_sha256=st.terminal_model_sha256,
                  resumed=not st.shuffle_stream_continuity)
    return st


# =====================================================================
# Evaluation (terminal model on the sealed set; no selection)
# =====================================================================

EVALUATION_SCHEMA = "ce1-evaluation-2"


def canonical_sha256(record: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def persist_evaluation(run_dir, config: CE1RunConfig, st: R.ArmState, preds, sealed: dict, *,
                       evaluator_accuracy: float | None, ledger: CE1Ledger,
                       tool_versions: dict | None = None) -> R.ArmState:
    """
    Persist one arm's sealed-set evaluation and bind it three ways:
    predictions file (sha256) <- evaluation.json (canonical sha256) <- ledger
    EVALUATED event. The reported accuracy is EXACTLY n_correct / n_samples
    with Keras's binary-accuracy rule (pred > 0.5); the Keras evaluator's
    float32 figure is kept only as a cross-check.

    Order: predictions -> evaluation.json -> ledger -> state. A crash before the
    ledger line leaves the arm TRAINING_COMPLETE (re-evaluated next pass); a crash
    after it leaves a ledger record that the next evaluation must agree with.
    """
    Y = sealed["Y"]
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    if preds.shape[0] != Y.shape[0]:
        raise ControllerError(f"{preds.shape[0]} predictions for {Y.shape[0]} sealed samples")
    n = int(Y.shape[0])
    n_correct = int(np.sum((preds > 0.5).astype(Y.dtype) == Y))
    accuracy = n_correct / n
    if evaluator_accuracy is not None and abs(evaluator_accuracy - accuracy) > 1e-6:
        raise ControllerError(f"Keras evaluator accuracy {evaluator_accuracy} disagrees with "
                              f"the prediction count {n_correct}/{n}")
    adir = R.arm_dir(run_dir, st.block_id, st.arm)
    pred_path = adir / "sealed_predictions.npz"
    DS._atomic_savez(pred_path, predictions=preds)
    record = {
        "schema": EVALUATION_SCHEMA, "block_id": st.block_id, "arm": st.arm,
        "accuracy": accuracy, "n_correct": n_correct, "n_samples": n,
        "evaluator_accuracy": evaluator_accuracy,
        "accuracy_rule": "n_correct / n_samples, prediction counted as 1 iff pred > 0.5 "
                         "(identical to keras binary_accuracy)",
        "model_sha256": st.terminal_model_sha256,
        "model_artifact": st.terminal_model_artifact,
        "sealed_evaluation_sha256": sealed["sha256"],
        "predictions_file": R.rel_to_run(run_dir, pred_path),
        "predictions_sha256": R.sha256_file(pred_path), "predictions_count": n,
        "checkpoint_rule": "FINAL_EPOCH", "selection": "none",
        "train_data_sha256": st.train_data_sha256,
        "validation_data_sha256": st.validation_data_sha256,
        "config_fingerprint": config.fingerprint(), "provenance": st.provenance,
        **(tool_versions or {}), "evaluated_utc": utc(),
    }
    ev_sha = canonical_sha256(record)
    R.atomic_write_json(adir / "evaluation.json", record)
    ledger.record("EVALUATED", block_id=st.block_id, arm=st.arm,
                  config_hash=config.fingerprint(), evaluation_sha256=ev_sha,
                  accuracy=accuracy, n_correct=n_correct, n_samples=n,
                  predictions_sha256=record["predictions_sha256"],
                  model_sha256=st.terminal_model_sha256,
                  sealed_evaluation_sha256=sealed["sha256"])
    st.evaluation, st.evaluation_sha256, st.status = record, ev_sha, R.EVALUATED
    R.save_arm_state(run_dir, st)
    return st


def verify_evaluation(run_dir, config: CE1RunConfig, st: R.ArmState, sealed: dict,
                      ledger_events: list | None = None) -> list:
    """
    Full integrity check of an EVALUATED arm. Returns problems (empty = verified).
    Nothing here compares accuracy with anything but its own bound evidence.
    """
    problems = []
    adir = R.arm_dir(run_dir, st.block_id, st.arm)
    epath = adir / "evaluation.json"
    if not epath.exists():
        return ["evaluation.json missing"]
    try:
        rec = json.loads(epath.read_text())
    except json.JSONDecodeError as exc:
        return [f"evaluation.json unreadable: {exc}"]
    rec_sha = canonical_sha256(rec)
    if st.evaluation is None or canonical_sha256(st.evaluation) != rec_sha:
        problems.append("evaluation.json differs from the evaluation bound in the arm state")
    if st.evaluation_sha256 != rec_sha:
        problems.append("evaluation.json hash != evaluation_sha256 recorded in the arm state")
    if rec.get("schema") != EVALUATION_SCHEMA:
        problems.append(f"evaluation schema {rec.get('schema')!r}")
    if rec.get("block_id") != st.block_id or rec.get("arm") != st.arm:
        problems.append("evaluation record belongs to another arm")
    if rec.get("model_sha256") != st.terminal_model_sha256:
        problems.append("evaluated model hash != terminal model hash")
    if rec.get("sealed_evaluation_sha256") != sealed["sha256"]:
        problems.append("evaluation used a different sealed set")
    if config.expected_sealed_sha256 and \
            rec.get("sealed_evaluation_sha256") != config.expected_sealed_sha256:
        problems.append("evaluation sealed set != pinned sealed set")
    if rec.get("config_fingerprint") != config.fingerprint():
        problems.append("evaluation config fingerprint != this run")
    Y = sealed["Y"]
    try:
        ppath = R.resolve_artifact(run_dir, rec.get("predictions_file"))
    except R.ResumeRefused as exc:
        return problems + [str(exc)]
    if ppath is None or not ppath.exists():
        return problems + ["prediction file missing"]
    if R.sha256_file(ppath) != rec.get("predictions_sha256"):
        return problems + ["prediction file hash != recorded predictions_sha256"]
    with np.load(ppath) as z:
        preds = z["predictions"].reshape(-1)
    if preds.shape[0] != Y.shape[0] or rec.get("predictions_count") != Y.shape[0] \
            or rec.get("n_samples") != Y.shape[0]:
        problems.append(f"prediction count {preds.shape[0]} / recorded "
                        f"{rec.get('n_samples')} != sealed count {Y.shape[0]}")
        return problems
    n_correct = int(np.sum((preds > 0.5).astype(Y.dtype) == Y))
    if rec.get("n_correct") != n_correct:
        problems.append(f"recorded n_correct {rec.get('n_correct')} != recomputed {n_correct}")
    if rec.get("accuracy") != n_correct / Y.shape[0]:
        problems.append("recorded accuracy != n_correct / n_samples")
    # ledger binding: the latest EVALUATED event must carry THIS record, and every
    # evaluation ever logged for this terminal model must agree on n_correct.
    evs = [e for e in (ledger_events if ledger_events is not None
                       else ledger_for(run_dir).events())
           if e.get("event") == "EVALUATED" and e.get("block_id") == st.block_id
           and e.get("arm") == st.arm]
    if not evs:
        problems.append("no EVALUATED ledger event for this arm")
    else:
        last = evs[-1]
        for k, v in (("evaluation_sha256", rec_sha), ("n_correct", rec.get("n_correct")),
                     ("accuracy", rec.get("accuracy")), ("model_sha256", rec.get("model_sha256")),
                     ("predictions_sha256", rec.get("predictions_sha256")),
                     ("sealed_evaluation_sha256", rec.get("sealed_evaluation_sha256")),
                     ("config_hash", rec.get("config_fingerprint"))):
            if last.get(k) != v:
                problems.append(f"ledger EVALUATED {k} != evaluation.json")
        same_model = {e.get("n_correct") for e in evs
                      if e.get("model_sha256") == st.terminal_model_sha256}
        if len(same_model) > 1:
            problems.append(f"conflicting ledger evaluations of the same model: "
                            f"n_correct values {sorted(same_model)} (methodology 3.8.8: "
                            "unresolved conflict)")
    return problems


def evaluate_arm(run_dir, config: CE1RunConfig, block_id: str, arm: str, sealed: dict, *,
                 ledger: CE1Ledger | None = None) -> R.ArmState:
    import keras

    from audit.cryptography.gohr.evaluate import GohrEvaluator

    ledger = ledger or ledger_for(run_dir)
    fp = config.fingerprint()
    st = R.load_arm_state(run_dir, block_id, arm)
    if st.status not in (R.TRAINING_COMPLETE, R.EVALUATED):
        raise ControllerError(f"{block_id}/{arm}: cannot evaluate status {st.status}")
    problems = R.validate_terminal_artifact(run_dir, st)
    if problems:
        raise R.ResumeRefused(f"{block_id}/{arm}: " + "; ".join(problems))
    if st.sealed_evaluation_sha256 != sealed["sha256"]:
        raise R.ResumeRefused(f"{block_id}/{arm}: bound to a different sealed set")
    ledger.record("EVALUATION", block_id=block_id, arm=arm, config_hash=fp,
                  terminal_model_sha256=st.terminal_model_sha256,
                  sealed_evaluation_sha256=sealed["sha256"])
    path = R.resolve_artifact(run_dir, st.terminal_model_artifact)
    try:
        model = keras.models.load_model(path)
        X, Y = sealed["X"], sealed["Y"]
        acc = float(GohrEvaluator(batch_size=config.batch_size).evaluate(model, (X, Y)))
        preds = model.predict(X, batch_size=config.batch_size, verbose=0)
        import tensorflow as tf
        return persist_evaluation(run_dir, config, st, preds, sealed,
                                  evaluator_accuracy=acc, ledger=ledger,
                                  tool_versions={"tensorflow": tf.__version__,
                                                 "keras": keras.__version__})
    except Exception as exc:
        st = R.load_arm_state(run_dir, block_id, arm)
        st.failure_count += 1
        R.save_arm_state(run_dir, st)            # status unchanged
        ledger.record("FAILURE", block_id=block_id, arm=arm, config_hash=fp,
                      phase="evaluation", reason=type(exc).__name__, detail=str(exc)[:500],
                      failure_count=st.failure_count)
        raise


# =====================================================================
# Block resolution (evidence gate)
# =====================================================================

def _commit_consistent(run_dir, block_id) -> tuple:
    """Cheap check: commit + every partition manifest + data file present and agreeing."""
    cpath = _commit_path(run_dir, block_id)
    if not cpath.exists():
        return False, "no dataset commit"
    try:
        commit = json.loads(cpath.read_text())
        ddir = _data_dir(run_dir, block_id)
        for role, key in (("train", "train_sha256"), ("validation", "validation_sha256"),
                          ("train_destroyed_labels", "destroyed_labels_sha256")):
            m = json.loads((ddir / f"{block_id}_{role}_manifest.json").read_text())
            if m["sha256"] != commit[key]:
                return False, f"{role} manifest hash != commit"
            if not DS._resolve_data_path(ddir, m).exists():
                return False, f"{role} data file missing"
    except (OSError, KeyError, json.JSONDecodeError, DS.DatasetStateError) as exc:
        return False, f"dataset commit unusable: {exc}"
    return True, commit


INVARIANTS = ("evaluation_bound", "terminal_epoch", "assignment",
              "shared_training_inputs", "label_construction", "shared_validation")
_NOT_COUNTED_CODE = {"terminal_epoch": "N-FINAL-EPOCH", "shared_training_inputs": "N-PAIRING",
                     "label_construction": "N-PAIRING", "shared_validation": "N-PAIRING"}
UNRESOLVED = ("PENDING", "NEEDS_OPERATOR", "LEGACY_RECOVERED_PENDING_EVALUATION")


def block_validity_gate(run_dir, config: CE1RunConfig, block_index: int, sealed, *,
                        final: bool = False, deep: bool = False) -> dict:
    """
    Derive a block's resolution from EVIDENCE ONLY (BLOCK_RESOLUTION_RULES):

      VALID                                counted
      NOT_COUNTED (final gate only)        permanent evidence verdict
      LEGACY_RECOVERED_PENDING_EVALUATION  unresolved
      LEGACY_RECOVERED_EVALUATED           awaiting the final gate
      NEEDS_OPERATOR                       unresolved (engineering; never a failure)
      PENDING                              unresolved

    Accuracies are read only to verify that they are bound to their evidence.
    """
    sealed = _sealed_dict(run_dir, config, sealed)
    bid = R.block_id_for(block_index)

    def needs_operator(code, reasons):
        return {"block_id": bid, "status": "NEEDS_OPERATOR", "code": code, "reasons": reasons}

    details = {arm: decide_arm_detail(run_dir, config, bid, arm, sealed) for arm in ARMS}
    for arm, d in details.items():
        if d["action"] == PAUSE:
            return needs_operator(d["code"], [f"{arm}: {r}" for r in d["reasons"]])
    states = {arm: R.load_arm_state(run_dir, bid, arm) for arm in ARMS}
    provs = {s.provenance for s in states.values() if s is not None}
    legacy = bool(provs) and provs != {"controller"}
    if legacy and (len(provs) > 1 or None in states.values()):
        return needs_operator("E-PAIRING-GUARD",
                              ["block mixes legacy-recovered and controller arms"])

    commit = None
    if not legacy and any(s is not None for s in states.values()):
        ok, commit = _commit_consistent(run_dir, bid)
        if not ok:
            return needs_operator("E-DATASET", [commit + " - restore the committed files; "
                                                "training data are not replayable"])
        if deep:
            try:
                load_block_context(run_dir, config, block_index, sealed)
            except DS.DatasetStateError as exc:
                return needs_operator("E-DATASET", [str(exc)])

    if not all(s is not None and s.status == R.EVALUATED for s in states.values()):
        arms = {a: (s.status if s else R.NOT_STARTED) for a, s in states.items()}
        if legacy:
            return {"block_id": bid, "status": "LEGACY_RECOVERED_PENDING_EVALUATION",
                    "arms": arms}
        return {"block_id": bid, "status": "PENDING", "arms": arms}

    b, d = states["baseline"], states["destroyed"]
    seeds = seeds_for(config)["seeds"][bid]
    ev = {}
    events = ledger_for(run_dir).events()
    ev["evaluation_bound"] = all(
        not R.validate_terminal_artifact(run_dir, s)
        and not verify_evaluation(run_dir, config, s, sealed, events)
        for s in states.values())
    want_iters = config.terminal_epoch * config.steps_per_epoch()
    iters = {}
    for arm, s in states.items():
        iters[arm] = keras_file_optimizer_iterations(
            R.resolve_artifact(run_dir, s.terminal_model_artifact))
    term_ok = all(v == want_iters for v in iters.values())
    if not legacy:
        want = list(range(1, config.terminal_epoch + 1))
        term_ok = term_ok and all([h["epoch"] for h in s.history] == want
                                  for s in states.values())
    ev["terminal_epoch"] = term_ok
    ev["assignment"] = all(not _identity_problems(s, _expected(config, seeds, a,
                                                               sealed["sha256"]))
                           for a, s in states.items())
    if legacy:
        # training/validation data were never persisted: permanently UNVERIFIABLE
        ev["shared_training_inputs"] = None
        ev["label_construction"] = None
        ev["shared_validation"] = None
    else:
        ev["shared_training_inputs"] = (b.train_data_sha256 == d.train_data_sha256
                                        == commit["x_train_sha256"])
        ev["label_construction"] = (
            b.train_labels_sha256 == commit["baseline_labels_sha256"]
            and d.train_labels_sha256 == d.destroyed_labels_sha256
            == commit["destroyed_labels_sha256"]
            and b.train_labels_sha256 != d.train_labels_sha256)
        ev["shared_validation"] = (b.validation_data_sha256 == d.validation_data_sha256
                                   == commit["validation_sha256"])
    if ev["evaluation_bound"] is False or ev["assignment"] is False:
        return needs_operator("E-EVALUATION-INTEGRITY", ["evidence binding failed"])

    base = {"block_id": bid, "evidence": ev, "optimizer_iterations": iters,
            "sealed_evaluation_sha256": sealed["sha256"],
            "assignment_flip": seeds["assignment_flip"],
            "model_seeds": {"baseline": seeds["baseline"], "destroyed": seeds["destroyed"]},
            "provenance": b.provenance,
            "arms": {a: {"model_sha256": s.terminal_model_sha256,
                         "evaluation_sha256": s.evaluation_sha256,
                         "resumed": not s.shuffle_stream_continuity,
                         "provenance": s.provenance} for a, s in states.items()}}
    violated = [k for k in INVARIANTS if ev[k] is False]
    unverifiable = [k for k in INVARIANTS if ev[k] is None]
    if (violated or unverifiable) and not final:
        status = "LEGACY_RECOVERED_EVALUATED" if legacy and not violated else "PENDING_FINAL_GATE"
        return {**base, "status": status, "note": "resolved only by the final CE1 gate"}
    if violated:
        return {**base, "status": "NOT_COUNTED", "code": _NOT_COUNTED_CODE[violated[0]],
                "counted": False, "reasons": [f"invariant {k} violated" for k in violated],
                "descriptive_accuracies": {a: s.evaluation["accuracy"]
                                           for a, s in states.items()}}
    if unverifiable:
        return {**base, "status": "NOT_COUNTED", "code": "N-INSUFFICIENT-EVIDENCE",
                "counted": False,
                "reasons": [f"invariant {k} is permanently unverifiable" for k in unverifiable],
                "descriptive_accuracies": {a: s.evaluation["accuracy"]
                                           for a, s in states.items()}}
    return {**base, "status": "VALID", "counted": True,
            "baseline_accuracy": b.evaluation["accuracy"],
            "destroyed_accuracy": d.evaluation["accuracy"],
            "delta": b.evaluation["accuracy"] - d.evaluation["accuracy"],
            "train_data_sha256": b.train_data_sha256,
            "validation_data_sha256": b.validation_data_sha256}


def record_block_resolution(run_dir, config, block_index, sealed, *, final=False,
                            deep=False, ledger=None) -> dict:
    """Snapshot the derived resolution; ledger only on a change of status."""
    res = block_validity_gate(run_dir, config, block_index, sealed, final=final, deep=deep)
    path = _block_dir(run_dir, res["block_id"]) / "block_resolution.json"
    prev = json.loads(path.read_text()).get("status") if path.exists() else None
    if res["status"] != "PENDING" or path.exists():
        R.atomic_write_json(path, {**res, "rules_version": BLOCK_RESOLUTION_RULES["version"],
                                   "final_gate": final, "utc": utc()})
    if res["status"] != prev and res["status"] != "PENDING":
        event = {"VALID": "BLOCK_VALID", "NOT_COUNTED": "BLOCK_NOT_COUNTED",
                 "LEGACY_RECOVERED_EVALUATED": "BLOCK_LEGACY_EVALUATED",
                 "NEEDS_OPERATOR": "PAUSED"}.get(res["status"])
        if event:
            (ledger or ledger_for(run_dir)).record(
                event, block_id=res["block_id"], arm=None, config_hash=config.fingerprint(),
                code=res.get("code"), final_gate=final)
    return res


def plan_block(run_dir, config: CE1RunConfig, block_index: int, sealed) -> dict:
    """Inspect a block and return per-arm actions (no side effects)."""
    sealed = _sealed_dict(run_dir, config, sealed)
    bid = R.block_id_for(block_index)
    arms = {}
    for arm in ARMS:
        d = decide_arm_detail(run_dir, config, bid, arm, sealed)
        arms[arm] = {"action": d["action"], "reasons": d["reasons"], "code": d["code"]}
    # A training action is only legal if the sibling was trained on the SAME
    # committed dataset (or has not started). Legacy-recovered arms carry no
    # dataset, so their sibling can never be (re)trained into the block.
    if any(a["action"] in (TRAIN, RESUME, RERUN) for a in arms.values()):
        commit = _commit_path(run_dir, bid)
        x_hash = json.loads(commit.read_text())["x_train_sha256"] if commit.exists() else None
        reason = None
        for arm in ARMS:
            if arms[arm]["action"] == PAUSE:
                continue
            st = R.load_arm_state(run_dir, bid, arm)
            if st is None:
                continue
            if st.provenance != "controller":
                reason = (f"{arm} is {st.provenance}: its training data were never "
                          "persisted, so no arm of this block may be (re)trained")
            elif arms[arm]["action"] != RERUN and (x_hash is None
                                                   or st.train_data_sha256 != x_hash):
                reason = (f"{arm} is bound to training data {st.train_data_sha256} but "
                          f"the block's committed dataset is {x_hash}; pairing would break")
            if reason:
                break
        if reason:
            arms = {a: {"action": PAUSE, "reasons": [reason], "code": "E-PAIRING-GUARD"}
                    for a in ARMS}
    return {"block_id": bid, "arms": arms}


def run_block(run_dir, config: CE1RunConfig, block_index: int, sealed: dict, *,
              generate_fn=None, ledger=None, extra_callbacks=None,
              train_executor=None, evaluate_executor=None) -> dict:
    """
    One controller pass over one block. A problem here is recorded and
    returned; it never touches another block's artifacts.

    `train_executor(block_index, {arm: action})` / `evaluate_executor(block_index,
    [arms])` let the 2-GPU launcher run arm tasks in isolated worker processes;
    by default they run sequentially in this process.
    """
    ledger = ledger or ledger_for(run_dir)
    fp = config.fingerprint()
    plan = plan_block(run_dir, config, block_index, sealed)
    bid = plan["block_id"]
    arms = plan["arms"]
    if any(a["action"] == PAUSE for a in arms.values()):
        res = record_block_resolution(run_dir, config, block_index, sealed, ledger=ledger)
        return {"block_id": bid, "status": res["status"], "plan": plan, "resolution": res}

    train_tasks = [arm for arm in ARMS if arms[arm]["action"] in (TRAIN, RESUME, RERUN)]
    for arm in ARMS:
        if arms[arm]["action"] == SKIP:
            ledger.record("SKIP", block_id=bid, arm=arm, config_hash=fp, reason="EVALUATED")

    errors = {}
    if train_tasks:
        ctx = prepare_block(run_dir, config, block_index, sealed,
                            generate_fn=generate_fn, ledger=ledger)
        if train_executor is None:
            for arm in train_tasks:
                try:
                    train_arm(run_dir, config, ctx, arm, arms[arm]["action"], ledger=ledger,
                              extra_callbacks=extra_callbacks)
                except Exception as exc:           # noqa: BLE001 - recorded
                    errors[arm] = f"{type(exc).__name__}: {exc}"
        else:
            del ctx                                 # workers load the dataset themselves
            errors.update(train_executor(block_index,
                                         {arm: arms[arm]["action"] for arm in train_tasks}))

    # Evaluate only once BOTH arms are trained.
    states = {arm: R.load_arm_state(run_dir, bid, arm) for arm in ARMS}
    trained = all(s is not None and s.status in (R.TRAINING_COMPLETE, R.EVALUATED)
                  for s in states.values())
    eval_tasks = []
    if trained:
        for arm in ARMS:
            if decide_arm(run_dir, config, bid, arm, sealed)[0] == EVALUATE:
                eval_tasks.append(arm)
        if eval_tasks and evaluate_executor is not None:
            errors.update(evaluate_executor(block_index, eval_tasks))
        else:
            for arm in eval_tasks:
                try:
                    evaluate_arm(run_dir, config, bid, arm, sealed, ledger=ledger)
                except Exception as exc:           # noqa: BLE001 - recorded
                    errors[arm] = f"{type(exc).__name__}: {exc}"
    res = record_block_resolution(run_dir, config, block_index, sealed, ledger=ledger)
    return {"block_id": bid, "status": res["status"], "plan": plan,
            "trained": train_tasks, "evaluated": eval_tasks, "errors": errors,
            "resolution": res}


# =====================================================================
# Operator actions (recorded; none can fail, exclude or select a block)
# =====================================================================

def _block_ever_evaluated(run_dir, block_id) -> bool:
    if any(e.get("event") in ("EVALUATION", "EVALUATED") and e.get("block_id") == block_id
           for e in ledger_for(run_dir).events()):
        return True
    return any((R.arm_dir(run_dir, block_id, a) / "evaluation.json").exists() for a in ARMS)


def operator_action(run_dir, config: CE1RunConfig, block_index: int, action: str, *,
                    reason: str, arm: str | None = None) -> dict:
    """
    GRANT_RETRY   arm paused by E-RETRY-CAP -> caps re-armed once
    REEVALUATE    arm paused by E-EVALUATION-INTEGRITY with an intact terminal model
                  -> deterministic re-score; must agree with every earlier ledger
                  evaluation of that model, otherwise the arm stays paused
    RESTART_BLOCK controller block with NO evaluation ever recorded -> archived and
                  restarted from a fresh committed dataset
    """
    if not reason or not reason.strip():
        raise ControllerError("an operator action requires a written reason")
    open_run(run_dir, config)
    sealed = ensure_sealed(run_dir, config)
    bid = R.block_id_for(block_index)
    led = ledger_for(run_dir)
    fp = config.fingerprint()
    if action == "GRANT_RETRY":
        d = decide_arm_detail(run_dir, config, bid, arm, sealed)
        if d["code"] != "E-RETRY-CAP":
            raise ControllerError(f"{bid}/{arm} is not paused by a retry cap ({d})")
        st = R.load_arm_state(run_dir, bid, arm)
        st.operator_retry_grants += 1
        R.save_arm_state(run_dir, st)
    elif action == "REEVALUATE":
        d = decide_arm_detail(run_dir, config, bid, arm, sealed)
        if d["code"] != "E-EVALUATION-INTEGRITY":
            raise ControllerError(f"{bid}/{arm} is not paused by an evaluation-integrity "
                                  f"failure ({d}); re-evaluation is not permitted")
        led.record("OPERATOR_ACTION", block_id=bid, arm=arm, config_hash=fp, action=action,
                   reason=reason, problems=d["reasons"])
        st = evaluate_arm(run_dir, config, bid, arm, sealed, ledger=led)
        return {"action": action, "block_id": bid, "arm": arm,
                "remaining_problems": verify_evaluation(run_dir, config, st, sealed)}
    elif action == "RESTART_BLOCK":
        states = [R.load_arm_state(run_dir, bid, a) for a in ARMS]
        if any(s is not None and s.provenance != "controller" for s in states):
            raise ControllerError("a legacy-recovered block cannot be restarted")
        if _block_ever_evaluated(run_dir, bid):
            raise ControllerError(f"{bid} has an evaluation on record; restarting it would be "
                                  "outcome-dependent. Refused.")
        dest = Path(run_dir) / "blocks" / f"_restarted_{bid}_{int(datetime.now().timestamp())}"
        shutil.move(str(_block_dir(run_dir, bid)), str(dest))
        arm = None
    else:
        raise ControllerError(f"unknown operator action {action!r}")
    led.record("OPERATOR_ACTION", block_id=bid, arm=arm, config_hash=fp, action=action,
               reason=reason)
    return {"action": action, "block_id": bid, "arm": arm}


# =====================================================================
# Whole run
# =====================================================================

def run_controller(run_dir, config: CE1RunConfig, *, sealed_source=None,
                   sealed_generate_fn=None, generate_fn=None, blocks=None,
                   extra_callbacks=None, train_executor=None, evaluate_executor=None,
                   repo_root=None) -> dict:
    """One restartable pass over the requested blocks (default: all)."""
    open_run(run_dir, config, repo_root=repo_root)
    ledger = ledger_for(run_dir)
    ledger.verify()                                 # fail closed on a malformed ledger
    sealed = ensure_sealed(run_dir, config, sealed_source=sealed_source,
                           generate_fn=sealed_generate_fn)
    manifest = json.loads((Path(run_dir) / RUN_MANIFEST).read_text())
    decision = manifest.get("legacy_block0_decision")
    if config.production and decision not in ("RECOVER", "RETRAIN"):
        raise ControllerError(
            "production requires an explicit, recorded decision for legacy block0 "
            "(RECOVER its terminal models, or RETRAIN block0) before any block runs")
    if decision == "RECOVER":
        st0 = [R.load_arm_state(run_dir, R.block_id_for(0), a) for a in ARMS]
        if not all(s is not None and s.provenance.startswith("legacy_recovery") for s in st0):
            raise ControllerError(
                "legacy_block0_decision=RECOVER but block0 has not been seeded from the "
                "legacy terminal models; run the recovery step first (the controller will "
                "never train block0 under a RECOVER decision)")
    results = []
    for i in (range(config.n_blocks) if blocks is None else blocks):
        try:
            results.append(run_block(run_dir, config, i, sealed, generate_fn=generate_fn,
                                     ledger=ledger, extra_callbacks=extra_callbacks,
                                     train_executor=train_executor,
                                     evaluate_executor=evaluate_executor))
        except Exception as exc:                    # noqa: BLE001 - isolate the block
            ledger.record("FAILURE", block_id=R.block_id_for(i), arm=None,
                          config_hash=config.fingerprint(), phase="block",
                          reason=type(exc).__name__, detail=str(exc)[:500])
            results.append({"block_id": R.block_id_for(i), "status": "ERROR",
                            "error": f"{type(exc).__name__}: {exc}"})
    return {"run_dir": str(run_dir), "blocks": results,
            "status": run_status(run_dir, config, sealed)}


def run_status(run_dir, config: CE1RunConfig, sealed: dict | None = None) -> dict:
    sealed = sealed or ensure_sealed(run_dir, config)
    per = {R.block_id_for(i): block_validity_gate(run_dir, config, i, sealed)["status"]
           for i in range(config.n_blocks)}
    return {"blocks": per,
            "n_valid_so_far": sum(v == "VALID" for v in per.values()),
            "needs_operator": [b for b, v in per.items() if v == "NEEDS_OPERATOR"],
            "awaiting_final_gate": [b for b, v in per.items()
                                    if v in ("LEGACY_RECOVERED_EVALUATED",
                                             "PENDING_FINAL_GATE")],
            "unresolved": [b for b, v in per.items() if v in UNRESOLVED],
            "note": "only the final gate resolves blocks; no count here is a stopping rule"}


def finalize(run_dir, config: CE1RunConfig, *, output_path=None, repo_root=None) -> tuple:
    """
    Apply the FINAL validity gate to all 8 predeclared blocks and, if the
    frozen counting rule is met, analyse ALL valid blocks in block order.

    Refuses while any block is unresolved (PENDING): there is no early stop.
    If fewer than min_valid_blocks are VALID (equivalently, more than
    failure_tolerance FAILED), a resolution report is written and no
    inferential certificate is produced.
    """
    from audit.cryptography.certificate import CERTIFICATE_SCHEMA_VERSION, write_certificate
    from audit.cryptography.experiments.ce1.design import paired_block_analysis
    from audit.cryptography.frozen_design import CE1_REPORTING_RULE
    from audit.cryptography.provenance import EXPERIMENT_DESIGN_VERSION, build_provenance
    from audit.cryptography.sealed_dataset import verify_shared_invariant

    ledger_for(run_dir).verify()
    sealed = ensure_sealed(run_dir, config)
    recs = [record_block_resolution(run_dir, config, i, sealed, final=True, deep=True)
            for i in range(config.n_blocks)]
    unresolved = {r["block_id"]: r["status"] for r in recs
                  if r["status"] not in ("VALID", "NOT_COUNTED")}
    if unresolved:
        raise ControllerError(f"unresolved blocks {unresolved}: every predeclared block must "
                              "be resolved by the frozen rules before analysis (no early "
                              "stop; NEEDS_OPERATOR is never converted into a failure)")
    valid = [r for r in recs if r["status"] == "VALID"]
    failed = [r for r in recs if r["status"] == "NOT_COUNTED"]
    report = {"n_blocks": config.n_blocks, "n_valid": len(valid), "n_failed": len(failed),
              "min_valid_blocks": config.min_valid_blocks,
              "failure_tolerance": CE1_SPEC.failure_tolerance,
              "blocks": recs, "rules": BLOCK_RESOLUTION_RULES, "utc": utc()}
    R.atomic_write_json(Path(run_dir) / "resolution_report.json", report)
    if len(valid) < config.min_valid_blocks or len(failed) > CE1_SPEC.failure_tolerance:
        ledger_for(run_dir).record("FINALIZE", block_id=None, arm=None,
                                   config_hash=config.fingerprint(), n_valid=len(valid),
                                   outcome="INSUFFICIENT_VALID_BLOCKS")
        raise ControllerError(
            f"{len(valid)} VALID / {len(failed)} NOT_COUNTED: the frozen rule requires >= "
            f"{config.min_valid_blocks} valid and <= {CE1_SPEC.failure_tolerance} failed. "
            "No inferential result may be reported; see resolution_report.json")
    shared = verify_shared_invariant(valid)
    results = paired_block_analysis([r["baseline_accuracy"] for r in valid],
                                    [r["destroyed_accuracy"] for r in valid],
                                    alpha=CE1_SPEC.alpha)
    not_counted = [{"block_id": r["block_id"], "code": r["code"], "reasons": r["reasons"],
                    "descriptive_only_accuracies": r.get("descriptive_accuracies")}
                   for r in failed]
    results.update({
        "blocks": recs, "valid_blocks_analysed": [r["block_id"] for r in valid],
        "n_blocks_requested": config.n_blocks, "n_blocks_valid": len(valid),
        "min_valid_blocks": config.min_valid_blocks,
        "failure_tolerance": CE1_SPEC.failure_tolerance,
        "failures": not_counted,
        "failed_blocks_note": ("failed / not-counted blocks are excluded by the predeclared "
                               "evidence rules only; any accuracies shown for them are "
                               "descriptive and do not enter the analysis"),
        "sealed_evaluation": shared, "seed_manifest": seeds_for(config),
        "reporting_rule": CE1_REPORTING_RULE,
        "block_resolution_rules": BLOCK_RESOLUTION_RULES,
        "resume_semantics": RESUME_SEMANTICS,
        "controller_protocol_version": CONTROLLER_PROTOCOL_VERSION,
        "config_fingerprint": config.fingerprint(),
        "training_protocol": config.scientific_dict(),
    })
    output_path = output_path or Path(run_dir) / "certificate.json"
    cert = {
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "experiment_design_version": EXPERIMENT_DESIGN_VERSION,
        "design_specification_version": DESIGN_SPECIFICATION_VERSION,
        "reference_configuration": REFERENCE.to_dict(),
        "provenance": build_provenance(config_id=EXPERIMENT_ID, probe_seed=config.seed_base,
                                       checkpoint_info=None),
        "preflight": {"production": config.production, "status": "CONTROLLER_VALIDATED"},
        "results": results,
        "claim_scope": results["interpretation_scope"],
    }
    if not config.production:
        cert["non_evidentiary"] = True
        cert["claim_scope"] = "NON-PRODUCTION CONFIG - structural validation only"
    ledger_for(run_dir).record("FINALIZE", block_id=None, arm=None,
                               config_hash=config.fingerprint(), n_valid=len(valid),
                               outcome="CERTIFIED")
    return write_certificate(cert, output_path, repo_root=repo_root), cert
