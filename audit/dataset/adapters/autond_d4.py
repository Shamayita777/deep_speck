"""
AutoND adapter for D4: the full staged DBitNet pipeline, including its stopping rule.

What is trained (read from the pinned checkout, 8d66e0a4, not assumed)
-------------------------------------------------------------------
``train_nets.train_neural_distinguisher`` runs a loop over rounds starting at
``starting_round``.  Per round it draws fresh training (10^7) and validation
(10^6) data, fits DBitNet for ``EPOCHS = 10`` epochs (batch 5000, Keras
default ``shuffle=True``) with ``ModelCheckpoint(monitor='val_loss',
save_best_only=True)``, and computes ``val_acc = max(history['val_acc'])``.
If ``val_acc <= ABORT_TRAINING_BELOW_ACC (0.505)`` the loop stops; otherwise
the round becomes ``best_round`` and the next round starts from this round's
best-val_loss weights (``model.load_weights``) on the same model object, so
the optimizer keeps its in-memory state.  Model: ``dbitnet.make_model(64)``,
``Adam(amsgrad=True)``, ``loss='mse'``, ``metrics=['acc']``.

This adapter runs that loop, stopping rule included, on PERSISTED per-round
partitions (one distinct partition per round, as upstream's fresh data per
round), from the registered starting round to a registered cap.  The STOP
decision uses the round's maximum validation accuracy; the carried and
evaluated checkpoint is the round's minimum-val_loss epoch -- the two differ
by design, exactly as upstream (see the training bundle's README).  Declared
departures: data are generated once at preflight and shared by the paired
fits and all replicates; TensorFlow is seeded; every epoch checkpoint is kept
and hash-chained; a registered cap bounds a fit that never stops.

Contract (``audit.dataset.d4_controlled_perturbation.D4Adapter``)
----------------------------------------------------------------
``train`` receives ONE training array: every registered round's training
partition, concatenated round-major, ``round_rows`` rows each; it splits them
at those offsets and refuses any other length, so the engine's label shuffle
and its verification cover every round.  Validation partitions (true labels,
both conditions) are given at construction.  The test partition is never
given (bound only as ``evaluation_digest``).  ``selection`` describes the
selected checkpoint of the COMPARISON round (the starting round, which every
fit trains whatever the stopping rule does); it is reloaded and used by
``predict``.  ``selection["progression"]`` records the pipeline's stopping
progression, recomputable from the committed records alone.

Why the lifecycle is not imported from ``gohr_d4``: that module executes
``import train_nets`` (Gohr's); the AutoND binding refuses any pinned module
name resolving outside the checkout.  This module imports no Gohr code.

Checkpoint lifecycle (per committed step = one (round, epoch))
-------------------------------------------------------------
(1) Keras writes ``stage_rRR/checkpoint_epoch_EEE.keras`` (full model incl.
optimizer state); (2) ``stage_rRR/checkpoint_epoch_EEE.metrics.json`` is
written atomically with the checkpoint SHA-256, ``val_loss``, ``val_acc``, the
step, the fit identity and the previous record's SHA-256 (one chain across
rounds); (3) ``state.json`` is replaced atomically, committing the step.

==  =============================================  =========================
#   Situation                                      Action
==  =============================================  =========================
1   committed step, record + checkpoint verify     used
2   incomplete write (``*.tmp``)                   quarantined, step rerun
3   checkpoint without record                      quarantined, step rerun
4   complete uncommitted orphan (record+ckpt)      quarantined, step rerun;
                                                   NEVER adopted
5   checkpoint hash mismatch                       refused, nothing moved
6   altered record / broken chain / bad head       refused, nothing moved
7   wrong condition/replicate/seed/config/data     refused, nothing moved
8   truncated or malformed state.json or record    refused, nothing moved
-   artifacts beyond the next step; artifacts      refused
    after the pipeline has stopped; a record
    without its checkpoint
==  =============================================  =========================

Round transitions are a deterministic function of committed evidence: round
r+1 starts from the LAST committed checkpoint of round r (optimizer state)
with the weights of round r's SELECTED checkpoint (upstream ``load_weights``).

Resume guarantees, stated exactly
---------------------------------
* A fit with N committed steps resumes at step N+1, from the committed
  checkpoint of step N (weights AND optimizer state, as saved by Keras).
  Committed steps are never retrained; a finished fit is never retrained.
* A resume at a round boundary rebuilds the boundary model by the transition
  rule above, so it is the same model an uninterrupted run would start from.
* NOT guaranteed: a bit-identical continuation of the training trajectory.
  The Keras shuffle order and other RNG state are not checkpointed; a resumed
  fit re-seeds TensorFlow from (seed, round, starting epoch), so the epochs
  after a resume differ from those an uninterrupted run would have produced.
  Every resume after a quarantine is recorded in ``retries``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from audit.dataset.integrity import (
    IntegrityError,
    labelled_partition_digest,
    read_json,
    sha256_file,
    write_json_atomic,
)

#: Per round: carried/evaluated checkpoint = argmin val_loss over the round's
#: committed epochs (ties: earliest); stop iff max val_acc over the round's
#: epochs <= stop threshold; the comparison round's selection is reloaded.
MODEL_SELECTION_RULE = ("per_round_argmin_val_loss_ties_earliest_carried_to_next_round_"
                        "stop_when_round_max_val_acc_at_most_threshold_comparison_round_selected_reloaded")

GENESIS_RECORD_SHA256 = "0" * 64
RECORD_SCHEMA = "d4-autond-epoch-record-2"
STATE_SCHEMA = "d4-autond-fit-state-2"
SELECTION_SCHEMA = "d4-autond-selection-2"
IDENTITY_FIELDS = ("seed", "condition", "replicate", "config_hash", "train_digest",
                   "round_train_digests", "validation_digests", "evaluation_digest", "model",
                   "run_binding", "adapter")

#: Identity of this adapter implementation, bound into every fit.
ADAPTER_IDENTITY = {"module": "audit.dataset.adapters.autond_d4", "record_schema": RECORD_SCHEMA,
                    "state_schema": STATE_SCHEMA, "selection_schema": SELECTION_SCHEMA,
                    "selection_rule": MODEL_SELECTION_RULE}
RECORD_FIELDS = frozenset({"schema_version", "step", "stage_round", "stage_epoch", "checkpoint",
                           "checkpoint_sha256", "val_loss", "val_acc", "logs", "previous_record_sha256",
                           *IDENTITY_FIELDS})

#: Batch size of every test-set prediction (run and verify --rerun-models).
PREDICT_BATCH_SIZE = 5000


# ============================================================
# Layout
# ============================================================


def stage_directory(fit_dir: Path, stage_round: int) -> Path:
    return Path(fit_dir) / f"stage_r{int(stage_round):02d}"


def epoch_checkpoint_path(fit_dir: Path, stage_round: int, epoch: int) -> Path:
    return stage_directory(fit_dir, stage_round) / f"checkpoint_epoch_{int(epoch):03d}.keras"


def epoch_record_path(fit_dir: Path, stage_round: int, epoch: int) -> Path:
    return stage_directory(fit_dir, stage_round) / f"checkpoint_epoch_{int(epoch):03d}.metrics.json"


def step_plan(rounds, epochs_per_round: int) -> list[tuple[int, int]]:
    """Every possible (round, epoch) step up to the cap; step k is plan[k-1]."""
    return [(int(r), e) for r in rounds for e in range(1, int(epochs_per_round) + 1)]


def _artifact_step(path: Path, plan: list) -> int | None:
    stem = path.name.split(".", 1)[0]
    if not stem.startswith("checkpoint_epoch_") or not path.parent.name.startswith("stage_r"):
        return None
    try:
        key = (int(path.parent.name[len("stage_r"):]), int(stem.rsplit("_", 1)[1]))
    except ValueError:
        return None
    return plan.index(key) + 1 if key in plan else -1


# ============================================================
# Record verification
# ============================================================


def _finite(value) -> bool:
    return isinstance(value, float) and bool(np.isfinite(value))


def verify_epoch_record(fit_dir: Path, *, step: int, plan: list, identity: dict, previous_sha256: str) -> dict:
    """Return the record of ``step`` if schema, identity, chain, metrics and checkpoint verify."""
    stage_round, stage_epoch = plan[step - 1]
    path = epoch_record_path(fit_dir, stage_round, stage_epoch)
    record = read_json(path, f"step {step} record")
    if record.get("schema_version") != RECORD_SCHEMA or set(record) != RECORD_FIELDS:
        raise IntegrityError(f"Epoch record {path} does not have the {RECORD_SCHEMA} schema.")
    for key in IDENTITY_FIELDS:
        if record[key] != identity[key]:
            raise IntegrityError(f"Epoch record {path} has {key}={record[key]!r}, expected "
                                 f"{identity[key]!r}. Refusing to mix fits.")
    expected_checkpoint = epoch_checkpoint_path(fit_dir, stage_round, stage_epoch).relative_to(fit_dir)
    if (record["step"], record["stage_round"], record["stage_epoch"]) != (step, stage_round, stage_epoch) \
            or record["checkpoint"] != expected_checkpoint.as_posix():
        raise IntegrityError(f"Epoch record {path} does not describe step {step} "
                             f"(round {stage_round}, epoch {stage_epoch}).")
    if record["previous_record_sha256"] != previous_sha256:
        raise IntegrityError(f"Epoch record {path} does not chain to the record of step {step - 1}; "
                             "a record was altered or replaced.")
    for metric in ("val_loss", "val_acc"):
        if not _finite(record[metric]):
            raise IntegrityError(f"Epoch record {path} has a missing or non-finite {metric}.")
        if record["logs"].get(metric) != record[metric]:
            raise IntegrityError(f"Epoch record {path}: {metric} disagrees with its own logs.")
    checkpoint = Path(fit_dir) / record["checkpoint"]
    if not checkpoint.is_file():
        raise IntegrityError(f"Checkpoint for step {step} is missing: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != record["checkpoint_sha256"]:
        raise IntegrityError(f"Checkpoint {checkpoint} hashes to {actual}, but its epoch record says "
                             f"{record['checkpoint_sha256']}. Refusing to use it.")
    return record


def verify_epoch_chain(fit_dir: Path, *, plan: list, steps: int, identity: dict) -> tuple[list, str]:
    records, head = [], GENESIS_RECORD_SHA256
    for step in range(1, int(steps) + 1):
        records.append(verify_epoch_record(fit_dir, step=step, plan=plan, identity=identity,
                                           previous_sha256=head))
        stage_round, stage_epoch = plan[step - 1]
        head = sha256_file(epoch_record_path(fit_dir, stage_round, stage_epoch))
    return records, head


# ============================================================
# Progression (the upstream stopping rule, recomputed from records)
# ============================================================


def round_summary(records: list, threshold: float) -> dict:
    """Selection and stop decision of one COMPLETE round."""
    best = min(records, key=lambda r: (r["val_loss"], r["stage_epoch"]))
    max_val_acc = max(r["val_acc"] for r in records)
    return {"stage_round": best["stage_round"], "selected_epoch": best["stage_epoch"],
            "selected_val_loss": best["val_loss"], "selected_val_acc": best["val_acc"],
            "checkpoint": best["checkpoint"], "checkpoint_sha256": best["checkpoint_sha256"],
            "epoch_checkpoint_sha256": [r["checkpoint_sha256"] for r in records],
            "max_val_acc": max_val_acc, "passed": bool(max_val_acc > threshold)}


def progression(records: list, *, rounds: list, epochs_per_round: int, threshold: float) -> dict:
    """Apply upstream's loop to committed records.

    status: "in_progress" (the next step must be trained), "stopped_at_threshold"
    (a complete round had max val_acc <= threshold), or "cap_reached" (every
    registered round passed).  best_round: the last passing round, or None.
    """
    summaries = []
    for index, stage_round in enumerate(rounds):
        block = records[index * epochs_per_round:(index + 1) * epochs_per_round]
        if len(block) < epochs_per_round:
            if len(records) > (index + 1) * epochs_per_round:
                raise IntegrityError("Records beyond an incomplete round.")
            status = "in_progress"
            break
        summary = round_summary(block, threshold)
        summaries.append(summary)
        if not summary["passed"]:
            status = "stopped_at_threshold"
            break
    else:
        status = "cap_reached"
    executed = len(summaries) * epochs_per_round
    if status != "in_progress" and len(records) != executed:
        raise IntegrityError(f"{len(records)} committed steps, but the stopping rule ends the pipeline after "
                             f"{executed}: training continued past a stop, or stopped early.")
    passed = [s for s in summaries if s["passed"]]
    return {"threshold": threshold, "status": status, "rounds_trained": [s["stage_round"] for s in summaries],
            "rounds_max_val_acc": [s["max_val_acc"] for s in summaries],
            "stop_round": summaries[-1]["stage_round"] if status == "stopped_at_threshold" else None,
            "best_round": passed[-1]["stage_round"] if passed else None,
            "best_val_acc": passed[-1]["max_val_acc"] if passed else None,
            "executed_steps": executed, "summaries": summaries}


def select_checkpoints(fit_dir: Path, *, plan: list, rounds: list, epochs_per_round: int, threshold: float,
                       identity: dict, chain_head: str, completed: int) -> dict:
    """Selection of a FINISHED fit, recomputed from its verified committed history."""
    records, head = verify_epoch_chain(fit_dir, plan=plan, steps=completed, identity=identity)
    if head != chain_head:
        raise IntegrityError("The epoch-record chain does not end at the head committed in state.json; "
                             "a record was altered, added or replaced.")
    prog = progression(records, rounds=rounds, epochs_per_round=epochs_per_round, threshold=threshold)
    if prog["status"] == "in_progress":
        raise IntegrityError("The fit has not finished: the stopping rule requires further rounds.")
    comparison = prog["summaries"][0]
    return {
        "schema_version": SELECTION_SCHEMA, "rule": MODEL_SELECTION_RULE,
        "selection_partition": "validation (per round)",
        "comparison_round": comparison["stage_round"], "selected_stage_round": comparison["stage_round"],
        "selected_epoch": comparison["selected_epoch"], "selected_val_loss": comparison["selected_val_loss"],
        "checkpoint": comparison["checkpoint"], "checkpoint_sha256": comparison["checkpoint_sha256"],
        "progression": prog, "record_chain_head": head, "completed_steps": int(completed),
        **{key: identity[key] for key in IDENTITY_FIELDS},
    }


# ============================================================
# State and recovery
# ============================================================


def write_state(state_path: Path, *, identity: dict, total_steps: int, completed: int, chain_head: str,
                retries: list) -> None:
    write_json_atomic(state_path, {
        "schema_version": STATE_SCHEMA, "identity": identity, "total_steps": int(total_steps),
        "completed_steps": int(completed), "record_chain_head": chain_head, "retries": list(retries),
    })


def load_committed_state(state_path: Path, fit_dir: Path, *, identity: dict, plan: list) -> dict:
    """Read and fully verify state.json (cases 1, 5, 6, 7, 8)."""
    state = read_json(state_path, "training state")
    if state.get("schema_version") != STATE_SCHEMA:
        raise IntegrityError(f"Training state {state_path} does not have the {STATE_SCHEMA} schema.")
    if state.get("identity") != identity or state.get("total_steps") != len(plan):
        raise IntegrityError(f"Training state {state_path} belongs to a different fit (condition, replicate, "
                             "seed, configuration, data, rounds or model). Refusing to resume.")
    completed = state.get("completed_steps")
    if not isinstance(completed, int) or isinstance(completed, bool) or not 0 <= completed <= len(plan):
        raise IntegrityError(f"Training state {state_path} has an invalid completed_steps.")
    if not isinstance(state.get("retries"), list):
        raise IntegrityError(f"Training state {state_path} has no retries list.")
    _, head = verify_epoch_chain(fit_dir, plan=plan, steps=completed, identity=identity)
    if head != state.get("record_chain_head"):
        raise IntegrityError("Committed epoch records do not end at the chain head in state.json.")
    return state


def classify_uncommitted(fit_dir: Path, *, plan: list, committed: int, chain_head: str, identity: dict,
                         finished: bool = False) -> dict:
    """Classify artifacts past the committed step without moving anything (cases 2-8)."""
    fit_dir = Path(fit_dir)
    files = []
    for stage in sorted(fit_dir.glob("stage_r*")):
        for path in sorted(stage.iterdir()):
            if not path.is_file():
                continue
            step = _artifact_step(path, plan)
            if step == -1:
                raise IntegrityError(f"{path} belongs to no registered (round, epoch); refusing.")
            if step is not None and step > committed:
                files.append((step, path))
    if not files:
        return {"files": [], "reason": None}
    if finished:
        raise IntegrityError(f"Artifacts exist after the pipeline stopped ({[str(p) for _, p in files][:4]}); "
                             "the write order cannot produce them. Refusing; nothing was moved.")
    nxt = committed + 1
    if {s for s, _ in files} != {nxt}:
        raise IntegrityError(f"Uncommitted artifacts exist for steps {sorted({s for s, _ in files})}; only "
                             f"step {nxt} can be uncommitted after a crash. Refusing; nothing was moved.")
    stage_round, stage_epoch = plan[nxt - 1]
    checkpoint = epoch_checkpoint_path(fit_dir, stage_round, stage_epoch)
    record = epoch_record_path(fit_dir, stage_round, stage_epoch)
    finals = {p for _, p in files if not p.name.endswith(".tmp")}
    if finals - {checkpoint, record}:
        raise IntegrityError(f"Unexpected uncommitted files {sorted(map(str, finals))}; refusing.")
    if record in finals and checkpoint not in finals:
        raise IntegrityError(f"Step {nxt} has a record but no checkpoint, which the write order cannot "
                             "produce. Refusing; nothing was moved.")
    if record in finals:
        verify_epoch_record(fit_dir, step=nxt, plan=plan, identity=identity, previous_sha256=chain_head)
        reason = "complete_uncommitted_orphan"
    elif checkpoint in finals:
        reason = "checkpoint_without_record"
    else:
        reason = "incomplete_write"
    return {"files": [str(p) for _, p in files], "reason": reason, "step": nxt,
            "stage_round": stage_round, "stage_epoch": stage_epoch}


def quarantine(fit_dir: Path, classification: dict) -> dict:
    """Move classified uncommitted files into quarantine/attempt_NN; delete nothing."""
    base = Path(fit_dir) / "quarantine"
    attempt = 1
    while (base / f"attempt_{attempt:02d}").exists():
        attempt += 1
    target = base / f"attempt_{attempt:02d}"
    target.mkdir(parents=True)
    moved = []
    for name in classification["files"]:
        source = Path(name)
        destination = target / f"{source.parent.name}__{source.name}"
        source.rename(destination)
        moved.append(destination.name)
    entry = {"step": classification["step"], "stage_round": classification["stage_round"],
             "stage_epoch": classification["stage_epoch"], "reason": classification["reason"],
             "quarantine": f"quarantine/{target.name}", "files": moved}
    write_json_atomic(target / "note.json", entry)
    return entry


# ============================================================
# Adapter
# ============================================================


def _fit_seed(seed: int, stage_round: int, initial_epoch: int) -> int:
    return int(np.random.SeedSequence([int(seed), int(stage_round), int(initial_epoch)])
               .generate_state(1, dtype=np.uint32)[0])


class AutoNDD4Adapter:
    """AutoND staged DBitNet pipeline under the D4 adapter contract.

    ``binding`` is a pin-verified ``AutoNDDatasetAdapter``; constructing it
    imported the pinned checkout, so ``dbitnet`` and ``train_nets`` are
    AutoND's and are checked to resolve inside the checkout before use.
    """

    def __init__(self, *, binding, rounds, round_rows, validation: dict, evaluation_digest: str,
                 epochs_per_round: int, batch_size: int, seed: int, upstream_commit: str,
                 stop_threshold: float, run_binding: dict, **kwargs) -> None:
        if kwargs.get("test_x") is not None or kwargs.get("test_y") is not None:
            raise ValueError("The D4 adapter must not receive test data; pass evaluation_digest.")
        unknown = set(kwargs) - {"test_x", "test_y"}
        if unknown:
            raise TypeError(f"Unexpected arguments {sorted(unknown)}.")
        self.binding = binding
        self.rounds = [int(r) for r in rounds]
        self.round_rows = [int(n) for n in round_rows]
        if len(self.rounds) != len(self.round_rows) or not self.rounds:
            raise ValueError("rounds and round_rows must be non-empty and of equal length.")
        if sorted(validation) != sorted(self.rounds):
            raise ValueError("Exactly one validation partition per registered round is required.")
        self.validation = {int(r): (validation[r][0], np.asarray(validation[r][1]).reshape(-1)) for r in self.rounds}
        self.evaluation_digest = str(evaluation_digest)
        self.epochs = int(epochs_per_round)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.upstream_commit = str(upstream_commit)
        self.stop_threshold = float(stop_threshold)
        if not (isinstance(run_binding, dict) and run_binding):
            raise ValueError("run_binding (preflight/run identity) is required.")
        self.run_binding = dict(run_binding)
        self.plan = step_plan(self.rounds, self.epochs)
        self.model = None
        self.selection = None

    # -- pinned code --------------------------------------------------

    def _pinned(self, name: str):
        module = sys.modules.get(name)
        root = Path(self.binding.autond_root).resolve()
        location = Path(getattr(module, "__file__", "") or "").resolve() if module else None
        if module is None or location is None or root not in location.parents:
            raise IntegrityError(f"{name} is not bound from the pinned checkout {root} (resolved to {location}).")
        return module

    def check_stop_threshold(self) -> None:
        upstream = getattr(self._pinned("train_nets"), "ABORT_TRAINING_BELOW_ACC", None)
        if upstream != self.stop_threshold:
            raise IntegrityError(f"Registered stop threshold {self.stop_threshold} differs from the pinned "
                                 f"train_nets.ABORT_TRAINING_BELOW_ACC = {upstream}.")

    def build_model(self):
        import tensorflow as tf

        model = self._pinned("dbitnet").make_model(2 * int(self.binding.plain_bits))
        model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True), loss="mse", metrics=["acc"])
        return model

    def model_description(self) -> dict:
        return {"architecture": "dbitnet", "builder": "dbitnet.make_model(2 * plain_bits)",
                "input_bits": 2 * int(self.binding.plain_bits), "optimizer": "Adam(amsgrad=True)",
                "loss": "mse", "metrics": ["acc"], "batch_size": self.batch_size, "shuffle": True,
                "rounds": self.rounds, "round_rows": self.round_rows, "epochs_per_round": self.epochs,
                "stop_threshold": self.stop_threshold, "comparison_round": self.rounds[0],
                "upstream_commit": self.upstream_commit, "cipher": self.binding.cipher_name,
                "delta_plain_hex": hex(self.binding.delta_plain_int)}

    # -- identity -----------------------------------------------------

    def split_rounds(self, train_x, train_y) -> list:
        train_y = np.asarray(train_y).reshape(-1)
        if len(train_x) != sum(self.round_rows) or len(train_y) != len(train_x):
            raise IntegrityError(f"Training arrays have {len(train_x)} rows; the registered rounds need "
                                 f"{sum(self.round_rows)} ({self.round_rows}). Refusing.")
        out, start = [], 0
        for rows in self.round_rows:
            out.append((train_x[start:start + rows], train_y[start:start + rows]))
            start += rows
        return out

    def fit_identity(self, train_x, train_y, *, condition, replicate, run_config_hash) -> dict:
        rounds = self.split_rounds(train_x, train_y)
        return {
            "seed": self.seed, "condition": str(condition), "replicate": int(replicate),
            "config_hash": str(run_config_hash), "train_digest": labelled_partition_digest(train_x, train_y),
            "round_train_digests": [labelled_partition_digest(x, y) for x, y in rounds],
            "validation_digests": [labelled_partition_digest(*self.validation[r]) for r in self.rounds],
            "evaluation_digest": self.evaluation_digest, "model": self.model_description(),
            "run_binding": self.run_binding, "adapter": ADAPTER_IDENTITY,
        }

    # -- lifecycle ----------------------------------------------------

    def _progress(self, fit_dir: Path, *, completed: int, identity: dict) -> dict:
        records, _ = verify_epoch_chain(fit_dir, plan=self.plan, steps=completed, identity=identity)
        return progression(records, rounds=self.rounds, epochs_per_round=self.epochs, threshold=self.stop_threshold)

    def _starting_model(self, fit_dir: Path, *, completed: int, identity: dict):
        """Model to continue from after ``completed`` committed steps."""
        import tensorflow as tf

        if completed == 0:
            tf.keras.utils.set_random_seed(_fit_seed(self.seed, self.rounds[0], 0))
            return self.build_model()
        last_round, last_epoch = self.plan[completed - 1]
        model = tf.keras.models.load_model(epoch_checkpoint_path(fit_dir, last_round, last_epoch), compile=True)
        if last_epoch == self.epochs:
            # Round boundary (reached only when the round passed): optimizer state from
            # the last epoch, weights from the round's selected epoch (load_weights).
            index = self.rounds.index(last_round)
            records, _ = verify_epoch_chain(fit_dir, plan=self.plan, steps=completed, identity=identity)
            chosen = round_summary(records[index * self.epochs:(index + 1) * self.epochs], self.stop_threshold)
            best = tf.keras.models.load_model(Path(fit_dir) / chosen["checkpoint"], compile=False)
            model.set_weights(best.get_weights())
        return model

    def train(self, train_x, train_y, *, checkpoint_path, state_path, replicate, condition, run_config_hash):
        import tensorflow as tf

        if not run_config_hash:
            raise ValueError("run_config_hash is required.")
        self.check_stop_threshold()
        fit_dir, state_path = Path(checkpoint_path), Path(state_path)
        fit_dir.mkdir(parents=True, exist_ok=True)
        identity = self.fit_identity(train_x, train_y, condition=condition, replicate=replicate,
                                     run_config_hash=run_config_hash)
        data = self.split_rounds(train_x, train_y)

        completed, head, retries = 0, GENESIS_RECORD_SHA256, []
        if state_path.exists():
            state = load_committed_state(state_path, fit_dir, identity=identity, plan=self.plan)
            completed, head, retries = state["completed_steps"], state["record_chain_head"], list(state["retries"])
        finished = self._progress(fit_dir, completed=completed, identity=identity)["status"] != "in_progress"
        classification = classify_uncommitted(fit_dir, plan=self.plan, committed=completed, chain_head=head,
                                              identity=identity, finished=finished)
        if classification["files"]:
            retries.append(quarantine(fit_dir, classification))
            write_state(state_path, identity=identity, total_steps=len(self.plan), completed=completed,
                        chain_head=head, retries=retries)
            print(f"Quarantined uncommitted step {classification['step']} ({classification['reason']}); "
                  "it will be retrained.")

        while self._progress(fit_dir, completed=completed, identity=identity)["status"] == "in_progress":
            index = completed // self.epochs
            stage_round, initial_epoch = self.rounds[index], completed % self.epochs
            self.model = self._starting_model(fit_dir, completed=completed, identity=identity)
            stage_dir = stage_directory(fit_dir, stage_round)
            stage_dir.mkdir(parents=True, exist_ok=True)
            tf.keras.utils.set_random_seed(_fit_seed(self.seed, stage_round, initial_epoch))
            callback = _committed_epoch_callback(fit_dir=fit_dir, state_path=state_path, identity=identity,
                                                 plan=self.plan, stage_round=stage_round,
                                                 offset=index * self.epochs, chain_head=head, retries=retries)
            x, y = data[index]
            self.model.fit(
                x, y, validation_data=self.validation[stage_round], epochs=self.epochs,
                initial_epoch=initial_epoch, batch_size=self.batch_size, shuffle=True, verbose=2,
                callbacks=[tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(stage_dir / "checkpoint_epoch_{epoch:03d}.keras"), save_weights_only=False,
                    save_best_only=False, save_freq="epoch", verbose=0), callback])
            state = load_committed_state(state_path, fit_dir, identity=identity, plan=self.plan)
            completed, head, retries = state["completed_steps"], state["record_chain_head"], list(state["retries"])
            if completed != (index + 1) * self.epochs:
                raise IntegrityError(f"Round {stage_round} ended with {completed} committed steps; expected "
                                     f"{(index + 1) * self.epochs}.")

        selection = select_checkpoints(fit_dir, plan=self.plan, rounds=self.rounds, epochs_per_round=self.epochs,
                                       threshold=self.stop_threshold, identity=identity, chain_head=head,
                                       completed=completed)
        selection["retries"] = retries
        selection_path = fit_dir / "selected_checkpoint.json"
        if selection_path.exists():
            if read_json(selection_path, "selection record") != selection:
                raise IntegrityError(f"Existing selection record {selection_path} disagrees with the recomputed "
                                     "selection. Refusing to overwrite it.")
        else:
            write_json_atomic(selection_path, selection)
        self.model = tf.keras.models.load_model(fit_dir / selection["checkpoint"], compile=True)
        self.selection = selection
        return self.model

    def predict(self, features):
        if self.selection is None or self.model is None:
            raise RuntimeError("predict() before train(): no selected checkpoint is loaded.")
        return self.model.predict(features, batch_size=PREDICT_BATCH_SIZE, verbose=0)


def _committed_epoch_callback(*, fit_dir, state_path, identity, plan, stage_round, offset, chain_head, retries):
    """Keras callback: after Keras saves a round-epoch checkpoint, write its record, then commit."""
    import tensorflow as tf

    class CommittedEpochCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.chain_head = chain_head

        def on_epoch_end(self, epoch, logs=None):
            stage_epoch = int(epoch + 1)
            step = offset + stage_epoch
            if plan[step - 1] != (stage_round, stage_epoch):
                raise IntegrityError(f"Step {step} is not (round {stage_round}, epoch {stage_epoch}).")
            checkpoint = epoch_checkpoint_path(fit_dir, stage_round, stage_epoch)
            if not checkpoint.is_file():
                raise IntegrityError(f"Step {step} checkpoint was not written: {checkpoint}")
            epoch_logs = {k: float(v) for k, v in (logs or {}).items() if np.isscalar(v)}
            for metric in ("val_loss", "val_acc"):
                if not np.isfinite(epoch_logs.get(metric, float("nan"))):
                    raise IntegrityError(f"Step {step}: {metric} is missing or non-finite.")
            record_path = epoch_record_path(fit_dir, stage_round, stage_epoch)
            if record_path.exists():
                raise IntegrityError(f"Refusing to overwrite existing epoch record {record_path}.")
            write_json_atomic(record_path, {
                "schema_version": RECORD_SCHEMA, "step": step, "stage_round": stage_round,
                "stage_epoch": stage_epoch, "checkpoint": checkpoint.relative_to(fit_dir).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint), "val_loss": epoch_logs["val_loss"],
                "val_acc": epoch_logs["val_acc"], "logs": epoch_logs,
                "previous_record_sha256": self.chain_head, **identity,
            })
            self.chain_head = sha256_file(record_path)
            write_state(state_path, identity=identity, total_steps=len(plan), completed=step,
                        chain_head=self.chain_head, retries=retries)

    return CommittedEpochCallback()
