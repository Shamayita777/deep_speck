#!/usr/bin/env python3
"""
The Keras half of the recovery claim, EXECUTED.

arm_recovery.RECOVERY_EXACTNESS claims that model weights, optimizer
state, the completed epoch index and the learning-rate schedule are
recovered exactly. Those are claims about what Keras does, not about
what this code does, so they are checked against real Keras here rather
than assumed from documentation. The review condition was explicit: do
not claim recovery is exact unless the epoch number, optimizer state,
learning-rate schedule and random/shuffle behaviour have actually been
checked.

The method throughout is a CONTROLLED COMPARISON. An interrupted-and-
resumed run is compared against an uninterrupted run of the same length
from the same initial weights, and the quantities that must agree are
required to agree exactly. A test that only inspected the resumed run
could not distinguish a correct resume from a plausible-looking one.

Skips itself when Keras is unavailable, so the generic suite still runs
in an environment without TensorFlow.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import keras
    HAVE_KERAS = True
except Exception:                                                # noqa: BLE001
    HAVE_KERAS = False

from audit.implementation.architecture_signature import architecture_signature
from audit.implementation.arm_recovery import (
    ArmRecoveryRecord,
    EpochLog,
    build_callbacks,
    resume_point,
)
from audit.implementation.durable_state import CommittedCheckpointStore, sha256_file

EPOCHS = 6
INTERRUPT_AFTER = 2


def _cyclic_lr(num_epochs, high_lr, low_lr):
    """The reference schedule, transcribed for this test only."""
    return lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (
        high_lr - low_lr)


def _model(seed=0):
    keras.utils.set_random_seed(seed)
    inp = keras.layers.Input(shape=(8,))
    x = keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(1e-5))(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=["acc"])
    return m


def _data(n=64, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 8)).astype("float32"),
            rng.integers(0, 2, n).astype("float32"),
            rng.random((n // 2, 8)).astype("float32"),
            rng.integers(0, 2, n // 2).astype("float32"))


class _LRRecorder(keras.callbacks.Callback if HAVE_KERAS else object):
    """Records the learning rate actually in force at each epoch."""

    def __init__(self):
        super().__init__()
        self.seen: dict[int, float] = {}

    def on_epoch_end(self, epoch, logs=None):
        self.seen[epoch] = float(
            keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))


@unittest.skipUnless(HAVE_KERAS, "Keras is not installed in this environment")
class TestKerasRecoveryIsActuallyExact(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._stores: dict[str, CommittedCheckpointStore] = {}
        self.X, self.Y, self.Xv, self.Yv = _data()

    def tearDown(self):
        self.tmp.cleanup()

    def _committed(self, d):
        """Path of the slot the manifest currently commits."""
        store = self._store(d)
        manifest = store.manifest()
        assert manifest is not None, "nothing has been committed for this arm"
        return store.slot_path(manifest.committed_slot)

    def _fit(self, model, *, initial_epoch, callbacks, epochs=EPOCHS):
        return model.fit(self.X, self.Y, initial_epoch=initial_epoch, epochs=epochs,
                         batch_size=16, validation_data=(self.Xv, self.Yv),
                         shuffle=True, callbacks=callbacks, verbose=0)

    def _arm(self, name):
        d = self.dir / name
        d.mkdir(parents=True, exist_ok=True)
        log = EpochLog(d / "epochs.jsonl")
        rec = ArmRecoveryRecord(arm_id=name, block_id="block0", seed=7, epochs_total=EPOCHS)
        self._stores[name] = CommittedCheckpointStore(
            arm_dir=d, arm_id=name, block_id="block0", seed=7, epochs_total=EPOCHS)
        return d, log, rec

    def _store(self, d):
        return self._stores[Path(d).name]

    def _callbacks(self, d, log, rec, *, session, lr_recorder):
        return [keras.callbacks.LearningRateScheduler(_cyclic_lr(10, 0.002, 0.0001)),
                lr_recorder] + build_callbacks(
            store=self._store(d), log=log, record=rec, record_path=d / "recovery.json",
            best_path=d / "best_val_loss.keras", session_id=session)

    # --- epoch index -------------------------------------------------
    def test_the_epoch_index_continues_and_repeats_nothing(self):
        d, log, rec = self._arm("epochs")
        m = _model()
        lr1 = _LRRecorder()
        self._fit(m, initial_epoch=0, epochs=INTERRUPT_AFTER,
                  callbacks=self._callbacks(d, log, rec, session="s1", lr_recorder=lr1))
        self.assertEqual(log.highest_contiguous(), INTERRUPT_AFTER)

        m2 = keras.models.load_model(self._committed(d), compile=True)
        lr2 = _LRRecorder()
        self._fit(m2, initial_epoch=log.highest_contiguous(),
                  callbacks=self._callbacks(d, log, rec, session="s2", lr_recorder=lr2))

        self.assertEqual([r["epoch"] for r in log.read()], list(range(EPOCHS)))
        self.assertEqual(log.reconcile(EPOCHS - 1), [])
        self.assertEqual(sorted(lr1.seen) + sorted(lr2.seen), list(range(EPOCHS)))

    # --- optimizer state ---------------------------------------------
    def test_the_optimizer_step_counter_continues_rather_than_restarting(self):
        d, log, rec = self._arm("optsteps")
        m = _model()
        self._fit(m, initial_epoch=0, epochs=INTERRUPT_AFTER,
                  callbacks=self._callbacks(d, log, rec, session="s1",
                                            lr_recorder=_LRRecorder()))
        steps_before = int(keras.ops.convert_to_numpy(m.optimizer.iterations))
        self.assertGreater(steps_before, 0)

        m2 = keras.models.load_model(self._committed(d), compile=True)
        self.assertIsNotNone(m2.optimizer, "the resume checkpoint carried no optimizer")
        steps_restored = int(keras.ops.convert_to_numpy(m2.optimizer.iterations))
        self.assertEqual(steps_restored, steps_before,
                         "Adam's step counter was reset by the reload")

        self._fit(m2, initial_epoch=INTERRUPT_AFTER,
                  callbacks=self._callbacks(d, log, rec, session="s2",
                                            lr_recorder=_LRRecorder()))
        self.assertGreater(int(keras.ops.convert_to_numpy(m2.optimizer.iterations)),
                           steps_restored)

    def test_adam_moment_estimates_survive_the_reload_exactly(self):
        """
        Weights alone are not enough: a reinitialized Adam restarts its
        first and second moment estimates at zero, so the first
        post-resume steps take the wrong size. The moments must come
        back bit-identical.
        """
        d, log, rec = self._arm("moments")
        m = _model()
        self._fit(m, initial_epoch=0, epochs=INTERRUPT_AFTER,
                  callbacks=self._callbacks(d, log, rec, session="s1",
                                            lr_recorder=_LRRecorder()))
        before = [np.array(keras.ops.convert_to_numpy(v)) for v in m.optimizer.variables]
        self.assertTrue(any(np.any(v != 0) for v in before),
                        "no optimizer slot was populated; the test proves nothing")

        m2 = keras.models.load_model(self._committed(d), compile=True)
        after = [np.array(keras.ops.convert_to_numpy(v)) for v in m2.optimizer.variables]
        self.assertEqual(len(before), len(after))
        for i, (a, b) in enumerate(zip(before, after)):
            self.assertEqual(a.shape, b.shape, f"optimizer variable {i} changed shape")
            np.testing.assert_array_equal(a, b, err_msg=f"optimizer variable {i} differs")

    def test_weights_survive_the_reload_exactly(self):
        d, log, rec = self._arm("weights")
        m = _model()
        self._fit(m, initial_epoch=0, epochs=INTERRUPT_AFTER,
                  callbacks=self._callbacks(d, log, rec, session="s1",
                                            lr_recorder=_LRRecorder()))
        before = [np.array(w) for w in m.get_weights()]
        m2 = keras.models.load_model(self._committed(d), compile=True)
        for i, (a, b) in enumerate(zip(before, [np.array(w) for w in m2.get_weights()])):
            np.testing.assert_array_equal(a, b, err_msg=f"weight tensor {i} differs")
        self.assertEqual(architecture_signature(m2)["digest"],
                         architecture_signature(m)["digest"])

    # --- learning-rate schedule --------------------------------------
    def test_the_lr_schedule_after_resume_matches_an_uninterrupted_run(self):
        """
        Controlled comparison: the rate in force at each epoch of the
        interrupted-then-resumed run must equal the rate at the same
        epoch of an uninterrupted run. Were `initial_epoch` not honoured,
        the post-resume schedule would restart at epoch 0 of the cycle
        and this comparison would fail.
        """
        d, log, rec = self._arm("lr_resumed")
        m = _model()
        lr1, lr2 = _LRRecorder(), _LRRecorder()
        self._fit(m, initial_epoch=0, epochs=INTERRUPT_AFTER,
                  callbacks=self._callbacks(d, log, rec, session="s1", lr_recorder=lr1))
        m2 = keras.models.load_model(self._committed(d), compile=True)
        self._fit(m2, initial_epoch=INTERRUPT_AFTER,
                  callbacks=self._callbacks(d, log, rec, session="s2", lr_recorder=lr2))
        resumed = {**lr1.seen, **lr2.seen}

        d2, log2, rec2 = self._arm("lr_straight")
        straight = _LRRecorder()
        self._fit(_model(), initial_epoch=0,
                  callbacks=self._callbacks(d2, log2, rec2, session="s0",
                                            lr_recorder=straight))

        self.assertEqual(sorted(resumed), sorted(straight.seen))
        for epoch in sorted(straight.seen):
            self.assertAlmostEqual(
                resumed[epoch], straight.seen[epoch], places=12,
                msg=f"learning rate at epoch {epoch} differs after a resume")

    def test_the_logged_learning_rate_matches_the_schedule(self):
        d, log, rec = self._arm("lr_logged")
        self._fit(_model(), initial_epoch=0,
                  callbacks=self._callbacks(d, log, rec, session="s1",
                                            lr_recorder=_LRRecorder()))
        schedule = _cyclic_lr(10, 0.002, 0.0001)
        for r in log.read():
            self.assertAlmostEqual(r["learning_rate"], schedule(r["epoch"]), places=9)

    # --- best-checkpoint record --------------------------------------
    def _commit_losses(self, d, log, losses, *, log_upto=None):
        """
        Drive a real arm through `losses`, committing each epoch through
        the two-slot store exactly as the committer does, and appending
        the derived log row only for the first `log_upto` epochs. That
        last part is how a crash between the manifest commit and the log
        append is reproduced without killing a Keras process.
        """
        store = self._store(d)
        model = _model()
        best_so_far = None
        best_epoch = None
        for epoch, val_loss in enumerate(losses):
            if best_so_far is None or val_loss < best_so_far:
                best_so_far, best_epoch = val_loss, epoch
            store.commit_epoch(epoch=epoch, save_fn=lambda p: model.save(p),
                               best_monitor_value=best_so_far,
                               best_checkpoint_epoch=best_epoch,
                               session_id="s1")
            if log_upto is None or epoch < log_upto:
                log.append({"epoch": epoch, "status": "COMPLETED", "val_loss": val_loss,
                            "val_accuracy": 0.5, "session_id": "s1"})
        return best_so_far, best_epoch

    def _best_callback(self, d, log, rec, *, session="s2"):
        callbacks = build_callbacks(
            store=self._store(d), log=log, record=rec, record_path=d / "recovery.json",
            best_path=d / "best_val_loss.keras", session_id=session)
        return next(c for c in callbacks
                    if isinstance(c, keras.callbacks.ModelCheckpoint))

    def test_a_resume_cannot_demote_the_best_checkpoint(self):
        """
        A fresh ModelCheckpoint starts from infinity, so without
        restoring `best` the first post-resume epoch overwrites the best
        checkpoint with a worse one - silently, since the file is still
        called "best". The callback factory restores it from the COMMIT
        MANIFEST; this checks that the restoration reaches the callback.
        """
        d, log, rec = self._arm("best")
        self._commit_losses(d, log, (0.30, 0.11, 0.25))
        best_cb = self._best_callback(d, log, rec)
        self.assertAlmostEqual(best_cb.best, 0.11,
                               msg="the best monitored value was not restored")
        self.assertNotEqual(best_cb.best, float("inf"))
        self.assertEqual(self._store(d).manifest().best_checkpoint_epoch, 1)

    def test_the_restored_best_comes_from_the_manifest_when_the_log_lags(self):
        """
        ITEM 5, measured on the real stack. The manifest is replaced
        before the log row is appended, so a kill between the two leaves
        the log one epoch short. Here epoch 2 sets the best and its log
        row is missing: reading the best from the log would restore 0.30
        and let the next epoch overwrite a checkpoint that was better by
        a wide margin.
        """
        d, log, rec = self._arm("best_lagging_log")
        self._commit_losses(d, log, (0.30, 0.28, 0.09), log_upto=2)

        self.assertEqual(log.completed_epochs(), [0, 1])
        self.assertEqual(self._store(d).manifest().committed_epoch, 2)
        self.assertAlmostEqual(log.best_value("val_loss", mode="min"), 0.28)

        best_cb = self._best_callback(d, log, rec)
        self.assertAlmostEqual(best_cb.best, 0.09,
                               msg="the best came from the lagging log, not the manifest")
        self.assertEqual(self._store(d).manifest().best_checkpoint_epoch, 2)

    def test_a_fresh_arm_restores_nothing_rather_than_trusting_the_log(self):
        """
        The log is derived and is appended only after a commit, so log
        rows with no manifest cannot arise from a normal run - they mean
        the manifest was lost. In that state the factory restores
        nothing: a value from an unverifiable source is not evidence,
        and the first epoch of the retry writes an honest best.
        """
        d, log, rec = self._arm("best_no_manifest")
        for epoch, vl in enumerate((0.30, 0.11, 0.25)):
            log.append({"epoch": epoch, "status": "COMPLETED", "val_loss": vl,
                        "val_accuracy": 0.5, "session_id": "s1"})
        self.assertIsNone(self._store(d).manifest())
        best_cb = self._best_callback(d, log, rec)
        self.assertNotAlmostEqual(best_cb.best or float("inf"), 0.11, places=6)

    def test_promotion_keeps_the_best_from_the_lost_epoch(self):
        """
        When a corrupt slot is quarantined, the epoch it held is lost and
        retrained - but ModelCheckpoint runs BEFORE the committer, so if
        that lost epoch set the best, its best-checkpoint file was
        already written and is intact. The promoted manifest therefore
        keeps the best value, and the retrained epoch cannot overwrite a
        better artifact with a worse one.
        """
        d, log, rec = self._arm("best_promoted")
        self._commit_losses(d, log, (0.30, 0.28, 0.07))
        store = self._store(d)
        manifest = store.manifest()
        self.assertEqual(manifest.committed_epoch, 2)
        self.assertAlmostEqual(manifest.best_monitor_value, 0.07)

        # corrupt the committed slot: epoch 2 is lost, epoch 1 survives
        store.slot_path(manifest.committed_slot).write_bytes(b"truncated")
        decision = store.prepare_for_resume()
        self.assertEqual(decision.epoch, 1)
        self.assertEqual(decision.source, "previous_promoted")
        self.assertTrue(decision.degraded)

        promoted = store.manifest()
        self.assertAlmostEqual(promoted.best_monitor_value, 0.07,
                               msg="the best from the lost epoch was discarded")
        self.assertEqual(promoted.best_checkpoint_epoch, 2)
        self.assertAlmostEqual(self._best_callback(d, log, rec).best, 0.07)

    # --- the resume checkpoint leads the log -------------------------
    def test_the_resume_checkpoint_is_written_before_the_epoch_is_logged(self):
        """
        Ordering matters. A kill between the two must leave a checkpoint
        one epoch AHEAD of the log - that epoch is simply redone. The
        reverse order would leave a log claiming an epoch whose weights
        were never saved, and the resume would then train from stale
        weights while reporting a continuous history.
        """
        d, log, rec = self._arm("ordering")
        m = _model()
        order: list[str] = []

        store = self._store(d)

        class Watcher(keras.callbacks.Callback):
            """Runs AFTER EpochCommitter, so the commit must already be durable."""
            def on_epoch_end(self, epoch, logs=None):
                manifest = store.manifest()
                order.append("committed" if (
                    manifest is not None and manifest.committed_epoch == epoch
                    and store.slot_path(manifest.committed_slot).is_file())
                    else "not-committed")

        cbs = self._callbacks(d, log, rec, session="s1", lr_recorder=_LRRecorder())
        self._fit(m, initial_epoch=0, epochs=1, callbacks=cbs + [Watcher()])
        self.assertEqual(order, ["committed"],
                         "the epoch was logged before its checkpoint was committed")
        self.assertTrue(self._committed(d).is_file())
        self.assertEqual(log.completed_epochs(), [0])
        rec_loaded = ArmRecoveryRecord.load(d / "recovery.json")
        self.assertEqual(rec_loaded.epochs_completed, 1)
        self.assertIsNotNone(rec_loaded.committed_sha256)
        resume, decision, _ = resume_point(self._store(d), log)
        self.assertEqual(resume, 1)
        self.assertEqual(decision.epoch, 0)

    # --- the shuffle stream: measured, not assumed -------------------
    def test_shuffling_is_active_and_seed_dependent(self):
        """
        Control for the test below. If shuffling were inactive or
        seed-independent, a resume would reproduce the batch sequence
        trivially and the exactness claim would be vacuous.
        """
        a, b, c = _BatchTrace(), _BatchTrace(), _BatchTrace()
        X, Y, Xv, Yv = _data(512, seed=3)
        for mdl, cb, shuffle in ((_model(42), a, True), (_model(99), b, True),
                                 (_model(42), c, False)):
            mdl.fit(X, Y, epochs=3, batch_size=64, validation_data=(Xv, Yv),
                    shuffle=shuffle, callbacks=[cb], verbose=0)
        self.assertNotEqual(a.traces, b.traces,
                            "the batch sequence does not depend on the global seed")
        self.assertNotEqual(a.traces, c.traces,
                            "shuffle=True does not differ from shuffle=False")

    def test_a_reseeded_resume_reproduces_the_batch_sequence_exactly(self):
        """
        The measured basis for listing the shuffle permutation under
        `recovered_exactly`.

        A resume in a FRESH PROCESS is simulated by reseeding to the
        arm's seed before the second fit, which is what the adapter does
        inside train_arm. The resumed run's per-epoch batch sequence must
        equal the uninterrupted control's at the same epochs, because
        Keras derives each epoch's permutation from the absolute epoch
        index rather than from how many epochs preceded it in this
        process.
        """
        X, Y, Xv, Yv = _data(512, seed=3)
        d, log, rec = self._arm("shuffle_resumed")
        first, second, control = _BatchTrace(), _BatchTrace(), _BatchTrace()

        m = _model(42)
        m.fit(X, Y, initial_epoch=0, epochs=INTERRUPT_AFTER, batch_size=64,
              validation_data=(Xv, Yv), shuffle=True, verbose=0,
              callbacks=self._callbacks(d, log, rec, session="s1",
                                        lr_recorder=_LRRecorder()) + [first])
        keras.utils.set_random_seed(42)                 # the fresh process
        m2 = keras.models.load_model(self._committed(d), compile=True)
        m2.fit(X, Y, initial_epoch=INTERRUPT_AFTER, epochs=EPOCHS, batch_size=64,
               validation_data=(Xv, Yv), shuffle=True, verbose=0,
               callbacks=self._callbacks(d, log, rec, session="s2",
                                         lr_recorder=_LRRecorder()) + [second])

        d2, log2, rec2 = self._arm("shuffle_control")
        m3 = _model(42)
        m3.fit(X, Y, initial_epoch=0, epochs=EPOCHS, batch_size=64,
               validation_data=(Xv, Yv), shuffle=True, verbose=0,
               callbacks=self._callbacks(d2, log2, rec2, session="s0",
                                         lr_recorder=_LRRecorder()) + [control])

        for epoch in range(INTERRUPT_AFTER, EPOCHS):
            self.assertEqual(
                second.traces.get(epoch), control.traces.get(epoch),
                f"epoch {epoch}: the resumed batch sequence differs from the uninterrupted "
                "control. On this Keras version the shuffle permutation is NOT reproducible "
                "across a resume, so RECOVERY_EXACTNESS must move the shuffle entry back to "
                "`not_recovered` and interrupted arms must be reported as non-bit-exact "
                "continuations. The arms remain scientifically valid - the design declares "
                "same_training_shuffle_stream=False - but the claim must be downgraded.")

        resumed_final = [np.array(w) for w in m2.get_weights()]
        control_final = [np.array(w) for w in m3.get_weights()]
        for i, (x, y) in enumerate(zip(resumed_final, control_final)):
            np.testing.assert_array_equal(
                x, y, err_msg=(f"weight tensor {i} differs between a resumed run and an "
                               "uninterrupted control; the resume is not bit-exact"))

    def test_within_epoch_position_is_not_recovered(self):
        """
        The one thing still listed under `not_recovered`: no record is
        written for an epoch that did not finish, so a mid-epoch
        interruption redoes that epoch rather than resuming inside it.
        """
        d, log, rec = self._arm("partial")
        m = _model()
        cbs = self._callbacks(d, log, rec, session="s1", lr_recorder=_LRRecorder())

        class DieMidEpoch(keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                if batch == 1:
                    raise RuntimeError("killed mid-epoch")

        with self.assertRaises(RuntimeError):
            self._fit(m, initial_epoch=0, epochs=1, callbacks=cbs + [DieMidEpoch()])
        self.assertEqual(log.completed_epochs(), [],
                         "a partial epoch was recorded as completed")
        self.assertEqual(log.highest_contiguous(), 0, "the partial epoch would not be redone")


class _BatchTrace(keras.callbacks.Callback if HAVE_KERAS else object):
    """Per-epoch per-batch loss trace: a fingerprint of the batch order."""

    def __init__(self):
        super().__init__()
        self.traces: dict[int, tuple] = {}
        self._cur: list = []

    def on_epoch_begin(self, epoch, logs=None):
        self._cur = []

    def on_train_batch_end(self, batch, logs=None):
        self._cur.append(round(float((logs or {}).get("loss", 0.0)), 9))

    def on_epoch_end(self, epoch, logs=None):
        self.traces[epoch] = tuple(self._cur)


class TestBestCheckpointIsWrittenAtomically(unittest.TestCase):
    """
    `ModelCheckpoint` writes its target in place, so a kill part-way
    through a save leaves a TRUNCATED file at a path still called
    "best_val_loss.keras". Nothing downstream would notice: the
    record's hash is recomputed from whatever is on disk at the next
    epoch end, so the torn bytes would be hashed and reported as the
    arm's best-validation artifact, and the certificate would name a
    model that cannot be loaded.

    The fix is the discipline the committed checkpoint already uses: the
    saver writes an INFLIGHT path, which is fsynced and then promoted
    with os.replace. The previously promoted best stays in place and
    loadable until its replacement is whole.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._stores: dict[str, CommittedCheckpointStore] = {}
        self.X, self.Y, self.Xv, self.Yv = _data()

    def tearDown(self):
        self.tmp.cleanup()

    def _arm(self, name):
        d = self.dir / name
        d.mkdir(parents=True, exist_ok=True)
        log = EpochLog(d / "epochs.jsonl")
        rec = ArmRecoveryRecord(arm_id=name, block_id="block0", seed=7,
                                epochs_total=EPOCHS)
        self._stores[name] = CommittedCheckpointStore(
            arm_dir=d, arm_id=name, block_id="block0", seed=7, epochs_total=EPOCHS)
        return d, log, rec

    def _callbacks(self, d, log, rec, *, session="s1"):
        return build_callbacks(
            store=self._stores[Path(d).name], log=log, record=rec,
            record_path=d / "recovery.json", best_path=d / "best_val_loss.keras",
            session_id=session)

    def _train(self, d, log, rec, *, initial_epoch=0, epochs=2, session="s1"):
        store = self._stores[Path(d).name]
        manifest = store.manifest()
        model = (keras.models.load_model(store.slot_path(manifest.committed_slot),
                                         compile=True)
                 if manifest is not None else _model())
        model.fit(self.X, self.Y, initial_epoch=initial_epoch, epochs=epochs,
                  batch_size=16, validation_data=(self.Xv, self.Yv), shuffle=True,
                  verbose=0, callbacks=self._callbacks(d, log, rec, session=session))

    def test_the_promoted_best_checkpoint_loads(self):
        d, log, rec = self._arm("atomic_best")
        self._train(d, log, rec)
        best = d / "best_val_loss.keras"
        self.assertTrue(best.is_file(), "no best checkpoint was promoted")
        keras.models.load_model(best, compile=True)     # raises if torn

    def test_no_inflight_file_is_left_behind_after_a_clean_run(self):
        d, log, rec = self._arm("atomic_clean")
        self._train(d, log, rec)
        leftovers = sorted(p.name for p in d.glob("*.inflight.keras"))
        self.assertEqual(leftovers, [],
                         f"inflight best-checkpoint writes were not promoted: {leftovers}")

    def test_a_torn_inflight_write_does_not_replace_the_promoted_best(self):
        """
        The crash this closes: the bytes of a half-finished save are
        present on disk when the next session starts. The promoted best
        must survive it unchanged, and must still load.
        """
        d, log, rec = self._arm("atomic_torn")
        self._train(d, log, rec)
        best = d / "best_val_loss.keras"
        good = best.read_bytes()

        inflight = d / "best_val_loss.inflight.keras"
        inflight.write_bytes(good[: len(good) // 3])     # a truncated save
        self.assertTrue(inflight.is_file())

        # a new session starts, resuming where the manifest committed
        self._train(d, log, rec, initial_epoch=2, epochs=4, session="s2")

        self.assertFalse(inflight.is_file(), "the torn inflight write was not discarded")
        keras.models.load_model(best, compile=True)
        self.assertTrue(any("inflight" in n for n in rec.reconciliation_notes),
                        f"the discard was not recorded: {rec.reconciliation_notes}")

    def test_the_recorded_hash_is_the_hash_of_the_promoted_file(self):
        """
        The record must describe the artifact that exists, not an
        intermediate one.
        """
        d, log, rec = self._arm("atomic_hash")
        self._train(d, log, rec)
        best = d / "best_val_loss.keras"
        self.assertEqual(rec.best_checkpoint_sha256, sha256_file(best))
        self.assertEqual(Path(rec.best_checkpoint), best)

    def test_a_torn_promoted_best_is_still_reported_honestly(self):
        """
        Scope limit, stated as a test. Promotion protects the file from
        a kill DURING a save; it cannot repair a file damaged after a
        successful promotion (bit rot, an external truncation). In that
        case the hash recorded at the next epoch end is the hash of the
        damaged file - the artifact is wrong but the record is not
        inconsistent with it, so the damage is detectable by anyone who
        tries to load it rather than hidden behind a stale hash.
        """
        d, log, rec = self._arm("atomic_rot")
        self._train(d, log, rec)
        best = d / "best_val_loss.keras"
        best.write_bytes(best.read_bytes()[:50])
        self.assertNotEqual(rec.best_checkpoint_sha256, sha256_file(best),
                            "external damage after promotion must not match the record")
        with self.assertRaises(Exception):
            keras.models.load_model(best, compile=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
