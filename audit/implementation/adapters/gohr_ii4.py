"""
Gohr/Speck32/64 adapter for the generic II-4 conformance-impact runner.

LOCATION: audit/implementation/adapters/ - II-4 is an Implementation
Integrity experiment, not an Experimental Validity one. It REUSES the
Gohr primitives in experimental_validity/gohr/ (model, speck, dataset,
train, evaluate), which were verified layer-config identical to the
reference train_nets.py at depths 5 and 10 and call-for-call equivalent
in their fit semantics; it does not import EV's experiment logic.

ALL Gohr-specific knowledge for II-4 lives here: Speck data generation,
the residual network, the reference training recipe, and how to read a
network's REALIZED residual depth out of an artifact. The generic runner
(audit/implementation/ii4_runner.py) never imports this module's
internals; it talks to it only through the ConformanceAdapter contract.
The generic runner never imports this module's internals; the dependency
points from adapter to runner contract only.

REFERENCE CONFORMANCE
---------------------
Every training hyperparameter is taken from the reference
`train_nets.py::train_speck_distinguisher`, including one that the EV
adapter currently omits: the reference builds the network with
`make_resnet(depth=depth, reg_param=10**-5)`. The EV adapter calls
`make_resnet(depth=config.depth)` and therefore silently uses the
function default reg_param=1e-4 - ten times the reference's L2 weight.
This adapter passes reg_param explicitly from the frozen protocol.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from audit.common.provenance import sha256_file
from audit.experimental_validity.gohr import dataset as gohr_dataset
from audit.experimental_validity.gohr import evaluate as gohr_evaluate
from audit.experimental_validity.gohr import model as gohr_model
from audit.experimental_validity.gohr import train as gohr_train


def count_residual_merges_in_config(model_config: dict) -> int:
    """Realized depth = number of residual `Add` merges in a Keras config."""
    layers = model_config.get("config", {}).get("layers", [])
    return sum(1 for layer in layers if layer.get("class_name") == "Add")


# Files whose content determines the Gohr instantiation of II-4. Paths are
# repository-relative; every one must exist or the runner refuses to start.
GOHR_II4_PROVENANCE_SOURCES = {
    "adapter": "audit/implementation/adapters/gohr_ii4.py",
    "ev_gohr_model": "audit/experimental_validity/gohr/model.py",
    "ev_gohr_speck": "audit/experimental_validity/gohr/speck.py",
    "ev_gohr_dataset": "audit/experimental_validity/gohr/dataset.py",
    "ev_gohr_train": "audit/experimental_validity/gohr/train.py",
    "ev_gohr_evaluate": "audit/experimental_validity/gohr/evaluate.py",
    "reference_speck": "speck.py",
    "reference_train_nets": "train_nets.py",
    "reference_train_5_rounds": "train_5_rounds.py",
    "reference_copy_speck": "audit/implementation/reference/speck.py",
    "reference_copy_train_nets": "audit/implementation/reference/train_nets.py",
    "reference_copy_train_5_rounds": "audit/implementation/reference/train_5_rounds.py",
}


@dataclass
class GohrII4Adapter:
    """
    Implements the generic ConformanceAdapter contract for the Gohr
    distinguisher. `protocol` is the frozen fixed-protocol dict.
    """

    protocol: dict[str, Any]

    # ------------------------------------------------------------------
    # provenance
    # ------------------------------------------------------------------
    def provenance_sources(self) -> dict[str, str]:
        return dict(GOHR_II4_PROVENANCE_SOURCES)

    def environment_details(self) -> dict[str, Any]:
        """
        Framework/hardware details the generic capture_environment() cannot
        know: TensorFlow/Keras build, CUDA/cuDNN, visible GPUs, driver.
        Every field is reported as observed; absent information is None,
        never guessed.
        """
        import subprocess

        out: dict[str, Any] = {}
        try:
            import tensorflow as tf
            build = dict(tf.sysconfig.get_build_info())
            out["tensorflow_version"] = tf.__version__
            out["tf_cuda_version"] = build.get("cuda_version")
            out["tf_cudnn_version"] = build.get("cudnn_version")
            out["tf_is_cuda_build"] = build.get("is_cuda_build")
            out["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
        except Exception as exc:  # pragma: no cover - environment dependent
            out["tensorflow_error"] = repr(exc)
        try:
            import keras
            out["keras_version"] = keras.__version__
        except Exception:
            out["keras_version"] = None
        try:
            smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False)
            out["nvidia_smi"] = smi.stdout.strip() or None
        except Exception:
            out["nvidia_smi"] = None
        return out

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    def generate_sealed_test_set(self, directory: Path) -> tuple[Path, str, int]:
        bundle = gohr_dataset.generate_dataset(
            n=self.protocol["sealed_test_samples"], rounds=self.protocol["num_rounds"],
            differential=tuple(self.protocol["differential"]), role="confirmatory_test",
            dataset_id="II4-GLOBAL-SEALED-TEST",
        )
        path = gohr_dataset.persist_dataset(bundle, directory)
        return path, bundle.combined_hash, bundle.n

    def load_sealed_test_set(self, path: Path, expected_hash: str):
        return gohr_dataset.load_dataset(
            path, dataset_id="II4-GLOBAL-SEALED-TEST", role="confirmatory_test",
            rounds=self.protocol["num_rounds"],
            differential=tuple(self.protocol["differential"]),
            expected_combined_hash=expected_hash,
        )

    def generate_block_datasets(self, block_id: str, directory: Path) -> dict[str, dict]:
        out = {}
        for role, n_key in (("train", "train_samples"), ("validation", "validation_samples")):
            bundle = gohr_dataset.generate_dataset(
                n=self.protocol[n_key], rounds=self.protocol["num_rounds"],
                differential=tuple(self.protocol["differential"]), role=role,
                dataset_id=f"II4-{block_id}-{role}",
            )
            path = gohr_dataset.persist_dataset(bundle, directory)
            out[role] = {"path": str(path), "hash": bundle.combined_hash, "n": bundle.n,
                         "dataset_id": bundle.dataset_id}
        return out

    def load_block_datasets(self, records: dict[str, dict]):
        return {
            role: gohr_dataset.load_dataset(
                rec["path"], dataset_id=rec["dataset_id"], role=role,
                rounds=self.protocol["num_rounds"],
                differential=tuple(self.protocol["differential"]),
                expected_combined_hash=rec["hash"],
            )
            for role, rec in records.items()
        }

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    def build_model(self, factor_value: int, *, seed: int):
        # Seed MUST precede construction: weight initialization draws from
        # the RNG state at build time.
        gohr_train.set_seed(seed)
        return gohr_model.make_resnet(depth=int(factor_value),
                                      reg_param=float(self.protocol["reg_param"]))

    def probe_model(self, model) -> int:
        """Realized depth of an IN-MEMORY model (counts Add layers)."""
        return sum(1 for layer in model.layers if layer.__class__.__name__ == "Add")

    def probe_realized_value(self, artifact_path: str) -> int:
        """
        Realized depth read from a SAVED artifact on disk, independently of
        the in-memory object and of the requested configuration.
        """
        with zipfile.ZipFile(artifact_path) as zf:
            config = json.loads(zf.read("config.json"))
        return count_residual_merges_in_config(config)

    # ------------------------------------------------------------------
    # training / evaluation
    # ------------------------------------------------------------------
    def train(self, model, block_data, *, checkpoint_path: str):
        train, val = block_data["train"], block_data["validation"]
        return gohr_train.train_model(
            model, train.X, train.Y, val.X, val.Y,
            epochs=int(self.protocol["epochs"]),
            batch_size=int(self.protocol["batch_size"]),
            shuffle=bool(self.protocol["shuffle"]),
            lr_high=float(self.protocol["lr_high"]),
            lr_low=float(self.protocol["lr_low"]),
            lr_period=int(self.protocol["lr_period"]),
            optimizer=str(self.protocol["optimizer"]),
            checkpoint_path=checkpoint_path,
        )

    def save_terminal_model(self, model, path: Path) -> dict[str, Any]:
        """
        Persist the TERMINAL-EPOCH model and verify the persisted artifact is
        weight-identical to the in-memory model. This is what makes
        'evaluate the saved terminal model' equivalent to 'evaluate the final
        in-memory model' - and is checked, not assumed.
        """
        import keras

        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        reloaded = keras.models.load_model(str(path), compile=False)
        original, restored = model.get_weights(), reloaded.get_weights()
        identical = len(original) == len(restored) and all(
            np.array_equal(a, b) for a, b in zip(original, restored))
        return {"path": str(path), "hash": sha256_file(path),
                "reload_weight_identical": bool(identical)}

    def load_terminal_model(self, path: str, expected_hash: str):
        import keras

        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(
                f"Terminal model {path} hash {actual} != recorded {expected_hash}; refusing "
                "to evaluate a substituted or corrupted model.")
        return keras.models.load_model(path, compile=False)

    def evaluate_terminal(self, model, sealed_test) -> float:
        return float(gohr_evaluate.evaluate_model(model, sealed_test.X, sealed_test.Y).accuracy)
