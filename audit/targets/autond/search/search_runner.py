#!/usr/bin/env python3
"""
Runner for AutoND's evolutionary input-difference search.

WHAT THIS IS. A harness that executes the SEARCH in AutoND's pinned
`optimizer.py` without modifying one byte of the upstream checkout, and
records enough about the execution that the run can be defended later:
every configuration value, every patch applied, every scoring call, the
RNG state where it is controllable, the stopping condition, elapsed time
and the returned differences.

WHAT THIS IS NOT. It does not train a neural network, it does not touch
the shipped checkpoints, and it does not reproduce AutoND's training
pipeline. The search and the training are separate stages of the target's
pipeline and this harness runs only the first.

----------------------------------------------------------------------
FOUR FACTS ESTABLISHED FROM THE PINNED SOURCE, 2026-10-14
----------------------------------------------------------------------

1. THE SEARCH NEEDS NO GPU AND NO TENSORFLOW. `optimizer.py` imports
   numpy, os, random, sys, copy, types, timeit and pandas - nothing else.
   It does NOT import `main`, and therefore does not transitively import
   `train_nets` or TensorFlow. Importing `optimizer` plus
   `ciphers.speck3264` leaves `tensorflow`, `keras` and `main` absent
   from `sys.modules`. `verify_no_tensorflow()` below asserts this at
   runtime, so the claim is re-checked on the machine that runs it rather
   than taken from this docstring.

2. NUMPY 1.26.x IS MANDATORY, NOT PREFERRED. `ciphers/speck3264.py`
   calls `check_testvectors()` at import time, and under NumPy 2 the
   expression `X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]` raises
   `OverflowError: Python integer 32768 out of bounds for uint8` because
   of NEP 50 integer promotion. The cipher module cannot be imported at
   all on NumPy 2. This is a property of the target, not a choice.

3. THE PAPER'S GENERATION COUNT CANNOT BE SET BY ASSIGNING THE MODULE
   CONSTANT. `optimizer.evo` is declared `def evo(f, n=NUM_GENERATIONS,
   ...)`, so `n` is bound to 5 when the module is imported. Assigning
   `optimizer.NUM_GENERATIONS = 50` afterwards leaves `evo.__defaults__`
   at 5, and `optimize()` calls `evo(...)` WITHOUT passing `n`. A
   programmatic override is therefore silently ignored. Reaching the
   paper's 50 generations requires either editing the upstream file - which
   this harness will not do - or rebinding the default, which it does
   explicitly and records as patch `P1-GENERATIONS`.

4. `nb_samples` HAS THE SAME DEFAULT-BINDING PROPERTY but is a real
   parameter of `optimize()`, so it is passed explicitly. No patch needed.

----------------------------------------------------------------------
RANDOMNESS, AND THE LIMIT OF REPRODUCIBILITY
----------------------------------------------------------------------
The search draws from two independent sources:

  * `numpy.random` - the evolutionary operators: initial population,
    mutation selection, mutated bit positions. Seedable.
  * `os.urandom` - the plaintexts and keys sampled each round inside
    `optimize()`. NOT seedable without substituting the function.

Seeding numpy therefore makes the search PARTIALLY reproducible and no
more. A run recorded with `numpy_seed` is not bit-reproducible, and this
harness never claims otherwise. For a fully deterministic run the
optional patch `P4-URANDOM` substitutes a seeded generator for
`optimizer.urandom`; it is OFF by default and is a DEVIATION from the
target's behaviour, recorded as such in the result.

----------------------------------------------------------------------
PAPER VERSUS SHIPPED CODE: THE POPULATION-INITIALISATION DISCREPANCY
----------------------------------------------------------------------
`optimize()` contains both variants, one commented out, with the
repository's own note:

    #diffs, scores = evo(f=..., gen=None)
    # The initial set of differences can be set to None, or to the
    # differences returned for the previous round. We use the second
    # option here, as opposed to the first (above) in the paper.
    diffs, scores = evo(f=..., gen=diffs, verbose=1)

So the SHIPPED code seeds each round's population with the previous
round's survivors; the PAPER starts each round from a fresh random
population. `population_init="previous"` runs the shipped behaviour with
no patch to the loop. `population_init="fresh"` applies patch
`P2-POPULATION-FRESH`, which forces `gen=None` on entry to `evo` and so
reproduces the paper's described procedure. Neither is "correct" - they
are two different procedures, and which one a result describes must be
stated with the result.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

RUNNER_VERSION = "autond-search-runner-v1"
PINNED_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"

#: Files whose content defines the search. Hashed into every result.
SEARCH_SOURCE_FILES = ("optimizer.py",)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class SearchRunnerError(RuntimeError):
    """Raised instead of proceeding with an unsound configuration."""


@dataclass
class Patch:
    """
    One declared modification to the imported upstream module.

    `changes_behaviour` separates a patch that alters what the search
    computes from one that only observes it. A reader must be able to
    tell at a glance which results are the target's own behaviour.
    """
    patch_id: str
    target: str
    reason: str
    before: str
    after: str
    changes_behaviour: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceBinding:
    """The pinned upstream source this run executed against."""
    autond_root: str
    head_commit: Optional[str]
    worktree_dirty: Optional[list]
    source_hashes: dict
    cipher_module: str
    cipher_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def verify_no_tensorflow() -> dict:
    """
    Re-check fact 1 on the machine that is running. Returns the
    observation; raises only if TensorFlow actually got imported, which
    would mean the search path is not what source inspection said.
    """
    observed = {m: (m in sys.modules) for m in ("tensorflow", "keras", "main", "torch")}
    if observed["tensorflow"] or observed["keras"]:
        raise SearchRunnerError(
            "TensorFlow or Keras was imported by the search path. Source inspection "
            "says optimizer.py imports only numpy and pandas, so something else in "
            f"this process pulled it in: {observed}. Refusing, because the "
            "'no GPU required' claim would no longer follow from what ran.")
    return observed


def read_head_commit(root: Path) -> Optional[str]:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text().strip()
    if ref.startswith("ref:"):
        p = root / ".git" / ref.split(" ", 1)[1]
        return p.read_text().strip() if p.is_file() else None
    return ref


def bind_source(autond_root: Path, cipher_name: str) -> tuple[Any, Any, SourceBinding]:
    """
    Import the pinned search code and the cipher, read-only.

    Nothing is written into `autond_root`. Python may create
    `__pycache__` there as a side effect of importing; that is bytecode,
    not a modification of the source, and `run_search.py --no-bytecode`
    suppresses it.
    """
    autond_root = autond_root.expanduser().resolve()
    if not (autond_root / "optimizer.py").is_file():
        raise SearchRunnerError(
            f"{autond_root} does not contain optimizer.py. Point --autond-root at the "
            f"pinned AutoND checkout ({PINNED_COMMIT}).")

    import numpy as np
    if not np.__version__.startswith("1.26"):
        raise SearchRunnerError(
            f"numpy {np.__version__} is installed. AutoND's ciphers/speck3264.py runs "
            "check_testvectors() at import time and raises OverflowError under NumPy 2 "
            "(NEP 50 integer promotion), so the cipher module cannot be imported at "
            "all. Install numpy==1.26.4.")

    sys.path.insert(0, str(autond_root))
    optimizer = importlib.import_module("optimizer")
    cipher = importlib.import_module(f"ciphers.{cipher_name}")
    verify_no_tensorflow()

    head = read_head_commit(autond_root)
    binding = SourceBinding(
        autond_root=str(autond_root),
        head_commit=head,
        worktree_dirty=None,
        source_hashes={f: sha256_file(autond_root / f) for f in SEARCH_SOURCE_FILES
                       if (autond_root / f).is_file()},
        cipher_module=f"ciphers.{cipher_name}",
        cipher_hash=sha256_file(autond_root / "ciphers" / f"{cipher_name}.py"),
    )
    if head and head != PINNED_COMMIT:
        raise SearchRunnerError(
            f"the checkout at {autond_root} is at {head}, not the pinned "
            f"{PINNED_COMMIT}. Refusing: a search run must name the revision it ran "
            "against, and silently accepting another one would make the record wrong.")
    return optimizer, cipher, binding


class GenerationLog:
    """Append-only JSONL record of every scoring call the search makes."""

    def __init__(self, path: Path):
        self.path = path
        self.records = 0
        self._fh = path.open("a", encoding="utf-8")

    def write(self, row: dict) -> None:
        row = {"seq": self.records, "at_utc": utc_now(), **row}
        self._fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.records += 1

    def close(self) -> None:
        self._fh.close()


def _install_generation_patch(optimizer, num_generations: int, patches: list) -> None:
    """Rebind evo's import-time `n` default. See fact 3 in the module docstring."""
    sig = inspect.signature(optimizer.evo)
    params = list(sig.parameters)
    if params[1] != "n":
        raise SearchRunnerError(
            f"optimizer.evo's second parameter is {params[1]!r}, not 'n'. The "
            "generation-count patch rebinds a positional default and must not be "
            "applied to a signature it was not written for.")
    before_n = optimizer.evo.__defaults__[0]
    new_defaults = list(optimizer.evo.__defaults__)
    new_defaults[0] = num_generations
    optimizer.evo.__defaults__ = tuple(new_defaults)
    patches.append(Patch(
        patch_id="P1-GENERATIONS",
        target="optimizer.evo.__defaults__[0]  (parameter 'n')",
        reason=("optimize() calls evo() without passing n, and evo's default was bound "
                "to NUM_GENERATIONS=5 at import. Assigning optimizer.NUM_GENERATIONS "
                "does NOT change it, so the paper's 50 generations is unreachable "
                "without rebinding this default or editing upstream."),
        before=repr(before_n), after=repr(num_generations),
        changes_behaviour=True))


def install_patches(optimizer, *, num_generations: Optional[int], population_init: str,
                    log: GenerationLog, urandom_seed: Optional[int]) -> tuple[list, Callable]:
    """
    Apply the declared patches and return (patches, restore).

    Every patch is recorded with what it replaced. `restore()` puts the
    module back, so a test can assert the upstream module is unchanged
    after a run.
    """
    patches: list[Patch] = []
    originals: dict[str, Any] = {}

    # ---- P1: the paper's generation count --------------------------------
    # num_generations=None means DO NOT PATCH: the search runs at the
    # import-time default the released code ships. Variant R1 uses this,
    # so it applies no behaviour-changing patch at all.
    originals["evo_defaults"] = optimizer.evo.__defaults__
    if num_generations is None:
        patches.append(Patch(
            patch_id="P1-GENERATIONS-NOT-APPLIED",
            target="optimizer.evo.__defaults__[0]  (parameter 'n')",
            reason=("variant R1 reproduces the released code as documented, so the "
                    "generation count is left at the value optimizer.py ships."),
            before=repr(optimizer.evo.__defaults__[0]),
            after=repr(optimizer.evo.__defaults__[0]) + "  (unchanged)",
            changes_behaviour=False))
    else:
        _install_generation_patch(optimizer, num_generations, patches)

    # ---- P2: population initialisation -----------------------------------

    originals["evo"] = optimizer.evo
    real_evo = optimizer.evo
    force_fresh = (population_init == "fresh")

    def evo_wrapper(f, n=None, num_bits=32, L=32, gen=None, verbose=0):
        call_n = real_evo.__defaults__[0] if n is None else n
        incoming_population = None if gen is None else int(len(gen))
        if force_fresh:
            gen = None
        t0 = time.perf_counter()
        log.write({"event": "evo_enter", "generations": call_n, "num_bits": num_bits,
                   "L": L, "incoming_population": incoming_population,
                   "population_init": population_init,
                   "population_passed_to_evo": None if gen is None else int(len(gen))})
        result = real_evo(f, call_n, num_bits, L, gen, verbose)
        diffs, scores = result
        log.write({"event": "evo_exit", "elapsed_seconds": time.perf_counter() - t0,
                   "returned_differences": int(len(diffs)),
                   "score_min": float(min(scores)), "score_max": float(max(scores))})
        return result

    #: Standard attribute so a caller - or a test - can reach the function
    #: that actually runs and confirm P1 was applied to IT, rather than to
    #: this wrapper, whose own `n` default is None by construction.
    evo_wrapper.__wrapped__ = real_evo
    optimizer.evo = evo_wrapper
    patches.append(Patch(
        patch_id=("P2-POPULATION-FRESH" if force_fresh else "P2-POPULATION-OBSERVE"),
        target="optimizer.evo",
        reason=("the paper initialises each round's population freshly; the shipped "
                "code passes the previous round's survivors (gen=diffs). 'fresh' "
                "forces gen=None to follow the paper; 'previous' leaves the shipped "
                "behaviour untouched and only logs it."),
        before="the module's own evo", after="logging wrapper" + (
            " that forces gen=None" if force_fresh else " (pass-through)"),
        changes_behaviour=force_fresh))

    # ---- P3: per-scoring-call observation (behaviour-neutral) ------------
    originals["evaluate"] = optimizer.evaluate_multiple_differences
    real_eval = optimizer.evaluate_multiple_differences

    def eval_wrapper(candidate_differences, pt0, keys, C0, nr, plain_bits, key_bits,
                     encrypt, scenario="single-key"):
        t0 = time.perf_counter()
        scores = real_eval(candidate_differences, pt0, keys, C0, nr, plain_bits,
                           key_bits, encrypt, scenario)
        log.write({"event": "score_batch", "round": int(nr),
                   "candidates": int(len(candidate_differences)),
                   "samples": int(len(pt0)),
                   "elapsed_seconds": time.perf_counter() - t0,
                   "score_max": float(scores.max()) if len(scores) else None,
                   "score_mean": float(scores.mean()) if len(scores) else None})
        return scores

    optimizer.evaluate_multiple_differences = eval_wrapper
    patches.append(Patch(
        patch_id="P3-SCORING-OBSERVE", target="optimizer.evaluate_multiple_differences",
        reason="per-batch timing and score statistics; return value passed through "
               "unchanged so the search computes exactly what it would have computed.",
        before="the module's own evaluate_multiple_differences",
        after="logging wrapper (pass-through)", changes_behaviour=False))

    # ---- P4: optional deterministic urandom ------------------------------
    if urandom_seed is not None:
        import numpy as np
        originals["urandom"] = optimizer.urandom
        rng = np.random.default_rng(urandom_seed)

        def seeded_urandom(n: int) -> bytes:
            return rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()

        optimizer.urandom = seeded_urandom
        patches.append(Patch(
            patch_id="P4-URANDOM", target="optimizer.urandom",
            reason=("DEVIATION, enabled only by --urandom-seed. Substitutes a seeded "
                    "generator for the operating-system entropy the target uses to "
                    "sample plaintexts and keys. It makes a run bit-reproducible and "
                    "it is NOT the target's behaviour; a result produced with this "
                    "patch must say so."),
            before="os.urandom", after=f"numpy default_rng({urandom_seed})",
            changes_behaviour=True))

    def restore() -> None:
        optimizer.evo = originals["evo"]
        optimizer.evo.__defaults__ = originals["evo_defaults"]
        optimizer.evaluate_multiple_differences = originals["evaluate"]
        if "urandom" in originals:
            optimizer.urandom = originals["urandom"]

    return patches, restore


def run_once(cfg: dict, optimizer, cipher, binding: SourceBinding,
             out_dir: Path, run_index: int) -> dict:
    """Execute one complete search run. Returns the result record."""
    import numpy as np

    run_dir = out_dir / f"run_{run_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = GenerationLog(run_dir / "generations.jsonl")

    numpy_seed = cfg.get("numpy_seed")
    if numpy_seed is not None:
        np.random.seed(numpy_seed + run_index)
    numpy_state_before = np.random.get_state()

    patches, restore = install_patches(
        optimizer,
        num_generations=cfg["num_generations"],
        population_init=cfg["population_init"],
        log=log,
        urandom_seed=(None if cfg.get("urandom_seed") is None
                      else cfg["urandom_seed"] + run_index))

    rec: dict[str, Any] = {
        "runner": RUNNER_VERSION,
        "run_index": run_index,
        "started_at_utc": utc_now(),
        "config": dict(cfg),
        "source": binding.to_dict(),
        "patches": [p.to_dict() for p in patches],
        "numpy_seed_used": None if numpy_seed is None else numpy_seed + run_index,
        "numpy_state_sha256": hashlib.sha256(
            np.asarray(numpy_state_before[1]).tobytes()).hexdigest(),
        "reproducibility": {
            "numpy_random": "seeded" if numpy_seed is not None else "unseeded",
            "os_urandom": ("substituted by P4-URANDOM (deviation)"
                           if cfg.get("urandom_seed") is not None
                           else "operating-system entropy; NOT reproducible"),
            "bit_reproducible": cfg.get("urandom_seed") is not None
                                and numpy_seed is not None,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "tensorflow_observed": verify_no_tensorflow(),
    }

    import search_only
    args = search_only.resolve_search_arguments(
        cipher_name=cfg["cipher"], scenario=cfg["scenario"],
        output_dir=str(run_dir), epsilon=cfg["epsilon"],
        nb_samples=cfg.get("num_samples"),
        cipher_module=(cipher if cfg.get("_cipher_module_supplied") else None))
    rec["search_arguments"] = args.describe()
    rec["training_boundary"] = search_only.assert_no_training_imports("before run")
    t0 = time.perf_counter()
    try:
        diffs_hex, highest_round = search_only.run_search(optimizer, args)
        rec["status"] = "COMPLETED"
        rec["returned_differences_hex"] = [str(d) for d in diffs_hex]
        rec["highest_non_random_round"] = int(highest_round)
    except BaseException as exc:                                    # noqa: BLE001
        rec["status"] = "FAILED"
        rec["exception"] = {"type": type(exc).__name__, "message": str(exc),
                            "traceback": traceback.format_exc()}
        raise
    finally:
        rec["elapsed_seconds"] = time.perf_counter() - t0
        rec["ended_at_utc"] = utc_now()
        rec["scoring_calls_logged"] = log.records
        restore()
        log.close()
        (run_dir / "result.json").write_text(
            json.dumps(rec, indent=2, sort_keys=True, default=str))
    return rec


def write_manifest(out_dir: Path, extra: dict) -> Path:
    """Hash every file produced, so the evidence bundle is self-checking."""
    files = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files[str(p.relative_to(out_dir))] = {
                "sha256": sha256_file(p), "bytes": p.stat().st_size}
    manifest = {"runner": RUNNER_VERSION, "written_at_utc": utc_now(),
                "pinned_commit": PINNED_COMMIT, "files": files, **extra}
    path = out_dir / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return path
