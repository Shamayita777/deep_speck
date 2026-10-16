#!/usr/bin/env python3
"""
Search-only entry point: AutoND's PART 1, without PART 2.

WHY THIS FILE EXISTS. The documented way to run AutoND is `python main.py`,
which runs the search and then immediately trains neural distinguishers on
every difference it found. There is no upstream command that runs the
search alone. This module provides one, by calling the SAME upstream
function with the SAME arguments and then stopping.

IT DOES NOT REIMPLEMENT THE SEARCH. The search is
`optimizer.optimize`, executed from the pinned checkout. Nothing here
reimplements, approximates or substitutes any part of it.

----------------------------------------------------------------------
WHAT IS MIRRORED, AND WHY MIRRORING RATHER THAN CALLING
----------------------------------------------------------------------
Upstream's search entry point is `main.findGoodInputDifferences`, which
at the pinned commit reads, verbatim (main.py:35-43):

    def findGoodInputDifferences(cipher_name, scenario, output_dir, epsilon = 0.1):
        cipher = importlib.import_module('ciphers.' + cipher_name, package='ciphers')
        s = cipher.__name__[8:] + "_" + scenario
        plain_bits = cipher.plain_bits
        key_bits = cipher.key_bits
        word_size = cipher.word_size
        encryption_function = cipher.encrypt
        best_differences, highest_round = optimizer.optimize(
            plain_bits, key_bits, encryption_function, scenario = scenario,
            log_file=f'{output_dir}/{s}', epsilon = epsilon)
        return best_differences, highest_round

That function cannot be called directly, because it lives in `main.py`,
whose module-level imports are:

    main.py:8   import optimizer
    main.py:9   import train_nets          <-- and train_nets.py:10 is
    train_nets.py:10   import tensorflow as tf      `import tensorflow`

so `import main` loads TensorFlow before a single line of the search
runs. `findGoodInputDifferences` itself never touches TensorFlow; only
its module neighbour does.

`resolve_search_arguments` below therefore reproduces those eight lines
and nothing else, so that the call made to `optimizer.optimize` is
argument-for-argument the call `main.py` would have made. `word_size` is
read and discarded exactly as upstream does - it is unused there too, and
it is kept here so the mirror is complete rather than tidied.

NOTE WHAT UPSTREAM DOES NOT PASS. `findGoodInputDifferences` does not
pass `nb_samples`, so `optimize` uses its import-time default of
`NUM_SAMPLES = 10**3`. Variant R1 reproduces that exactly, passing no
`nb_samples` at all. Supplying `nb_samples=10**4` is a DEVIATION from
what the released code does when run as documented, and R2 records it as
one.

----------------------------------------------------------------------
THE TRAINING BOUNDARY
----------------------------------------------------------------------
This module imports `optimizer` and one cipher module. It does not
import `main`, `train_nets`, `gohrnet`, `dbitnet`, `polish_nets`,
`eval_nets`, TensorFlow or Keras, and `assert_no_training_imports()`
re-checks that at runtime. `tests/test_no_training_import.py` proves it
both statically, by walking the real import graph of the pinned source
with `ast`, and dynamically, in a subprocess that binds the real
Speck32/64 cipher. Neither proof uses a fixture cipher.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

#: Modules that mean neural-network training has been pulled in. The
#: search needs none of them, and their presence means the search-only
#: boundary has been crossed.
TRAINING_MODULES = ("tensorflow", "keras", "train_nets", "polish_nets",
                    "gohrnet", "dbitnet", "eval_nets", "main")


class TrainingBoundaryViolation(RuntimeError):
    """Raised when a training module reached a search-only process."""


def assert_no_training_imports(stage: str) -> dict:
    """
    Fail closed if any training module is loaded. Returns the
    observation so it can be recorded in the run artifact; a run that
    cannot show this is a run whose 'no training' claim is unsupported.
    """
    import sys
    present = sorted(m for m in TRAINING_MODULES if m in sys.modules)
    if present:
        raise TrainingBoundaryViolation(
            f"{stage}: {present} present in sys.modules. This entry point runs "
            "AutoND's search only; a training module here means the boundary was "
            "crossed and the run's 'search only, no GPU' claim no longer follows "
            "from what executed. Refusing.")
    return {"stage": stage, "training_modules_present": [],
            "checked": list(TRAINING_MODULES)}


@dataclass(frozen=True)
class SearchArguments:
    """Exactly the arguments `main.findGoodInputDifferences` would build."""
    cipher_module: Any
    plain_bits: int
    key_bits: int
    word_size: int
    encryption_function: Callable
    scenario: str
    log_file: str
    epsilon: float
    #: None reproduces upstream, which does not pass nb_samples at all.
    nb_samples: Optional[int]

    def as_call_kwargs(self) -> dict:
        kwargs = {"scenario": self.scenario, "log_file": self.log_file,
                  "epsilon": self.epsilon}
        if self.nb_samples is not None:
            kwargs["nb_samples"] = self.nb_samples
        return kwargs

    def describe(self) -> dict:
        return {
            "cipher_module": self.cipher_module.__name__,
            "plain_bits": self.plain_bits, "key_bits": self.key_bits,
            "word_size_read_and_unused_as_upstream_does": self.word_size,
            "scenario": self.scenario, "log_file": self.log_file,
            "epsilon": self.epsilon,
            "nb_samples_passed": self.nb_samples,
            "nb_samples_note": (
                "not passed; optimize() uses its import-time default "
                "NUM_SAMPLES = 10**3, exactly as main.findGoodInputDifferences "
                "leaves it" if self.nb_samples is None else
                f"passed explicitly as {self.nb_samples}; upstream passes nothing "
                "here, so this is a recorded deviation"),
        }


def resolve_search_arguments(cipher_name: str, scenario: str, output_dir: str,
                             epsilon: float = 0.1,
                             nb_samples: Optional[int] = None,
                             cipher_module: Any = None) -> SearchArguments:
    """
    Mirror of main.findGoodInputDifferences' argument construction.

    `s` and therefore the log-file prefix are built by upstream's own
    expression `cipher.__name__[8:] + "_" + scenario`, which strips the
    leading "ciphers." - so for speck3264 single-key the prefix is
    `<output_dir>/speck3264_single-key`, matching the README's
    `results/speck3264_single-key`.
    """
    #: `cipher_module` bypasses the import for logging/serialisation tests
    #: that must not touch a real cipher. Every reproduction path leaves it
    #: None, so the real runs resolve the cipher exactly as upstream does.
    cipher = (cipher_module if cipher_module is not None
              else importlib.import_module("ciphers." + cipher_name, package="ciphers"))
    s = (cipher.__name__[8:] if cipher.__name__.startswith("ciphers.")
         else cipher.__name__) + "_" + scenario
    return SearchArguments(
        cipher_module=cipher,
        plain_bits=cipher.plain_bits,
        key_bits=cipher.key_bits,
        word_size=cipher.word_size,
        encryption_function=cipher.encrypt,
        scenario=scenario,
        log_file=f"{output_dir}/{s}",
        epsilon=epsilon,
        nb_samples=nb_samples,
    )


def run_search(optimizer, args: SearchArguments):
    """
    Invoke the real upstream search and stop.

    The boundary is asserted immediately before and immediately after the
    call, so a violation cannot be introduced by the search itself or by
    anything the harness does around it.
    """
    assert_no_training_imports("before optimizer.optimize")
    result = optimizer.optimize(
        args.plain_bits, args.key_bits, args.encryption_function,
        **args.as_call_kwargs())
    assert_no_training_imports("after optimizer.optimize")
    return result
