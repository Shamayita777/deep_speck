"""AutoND dataset adapter for the dataset-integrity pillar.

AutoND is the SECOND audited target. This adapter exists so the generic D0-D3
engines can analyse an AutoND dataset instance without any change to their
audit logic. It mirrors the surface `adapters/gohr.py` exposes for the
dataset-side stages and nothing more:

    feature_bits          -> int
    generate_partition()  -> (X, Y)
    dataset_provenance()  -> dict

The model/training half of the Gohr adapter (`build_model`, `train`,
`evaluate`, `predict`, `save`, `load`) is deliberately NOT implemented here.
D0-D3 do not need it, and D4/D5 for AutoND are a separately scoped experiment
whose interface fit has not been established. Declaring stubs that raise would
let a caller believe the training path exists; omitting them makes the absence
structural.

THE ONE STRUCTURAL DIFFERENCE FROM GOHR
---------------------------------------
Gohr's generator bakes the input difference into `speck.make_train_data(n, nr)`.
AutoND's generator is parameterised by the difference:

    main.make_train_data(encryption_function, plain_bits, key_bits,
                         n, nr, delta_state, delta_key)

So the difference is a CONSTRUCTOR argument here, and a dataset instance is
identified by (cipher, scenario, delta_plain, delta_key, num_rounds). Two
AutoND datasets that differ only in delta are different instances and must not
be compared as if they were one.

VERIFIED AGAINST THE PINNED SOURCE (not assumed)
------------------------------------------------
For speck3264 with delta = 0x400000, num_rounds = 8, n = 20000:
    plain_bits = 32, key_bits = 64, word_size = 16
    X: shape (n, 64), dtype uint8, values {0, 1}
    Y: shape (n,),    dtype uint8, values {0, 1}
    feature width 64 == 2 * plain_bits
    C = np.hstack([C0, C1]) -> C0 occupies columns 0..31, C1 columns 32..63
    class ratio ~0.504 at n=20000; per-bit means within [0.4910, 0.5077]
    two calls with identical arguments are NOT equal -> os.urandom

RANDOMNESS, STATED ACCURATELY
-----------------------------
`main.make_train_data` draws keys, plaintexts, the label vector and the
random-class substitution from `os.urandom`. There is NO seed parameter and no
seeded generator anywhere on that path, so an integer seed cannot reproduce an
AutoND dataset. `exact_replay_available` is therefore False and the canonical
instance must be persisted, not regenerated. This matches Gohr's situation and
is recorded rather than worked around.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np

#: Identifies the audited target for the verified SPECK32/64 case study. Kept
#: distinct from Gohr's "gohr-speck" so evidence for the two targets can never
#: be confused by dataset_id alone.
#:
#: This is the module-level constant for the case study. The per-instance
#: identifier is DERIVED from the cipher actually configured -- see
#: `AutoNDDatasetAdapter.dataset_id` -- because a FIXED identifier would label
#: SIMON data as SPECK data.
DATASET_ID = "autond-speck3264"

#: Ciphers whose representation has been verified against the pinned source.
#: Only speck3264 has been checked (feature width, dtype, value set, delta
#: encoding, class balance). Another cipher is not refused by the adapter --
#: the generator is generic -- but it is NOT a verified configuration, and the
#: experiment layer restricts itself to this set.
VERIFIED_CIPHERS = ("speck3264",)

#: The generation procedure this adapter invokes, named for the manifest.
GENERATOR_NAME = "main.make_train_data"

#: Published full-round counts, for reference only. They are NOT defaults:
#: num_rounds is always supplied explicitly, because an AutoND dataset at a
#: different round count is a different dataset instance.
NOMINAL_ROUNDS = {"speck3264": 22, "simon3264": 32}

SCENARIOS = ("single-key", "related-key")


#: Top-level module names the pinned checkout binds. `main.py` imports
#: `optimizer` and `train_nets` as absolute top-level modules, so they end up
#: in sys.modules under names the checkout does not own. They are guarded for
#: the same reason `main` is.
PINNED_TOP_LEVEL_MODULES = ("main", "optimizer", "train_nets",
                            "dbitnet", "gohrnet")

#: Checkout-relative files whose executable top-level code runs when the
#: generator is imported.
#:
#: DETERMINED, NOT ASSUMED. Obtained by importing the pinned `main` and
#: listing every entry in `sys.modules` whose `__file__` resolves inside the
#: checkout:
#:
#:     ciphers.speck3264   ciphers/speck3264.py
#:     dbitnet             dbitnet.py
#:     gohrnet             gohrnet.py
#:     main                main.py
#:     optimizer           optimizer.py
#:     train_nets          train_nets.py
#:
#: `main` imports `optimizer` and `train_nets`; `train_nets` imports `dbitnet`
#: and `gohrnet`. Earlier revisions of this adapter pinned only `main.py` and
#: the cipher, so FOUR modules' top-level code executed without ever being
#: checked against the pin. The cipher is configuration-dependent and is added
#: per instance, so it is not listed here.
PINNED_IMPORT_GRAPH_FILES = (
    "main.py", "optimizer.py", "train_nets.py", "dbitnet.py", "gohrnet.py",
)


class SourceBindingError(RuntimeError):
    """The executed source could not be shown to be the requested source.

    Distinct from FileNotFoundError (the checkout is absent) and ValueError
    (the configuration is invalid): this means an import would, or did, bind
    code from somewhere other than the checkout named.
    """


def _git_tracked_modifications(root: Path) -> list[str]:
    """Tracked files that differ from HEAD, or [] when none / not a repo.

    Untracked files are deliberately NOT reported here: they are handled by
    `_shadowing_candidates`, because an untracked file is not a modification
    of the approved revision but can still change which module is imported.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [line[3:].strip() for line in out.stdout.splitlines()
            if line.strip()]


def _shadowing_candidates(root: Path, cipher_name: str) -> list[str]:
    """Paths in the checkout that could change which module gets imported.

    Three vectors:

    * a DIRECTORY named like a pinned module, containing `__init__.py`.
      Packages take precedence over modules, so `optimizer/__init__.py`
      would be imported instead of `optimizer.py`.
    * `ciphers/__init__.py`. The pinned `ciphers` is an implicit NAMESPACE
      package; adding `__init__.py` makes it a regular package and changes
      the import semantics the post-import namespace check relies on.
    * any `.py` file at the checkout root or under `ciphers/` that git does
      not track. A clean tracked diff says nothing about those.
    """
    import subprocess
    found: list[str] = []

    for name in PINNED_TOP_LEVEL_MODULES:
        pkg = root / name
        if pkg.is_dir() and (pkg / "__init__.py").is_file():
            found.append(f"{name}/__init__.py (package shadows {name}.py)")

    if (root / "ciphers" / "__init__.py").is_file():
        found.append("ciphers/__init__.py (turns the pinned implicit "
                     "namespace package into a regular package)")

    try:
        out = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain",
             "--untracked-files=all"],
            capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if not line.startswith("??"):
                    continue
                rel = line[3:].strip()
                if not rel.endswith(".py"):
                    continue
                parts = Path(rel).parts
                if len(parts) == 1 or (len(parts) == 2
                                       and parts[0] == "ciphers"):
                    found.append(f"{rel} (untracked, on the import path)")
    except (OSError, subprocess.SubprocessError):
        pass
    return sorted(set(found))


def _git_head(root: Path) -> str | None:
    """The checkout's git HEAD, or None when it is not a git working tree."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _module_locations(mod: Any) -> list[Path]:
    """Every filesystem location a module draws code from.

    A normal module has one `__file__`. A namespace package has NO `__file__`
    and instead a `__path__` with one entry per contributing directory, so
    both have to be consulted or a foreign namespace portion goes unnoticed.
    """
    out: list[Path] = []
    f = getattr(mod, "__file__", None)
    if f:
        try:
            out.append(Path(f).resolve())
        except OSError:
            pass
    if not out:
        for entry in list(getattr(mod, "__path__", []) or []):
            try:
                out.append(Path(str(entry)).resolve())
            except OSError:
                pass
    return out


def _is_within(path: Path, root: Path) -> bool:
    """True if `path` is `root` or lies beneath it, after symlink resolution."""
    try:
        path = Path(path).resolve()
        root = Path(root).resolve()
    except OSError:
        return False
    return path == root or root in path.parents


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a large source file is fine."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class AutoNDDatasetAdapter:
    """Presents the pinned AutoND generator to the dataset-integrity engines.

    Parameters
    ----------
    autond_root:
        The pinned AutoND checkout. Imported, never modified: it is the
        audited subject.
    cipher_name:
        Module name under ``ciphers/``, e.g. ``speck3264``.
    delta_plain_int:
        Plaintext difference as an integer, e.g. ``0x400000``.
    num_rounds:
        Explicit round count. Required; there is no default, because a silent
        default would let two different dataset instances share a provenance
        record.
    delta_key_int:
        Key difference. Must be 0 for the single-key scenario.
    scenario:
        ``single-key`` or ``related-key``.
    """

    DATASET_ID = DATASET_ID
    GENERATOR_NAME = GENERATOR_NAME

    @property
    def dataset_id(self) -> str:
        """The identifier for THIS instance, derived from the cipher.

        A fixed identifier is a correctness bug once more than one cipher can
        be configured: data generated with SIMON32/64 would be recorded under
        `autond-speck3264`, and the audit artifacts would then disagree with
        the data they describe. For the verified speck3264 configuration this
        returns exactly `DATASET_ID`, so no existing instance changes.
        """
        return f"autond-{self.cipher_name}"

    @property
    def is_verified_configuration(self) -> bool:
        """Whether this cipher's representation has been verified."""
        return self.cipher_name in VERIFIED_CIPHERS

    def __init__(
        self,
        *,
        autond_root: Path | str,
        cipher_name: str = "speck3264",
        delta_plain_int: int,
        num_rounds: int,
        delta_key_int: int = 0,
        scenario: str = "single-key",
        expected_source_digests: dict[str, str] | None = None,
        expected_upstream_commit: str | None = None,
    ) -> None:
        # resolve() collapses symlinks and relative segments, so every later
        # containment test compares absolute, canonical paths.
        self.autond_root = Path(autond_root).expanduser().resolve()
        #: Optional pin. When given, it is enforced BEFORE the checkout's
        #: modules are imported, and re-checked after. Absent by default so
        #: the adapter stays reusable for controlled development; the
        #: production experiment always supplies it.
        self.expected_source_digests = dict(expected_source_digests or {})
        self.expected_upstream_commit = (
            str(expected_upstream_commit) if expected_upstream_commit else None)
        self.cipher_name = str(cipher_name)
        self.delta_plain_int = int(delta_plain_int)
        self.delta_key_int = int(delta_key_int)
        self.scenario = str(scenario)

        # bool is a subclass of int, so `num_rounds=True` would otherwise pass
        # the >= 1 test and silently run the generator at ONE round while the
        # manifest recorded `True`. A round count is the single most
        # instance-defining parameter here, so a boolean is refused rather
        # than coerced.
        if isinstance(num_rounds, bool):
            raise TypeError(
                f"num_rounds must be an integer, not a bool ({num_rounds!r}). "
                f"bool is a subclass of int, so True would silently mean one "
                f"round and mislabel the dataset instance.")
        if int(num_rounds) < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds!r}")
        self.num_rounds = int(num_rounds)

        if self.scenario not in SCENARIOS:
            raise ValueError(
                f"scenario must be one of {SCENARIOS}, got {self.scenario!r}")
        if self.scenario == "single-key" and self.delta_key_int != 0:
            raise ValueError(
                "the single-key scenario requires delta_key_int == 0; got "
                f"{self.delta_key_int!r}. A nonzero key difference is the "
                "related-key scenario and a different dataset instance.")

        self._main, self._cipher = self._import_pinned()

        self.plain_bits = int(self._cipher.plain_bits)
        self.key_bits = int(self._cipher.key_bits)
        self.word_size = int(self._cipher.word_size)

        if not 0 <= self.delta_plain_int < (1 << self.plain_bits):
            raise ValueError(
                f"delta_plain_int {hex(self.delta_plain_int)} does not fit in "
                f"{self.plain_bits} plaintext bits")
        if self.delta_plain_int == 0:
            raise ValueError(
                "delta_plain_int == 0 produces identical plaintext pairs, so "
                "the positive class carries no difference signal. Refusing: "
                "this is a configuration error, not a dataset to audit.")

        # Bit-vector forms, built by the TARGET's own helper so the ordering is
        # the source's and not this adapter's interpretation of it.
        self._delta_plain = self._main.integer_to_binary_array(
            self.delta_plain_int, self.plain_bits)
        self._delta_key: Any = 0
        if self.scenario == "related-key":
            self._delta_key = self._main.integer_to_binary_array(
                self.delta_key_int, self.key_bits)

    # -- import --------------------------------------------------------

    def _import_pinned(self):
        """Import the pinned checkout's `main` and cipher module, and BIND
        provenance to the files actually loaded.

        WHY THIS IS NOT JUST `sys.path.insert` + `import_module`
        -------------------------------------------------------
        `importlib.import_module` consults `sys.modules` FIRST. If anything in
        the process has already imported a module named `main`, `optimizer`,
        `train_nets`, `ciphers` or `ciphers.<cipher>` — another AutoND
        checkout, or simply a user script called `main.py` — the cached module
        is returned and the requested checkout is never read. Manipulating
        `sys.path` does nothing about that. The earlier version of this method
        then recorded digests of files under `autond_root`, producing
        provenance that named code which had not been executed.

        Two properties close that:

        1. A pre-existing module under any of those names that does NOT
           resolve inside the requested checkout is a hard refusal. It is NOT
           evicted from `sys.modules`: deleting another component's module is
           an unsafe side effect, and a half-initialised duplicate is worse
           than a clean failure.
        2. Every digest is computed from the loaded module's own `__file__`,
           so `source_digests` cannot name a file other than the one imported.

        `ciphers` has no `__init__.py` in the pinned checkout, so it is an
        implicit NAMESPACE package: every `ciphers/` directory on `sys.path`
        contributes a portion to the same package. Its `__path__` is therefore
        checked entry by entry, not just its `__file__` (a namespace package
        has none).

        CONSTRAINT, STATED RATHER THAN WORKED AROUND
        --------------------------------------------
        `main.py` imports `optimizer` and `train_nets` as TOP-LEVEL modules,
        so the checkout must be on `sys.path` for it to import at all, and the
        modules it binds stay in `sys.modules` afterwards under names it does
        not own. Consequently **two different AutoND checkouts cannot both be
        loaded in one process.** The second is refused by rule 1 above rather
        than silently served the first one's code. Audit a second checkout in
        a separate process.
        """
        root = self.autond_root
        if not (root / "main.py").is_file():
            raise FileNotFoundError(
                f"{root} has no main.py; point autond_root at the "
                f"pinned AutoND checkout.")
        cipher_path = (root / "ciphers" / f"{self.cipher_name}.py").resolve()
        if not cipher_path.is_file():
            raise FileNotFoundError(f"{cipher_path} does not exist")

        # (0) PIN ENFORCEMENT BEFORE IMPORT.
        #
        # Importing executes the checkout's code. Checking the pin only after
        # import means unapproved source has already run in this process by
        # the time the mismatch is reported, and `main.py` pulls in
        # `optimizer`, `train_nets`, TensorFlow and Keras on the way. So the
        # digests are compared against the files on disk FIRST, and the
        # post-import checks below are kept as defence in depth: they catch
        # the case where what actually got bound is not the file that was
        # hashed here.
        self.pin_verified_preimport = False
        if self.expected_upstream_commit:
            observed = _git_head(root)
            if observed is None:
                raise SourceBindingError(
                    f"REFUSED: a revision pin "
                    f"({self.expected_upstream_commit}) was supplied but "
                    f"{root} does not report a git revision, so the revision "
                    f"cannot be confirmed. Refusing rather than importing on "
                    f"the strength of the digests alone when a revision was "
                    f"explicitly required.")
            if observed != self.expected_upstream_commit:
                raise SourceBindingError(
                    f"REFUSED: {root} is at revision {observed} but the pin "
                    f"requires {self.expected_upstream_commit}. Nothing was "
                    f"imported.")
        if self.expected_source_digests:
            # Every module the import graph executes must be covered. A pin
            # that covers only the entry point leaves the rest to run
            # unchecked, which is the defect this closes.
            required = set(PINNED_IMPORT_GRAPH_FILES)
            required.add(f"ciphers/{self.cipher_name}.py")
            uncovered = sorted(required - set(self.expected_source_digests))
            if uncovered:
                raise SourceBindingError(
                    f"REFUSED: the pin does not cover {uncovered}, whose "
                    f"top-level code executes when the generator is imported. "
                    f"An uncovered module could differ from the approved "
                    f"revision and would run before anything noticed. Nothing "
                    f"was imported.")

            # Tracked files must match the approved revision. This is separate
            # from the digest check: it catches a modification to a file the
            # pin does not enumerate but which the graph may still reach.
            dirty = _git_tracked_modifications(root)
            if dirty:
                raise SourceBindingError(
                    f"REFUSED: the checkout {root} has {len(dirty)} modified "
                    f"tracked file(s): {dirty[:8]}. It does not match the "
                    f"approved revision. Nothing was imported.")

            # A clean tracked diff does not make the import path unambiguous:
            # an untracked file can still shadow a pinned module. In Python a
            # PACKAGE takes precedence over a module of the same name, so an
            # untracked `optimizer/` directory would be imported instead of
            # `optimizer.py`.
            ambiguous = _shadowing_candidates(root, self.cipher_name)
            if ambiguous:
                raise SourceBindingError(
                    f"REFUSED: the checkout contains path(s) that could "
                    f"shadow a pinned import: {ambiguous}. The module that "
                    f"would execute cannot be attributed to the approved "
                    f"revision. Nothing was imported.")

            on_disk: dict[str, str] = {}
            for rel in sorted(self.expected_source_digests):
                candidate = (root / rel).resolve()
                if not _is_within(candidate, root):
                    raise SourceBindingError(
                        f"REFUSED: pinned path {rel!r} resolves to "
                        f"{candidate}, outside the checkout {root}.")
                if not candidate.is_file():
                    raise SourceBindingError(
                        f"REFUSED: the pin requires {rel!r}, which does not "
                        f"exist under {root}. Nothing was imported.")
                on_disk[rel] = _sha256_file(candidate)
            bad = {rel: {"expected": want, "on_disk": on_disk.get(rel)}
                   for rel, want in self.expected_source_digests.items()
                   if on_disk.get(rel) != want}
            if bad:
                raise SourceBindingError(
                    f"REFUSED: source pin mismatch for {root}: {bad}. "
                    f"Checked BEFORE import, so no code from this checkout "
                    f"has been executed.")
            self.pin_verified_preimport = True

        cipher_module_name = f"ciphers.{self.cipher_name}"
        guarded = PINNED_TOP_LEVEL_MODULES + ("ciphers", cipher_module_name)

        # (1) Refuse a foreign module already occupying one of these names.
        conflicts = {}
        for name in guarded:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            locations = _module_locations(mod)
            if not locations:
                # A module with no discoverable location (built-in, extension,
                # or a stub) cannot be shown to come from this checkout.
                conflicts[name] = "<no file location>"
            elif not all(_is_within(loc, root) for loc in locations):
                conflicts[name] = ", ".join(str(p) for p in locations)
        if conflicts:
            raise SourceBindingError(
                f"REFUSED: module name(s) already bound outside the requested "
                f"checkout {root}: "
                + "; ".join(f"{k} -> {v}" for k, v in sorted(conflicts.items()))
                + ". Importing now would return the cached module while "
                  "provenance recorded this checkout's files, so the recorded "
                  "source would not be the executed source. These modules are "
                  "NOT evicted, because removing another component's module is "
                  "an unsafe side effect. Audit this checkout in a separate "
                  "process.")

        # (2) Import with the checkout on sys.path, then take it back off.
        inserted = False
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
            inserted = True
        try:
            main = importlib.import_module("main")
            cipher = importlib.import_module(cipher_module_name)
        finally:
            if inserted:
                try:
                    sys.path.remove(str(root))
                except ValueError:
                    pass

        # (3) Confirm what was actually bound resolves inside the checkout.
        bound: dict[str, Path] = {}
        for name, mod in (("main", main), (cipher_module_name, cipher)):
            locations = _module_locations(mod)
            if not locations:
                raise SourceBindingError(
                    f"REFUSED: module {name!r} imported with no resolvable "
                    f"file location, so its provenance cannot be established.")
            for loc in locations:
                if not _is_within(loc, root):
                    raise SourceBindingError(
                        f"REFUSED: module {name!r} resolved to {loc}, which is "
                        f"outside the requested checkout {root}. The executed "
                        f"source is not the source this adapter was asked for.")
            bound[name] = locations[0]

        # The namespace package itself: every portion must be inside the
        # checkout, or a foreign `ciphers/` directory is contributing modules.
        ns = sys.modules.get("ciphers")
        if ns is not None:
            portions = _module_locations(ns)
            outside = [p for p in portions if not _is_within(p, root)]
            if outside or not portions:
                raise SourceBindingError(
                    f"REFUSED: the `ciphers` namespace package draws from "
                    f"{[str(p) for p in (outside or portions)]}, outside the "
                    f"requested checkout {root}. A foreign `ciphers/` "
                    f"directory on sys.path can supply cipher modules, so the "
                    f"cipher that gets imported would not be the pinned one.")

        # The sibling modules main.py pulls in must come from here too.
        for name in PINNED_TOP_LEVEL_MODULES:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            for loc in _module_locations(mod):
                if not _is_within(loc, root):
                    raise SourceBindingError(
                        f"REFUSED: after importing, {name!r} resolves to "
                        f"{loc}, outside {root}.")

        for attr in ("plain_bits", "key_bits", "word_size", "encrypt"):
            if not hasattr(cipher, attr):
                raise AttributeError(
                    f"{cipher_path} has no {attr!r}; it does not match the "
                    f"upstream cipher module shape this adapter expects.")
        if not hasattr(main, "make_train_data"):
            raise AttributeError("the pinned main.py has no make_train_data")

        # (4) Digests of the files ACTUALLY LOADED, taken from each module's
        # own __file__ rather than from the requested path. Provenance
        # therefore cannot name a file that was not executed.
        # Digests of EVERY graph module that is now bound, taken from each
        # module's own __file__. Recording only the entry point would leave
        # the provenance silent about four modules whose code ran.
        self.source_digests = {
            "main.py": _sha256_file(bound["main"]),
            f"ciphers/{self.cipher_name}.py": _sha256_file(
                bound[cipher_module_name]),
        }
        self.source_paths = {
            "main.py": str(bound["main"]),
            f"ciphers/{self.cipher_name}.py": str(bound[cipher_module_name]),
        }
        for rel in PINNED_IMPORT_GRAPH_FILES:
            if rel == "main.py":
                continue
            mod_name = rel[:-3]
            mod = sys.modules.get(mod_name)
            if mod is None:
                # The graph did not load it in this configuration. Recorded as
                # not-loaded rather than silently omitted, so a reader can see
                # the difference between "clean" and "absent".
                self.source_paths[rel] = "NOT_LOADED"
                continue
            locations = _module_locations(mod)
            if not locations:
                raise SourceBindingError(
                    f"REFUSED: module {mod_name!r} is bound with no "
                    f"resolvable file location; its provenance cannot be "
                    f"established.")
            loc = locations[0]
            if not _is_within(loc, root):
                raise SourceBindingError(
                    f"REFUSED: module {mod_name!r} resolved to {loc}, outside "
                    f"the requested checkout {root}.")
            self.source_digests[rel] = _sha256_file(loc)
            self.source_paths[rel] = str(loc)

        # Post-import re-check, as defence in depth. Step (0) hashed the
        # files on disk; this compares the digests of the files that were
        # ACTUALLY BOUND (taken from each module's own __file__). A difference
        # between the two would mean the import resolved somewhere other than
        # the hashed path.
        if self.expected_source_digests:
            bad = {k: {"expected": v, "loaded": self.source_digests.get(k)}
                   for k, v in self.expected_source_digests.items()
                   if k in self.source_digests
                   and self.source_digests.get(k) != v}
            if bad:
                raise SourceBindingError(
                    f"REFUSED: post-import source pin mismatch for {root}: "
                    f"{bad}. The modules that were bound are not the files "
                    f"the pin approved.")
        return main, cipher

    # -- dataset surface ----------------------------------------------

    @property
    def feature_bits(self) -> int:
        """Columns in X.

        `make_train_data` returns ``np.hstack([C0, C1])`` and both ciphertexts
        are ``plain_bits`` wide, so the feature width is ``2 * plain_bits``.
        Verified as 64 for speck3264 rather than assumed.
        """
        return 2 * self.plain_bits

    @property
    def half_width(self) -> int:
        """Columns per ciphertext half: C0 is [0, half), C1 is [half, 2*half)."""
        return self.plain_bits

    def generate_partition(
        self,
        samples: int,
        *,
        num_rounds: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One partition from the pinned generator, unmodified.

        Mirrors `GohrAdapter.generate_partition`: a thin wrapper that calls the
        target's generator and nothing else. No reshaping, no dtype coercion,
        no reordering — whatever the audited code produces is what gets
        audited.
        """
        # See the constructor: bool passes an int test. `samples=True` would
        # otherwise generate a ONE-ROW partition that every downstream
        # statistic would then be computed over.
        if isinstance(samples, bool):
            raise TypeError(
                f"samples must be an integer, not a bool ({samples!r}); "
                f"True would silently generate a one-row partition.")
        if int(samples) < 1:
            raise ValueError("samples must be >= 1")
        if isinstance(num_rounds, bool):
            raise TypeError(
                f"num_rounds must be an integer, not a bool ({num_rounds!r}); "
                f"True would silently run the search at one round.")
        rounds = self.num_rounds if num_rounds is None else int(num_rounds)
        if rounds < 1:
            raise ValueError("num_rounds must be >= 1")

        X, Y = self._main.make_train_data(
            self._cipher.encrypt,
            self.plain_bits,
            self.key_bits,
            int(samples),
            rounds,
            self._delta_plain,
            self._delta_key,
        )
        return X, Y

    def dataset_provenance(
        self,
        *,
        train_samples: int,
        validation_samples: int,
        test_samples: int,
    ) -> dict[str, Any]:
        """Provenance in the shape the Gohr adapter established."""
        return {
            # Derived, not the class constant: see `dataset_id`.
            "dataset_id": self.dataset_id,
            "dataset_version": (
                f"pinned-main-make-train-data-{self.cipher_name}"
                f"-r{self.num_rounds}"),
            "verified_configuration": self.is_verified_configuration,
            "verified_configuration_note": (
                f"Representation verified against the pinned source for "
                f"{list(VERIFIED_CIPHERS)} only. A cipher outside that set "
                f"generates data this adapter has not checked the shape, "
                f"dtype or difference encoding of."),
            "generation_procedure": (
                "AutoNDDatasetAdapter.generate_partition() -> "
                "main.make_train_data(encrypt, plain_bits, key_bits, n, nr, "
                "delta_state, delta_key)"
            ),
            "generation_parameters": {
                "generator": self.GENERATOR_NAME,
                "cipher": self.cipher_name,
                "scenario": self.scenario,
                "num_rounds": self.num_rounds,
                "train_samples": int(train_samples),
                "validation_samples": int(validation_samples),
                "test_samples": int(test_samples),
                "input_difference": {
                    "delta_plain_hex": hex(self.delta_plain_int),
                    "delta_key_hex": hex(self.delta_key_int),
                    "delta_plain_popcount": int(self._delta_plain.sum()),
                },
                "representation": {
                    "feature_bits": self.feature_bits,
                    "layout": "np.hstack([C0, C1])",
                    "c0_columns": [0, self.half_width],
                    "c1_columns": [self.half_width, 2 * self.half_width],
                    "plain_bits": self.plain_bits,
                    "key_bits": self.key_bits,
                    "word_size": self.word_size,
                },
                "randomness_source": "os.urandom",
                "deterministic_seed": None,
            },
            # Which source bytes produced this instance. Recorded so a
            # manifest names the code, not just the configuration; a reader
            # can compare these against the declared pin themselves.
            "source_digests": dict(self.source_digests),
            "source_digest_note": (
                "SHA-256 of the imported pinned files. Recorded, not enforced: "
                "the adapter does not refuse a checkout whose digests differ "
                "from any declared pin, so a reader must compare them."
            ),
            # No seed exists on this path. Recorded as None rather than
            # fabricating a value that would imply replayability.
            "generation_random_seed": None,
            "exact_replay_available": False,
            "exact_replay_note": (
                "main.make_train_data draws keys, plaintexts, the label vector "
                "and the random-class substitution from os.urandom. There is no "
                "seed parameter on this path, so an integer seed cannot "
                "reproduce the instance; the persisted arrays are the only "
                "reproducible reference."
            ),
        }
