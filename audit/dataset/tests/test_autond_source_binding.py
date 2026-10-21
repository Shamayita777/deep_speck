"""
Source-binding tests for the AutoND dataset adapter.

Scientific status
-----------------
NON-EVIDENTIARY. These tests establish that the adapter either loads the
checkout it was asked for, or refuses — and that the digests it records name
the files actually executed. They say nothing about any dataset.

What this suite exists to prevent
---------------------------------
`importlib.import_module` consults `sys.modules` before `sys.path`. An earlier
version of the adapter inserted the checkout on `sys.path`, imported `main`,
and then recorded SHA-256 digests of files under that checkout. If anything in
the process had already imported a module named `main` — another AutoND
checkout, or a user script — the cached module was returned, the requested
checkout was never read, and the recorded provenance named code that had not
run. Provenance that can name unexecuted code is worse than no provenance,
because it is believed.

ISOLATION
---------
Every adversarial case runs in a FRESH SUBPROCESS. Poisoning `sys.modules`
with a decoy `main` or `ciphers` is irreversible within a process — the point
of the refusal is precisely that the adapter will not evict another
component's modules — so running these in-process would contaminate every
later test. Each subprocess prints one JSON line, which the parent asserts on.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

_CANDIDATE_ROOTS = (os.environ.get("AUTOND_ROOT"),
                    "~/Documents/AutoND-upstream")


def _pinned_root() -> Path:
    for cand in _CANDIDATE_ROOTS:
        if not cand:
            continue
        p = Path(cand).expanduser()
        if (p / "main.py").is_file():
            return p.resolve()
    pytest.skip("pinned AutoND checkout not found; set AUTOND_ROOT. "
                "SKIPPED IS NOT PASSED.")


@pytest.fixture(scope="module")
def root() -> Path:
    return _pinned_root()


def _run(body: str, tmp_path: Path) -> dict:
    """Run a scenario in a fresh interpreter and return its JSON verdict."""
    script = tmp_path / "scenario.py"
    script.write_text(textwrap.dedent(body))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, env=env,
                          cwd=str(tmp_path), timeout=300)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("@@")]
    if not lines:
        raise AssertionError(
            f"scenario produced no verdict.\nstdout:\n{proc.stdout[-3000:]}\n"
            f"stderr:\n{proc.stderr[-3000:]}")
    return json.loads(lines[-1][2:])


def _full_pin() -> dict:
    """The frozen pin, imported rather than copied.

    The adapter now requires a pin covering EVERY module the import graph
    executes, so tests that want to exercise one bad entry must start from the
    complete set rather than a one-file pin.
    """
    from audit.dataset.experiments.d0_autond_canonical import (
        AUTOND_SOURCE_PIN)
    return dict(AUTOND_SOURCE_PIN["source_digests"])


def _decoy_checkout(tmp_path: Path, name: str = "decoy") -> Path:
    """A directory that LOOKS importable under the pinned module names.

    Deliberately minimal: these modules only need to import successfully so
    they can occupy the names. The adapter must refuse rather than mistake
    them for AutoND, and it never gets far enough to call into them.
    """
    d = tmp_path / name
    (d / "ciphers").mkdir(parents=True)
    (d / "main.py").write_text("MARKER = 'decoy-main'\n")
    (d / "optimizer.py").write_text("MARKER = 'decoy-optimizer'\n")
    (d / "train_nets.py").write_text("MARKER = 'decoy-train-nets'\n")
    (d / "ciphers" / "speck3264.py").write_text(
        "MARKER = 'decoy-cipher'\nplain_bits = 1\nkey_bits = 1\n"
        "word_size = 1\n\ndef encrypt(*a, **k):\n    raise AssertionError('decoy')\n")
    return d


# ---------------------------------------------------------------------
# 1. The control: the requested checkout loads
# ---------------------------------------------------------------------

def test_requested_checkout_loads_in_a_clean_process(root, tmp_path):
    verdict = _run(f"""
        import json, sys
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
        print("@@" + json.dumps({{
            "ok": True,
            "main_file": a.source_paths["main.py"],
            "cipher_file": a.source_paths["ciphers/speck3264.py"],
            "digests": a.source_digests,
            "plain_bits": a.plain_bits,
        }}))
    """, tmp_path)
    assert verdict["ok"] is True
    assert verdict["plain_bits"] == 32
    assert Path(verdict["main_file"]).resolve() == (root / "main.py").resolve()
    assert Path(verdict["cipher_file"]).resolve() == (
        root / "ciphers" / "speck3264.py").resolve()


# ---------------------------------------------------------------------
# 2. A foreign module named `main` must not be mistaken for AutoND
# ---------------------------------------------------------------------

def test_preloaded_foreign_main_is_refused_not_used(root, tmp_path):
    decoy = _decoy_checkout(tmp_path)
    verdict = _run(f"""
        import json, sys
        sys.path.insert(0, {str(decoy)!r})
        import main                       # the decoy now owns the name
        assert getattr(main, "MARKER", None) == "decoy-main"
        sys.path.remove({str(decoy)!r})
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        try:
            AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
            print("@@" + json.dumps({{"refused": False}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{
                "refused": True, "message": str(exc),
                # The decoy must still be present: refusing must not evict
                # another component's module.
                "decoy_still_loaded": getattr(
                    sys.modules.get("main"), "MARKER", None) == "decoy-main",
            }}))
    """, tmp_path)
    assert verdict["refused"] is True, (
        "a decoy module named `main` was accepted; the adapter would have "
        "recorded the pinned checkout's digests for code it never executed.")
    assert "main" in verdict["message"]
    assert "REFUSED" in verdict["message"]
    assert verdict["decoy_still_loaded"] is True, (
        "the refusal evicted another component's module from sys.modules; "
        "that is the unsafe side effect this design avoids.")


def test_refusal_names_the_offending_path(root, tmp_path):
    decoy = _decoy_checkout(tmp_path)
    verdict = _run(f"""
        import json, sys
        sys.path.insert(0, {str(decoy)!r})
        import main
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        try:
            AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
            print("@@" + json.dumps({{"refused": False, "message": ""}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{"refused": True, "message": str(exc)}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert "decoy" in verdict["message"], (
        "the refusal should name where the conflicting module came from, so "
        "the operator can find it")


# ---------------------------------------------------------------------
# 3. A foreign `ciphers` namespace portion must be detected
# ---------------------------------------------------------------------

def test_preloaded_foreign_cipher_module_is_refused(root, tmp_path):
    decoy = _decoy_checkout(tmp_path)
    verdict = _run(f"""
        import json, sys
        sys.path.insert(0, {str(decoy)!r})
        import ciphers.speck3264 as c
        assert getattr(c, "MARKER", None) == "decoy-cipher"
        sys.path.remove({str(decoy)!r})
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        try:
            AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
            print("@@" + json.dumps({{"refused": False, "message": ""}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{"refused": True, "message": str(exc)}}))
    """, tmp_path)
    assert verdict["refused"] is True, (
        "a decoy ciphers.speck3264 was accepted; the audited cipher would not "
        "have been the pinned one.")
    assert "ciphers" in verdict["message"]


def test_foreign_ciphers_directory_on_syspath_is_detected(root, tmp_path):
    """The namespace-package hazard, with nothing pre-imported.

    `ciphers` has no `__init__.py` in the pinned checkout, so it is an
    implicit namespace package and EVERY `ciphers/` directory on `sys.path`
    contributes a portion to it. Checking only the imported cipher's
    `__file__` would miss a second portion that could supply other cipher
    modules.
    """
    decoy = _decoy_checkout(tmp_path)
    verdict = _run(f"""
        import json, sys
        # On sys.path but NOT imported: the namespace package will span both.
        sys.path.append({str(decoy)!r})
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        try:
            a = AutoNDDatasetAdapter(autond_root={str(root)!r},
                                     delta_plain_int=0x400000, num_rounds=8)
            paths = [str(p) for p in getattr(
                sys.modules.get("ciphers"), "__path__", [])]
            print("@@" + json.dumps({{
                "refused": False, "ciphers_path": paths,
                "cipher_file": a.source_paths["ciphers/speck3264.py"]}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{"refused": True, "message": str(exc)}}))
    """, tmp_path)
    if verdict["refused"]:
        assert "ciphers" in verdict["message"]
    else:
        # Acceptable only if the namespace package drew NOTHING from the decoy;
        # then no foreign code could have been supplied.
        assert all("decoy" not in p for p in verdict["ciphers_path"]), (
            f"the ciphers namespace package spans a foreign directory "
            f"{verdict['ciphers_path']} and the adapter did not refuse")
        assert "decoy" not in verdict["cipher_file"]


# ---------------------------------------------------------------------
# 4. A source-pin mismatch is rejected
# ---------------------------------------------------------------------

def test_source_pin_mismatch_is_refused(root, tmp_path):
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        try:
            AutoNDDatasetAdapter(
                autond_root={str(root)!r}, delta_plain_int=0x400000,
                num_rounds=8,
                expected_source_digests=dict(
                    _P.AUTOND_SOURCE_PIN["source_digests"],
                    **{{"main.py": "0"*64}}))
            print("@@" + json.dumps({{"refused": False, "message": ""}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{"refused": True, "message": str(exc)}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert "pin mismatch" in verdict["message"]


def test_correct_source_pin_is_accepted(root, tmp_path):
    """The pin must accept the real thing, or it is merely a blocker."""
    expected = hashlib.sha256((root / "main.py").read_bytes()).hexdigest()
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(
            autond_root={str(root)!r}, delta_plain_int=0x400000, num_rounds=8,
            expected_source_digests=_P.AUTOND_SOURCE_PIN["source_digests"])
        print("@@" + json.dumps({{"ok": True,
                                  "digest": a.source_digests["main.py"]}}))
    """, tmp_path)
    assert verdict["ok"] is True
    assert verdict["digest"] == expected


def test_pin_mismatch_is_detected_BEFORE_any_import(root, tmp_path):
    """The ordering property, not just the refusal.

    Importing executes the checkout's code, and `main.py` drags in
    `optimizer`, `train_nets`, TensorFlow and Keras on the way. Checking the
    pin only after import means unapproved source has already run. This
    asserts that after a pin mismatch NONE of the checkout's modules are in
    `sys.modules` — i.e. nothing was executed.
    """
    verdict = _run(f"""
        import json, sys
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        watched = ("main", "optimizer", "train_nets", "ciphers",
                   "ciphers.speck3264", "tensorflow", "keras")
        before = [m for m in watched if m in sys.modules]
        try:
            AutoNDDatasetAdapter(
                autond_root={str(root)!r}, delta_plain_int=0x400000,
                num_rounds=8,
                expected_source_digests=dict(
                    _P.AUTOND_SOURCE_PIN["source_digests"],
                    **{{"main.py": "0"*64}}))
            print("@@" + json.dumps({{"refused": False}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{
                "refused": True, "message": str(exc),
                "before": before,
                "after": [m for m in watched if m in sys.modules]}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert verdict["before"] == []
    assert verdict["after"] == [], (
        f"the checkout's modules {verdict['after']} were imported before the "
        f"pin was checked; unapproved source executed in this process")
    assert "BEFORE import" in verdict["message"]


def test_revision_mismatch_is_detected_before_any_import(root, tmp_path):
    verdict = _run(f"""
        import json, sys
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        watched = ("main", "optimizer", "train_nets", "tensorflow")
        try:
            AutoNDDatasetAdapter(
                autond_root={str(root)!r}, delta_plain_int=0x400000,
                num_rounds=8, expected_upstream_commit="d"*40)
            print("@@" + json.dumps({{"refused": False}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{
                "refused": True, "message": str(exc),
                "after": [m for m in watched if m in sys.modules]}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert verdict["after"] == []
    assert "Nothing was imported" in verdict["message"]


def test_correct_revision_pin_is_accepted(root, tmp_path):
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(
            autond_root={str(root)!r}, delta_plain_int=0x400000, num_rounds=8,
            expected_upstream_commit=
                "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779")
        print("@@" + json.dumps({{"ok": True,
                                  "preimport": a.pin_verified_preimport}}))
    """, tmp_path)
    assert verdict["ok"] is True


def test_a_pinned_file_that_does_not_exist_is_refused_before_import(root,
                                                                    tmp_path):
    verdict = _run(f"""
        import json, sys
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        try:
            AutoNDDatasetAdapter(
                autond_root={str(root)!r}, delta_plain_int=0x400000,
                num_rounds=8,
                expected_source_digests=dict(
                    _P.AUTOND_SOURCE_PIN["source_digests"],
                    **{{"no_such_file.py": "0"*64}}))
            print("@@" + json.dumps({{"refused": False}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{
                "refused": True, "message": str(exc),
                "after": [m for m in ("main", "tensorflow")
                          if m in sys.modules]}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert verdict["after"] == []
    assert "does not exist" in verdict["message"]


def test_a_pinned_path_escaping_the_checkout_is_refused(root, tmp_path):
    """A pin entry must not be able to name a file outside the checkout."""
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        try:
            AutoNDDatasetAdapter(
                autond_root={str(root)!r}, delta_plain_int=0x400000,
                num_rounds=8,
                expected_source_digests=dict(
                    _P.AUTOND_SOURCE_PIN["source_digests"],
                    **{{"../../etc/hostname": "0"*64}}))
            print("@@" + json.dumps({{"refused": False}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{"refused": True,
                                      "message": str(exc)}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert "outside the checkout" in verdict["message"]


def test_preimport_flag_records_that_the_pin_was_checked(root, tmp_path):
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        import hashlib
        from pathlib import Path
        a = AutoNDDatasetAdapter(
            autond_root={str(root)!r}, delta_plain_int=0x400000, num_rounds=8,
            expected_source_digests=_P.AUTOND_SOURCE_PIN["source_digests"])
        b = AutoNDDatasetAdapter(
            autond_root={str(root)!r}, delta_plain_int=0x400000, num_rounds=8)
        print("@@" + json.dumps({{"pinned": a.pin_verified_preimport,
                                  "unpinned": b.pin_verified_preimport}}))
    """, tmp_path)
    assert verdict["pinned"] is True
    assert verdict["unpinned"] is False, (
        "an adapter with no pin must not claim the pin was verified")


# ---------------------------------------------------------------------
# 4b. The WHOLE import graph is verified before any of it executes
# ---------------------------------------------------------------------
# `main.py` imports `optimizer` and `train_nets`; `train_nets` imports
# `dbitnet` and `gohrnet`. Earlier revisions pinned only `main.py` and the
# cipher, so four modules' top-level code ran without ever being compared
# with the pin. A post-import refusal is no use here: by then the unapproved
# code has executed.
#
# Each test below appends a SENTINEL to the module it tampers with — a line
# that writes a marker file when the module's top level runs. The assertion is
# that the marker does NOT exist. Asserting only that an exception was raised
# would pass even if the code had run and something failed afterwards.

def _tampered_checkout(root: Path, dest: Path, target_rel: str,
                       sentinel: Path) -> Path:
    """A copy of the checkout with one module carrying an import sentinel.

    The REAL checkout is never touched: everything happens in `dest`.
    """
    import shutil
    shutil.copytree(root, dest,
                    ignore=shutil.ignore_patterns(".git", "__pycache__",
                                                  "model_weights"))
    target = dest / target_rel
    target.write_text(
        target.read_text()
        + "\n# appended by a test: writes a marker iff this module executes\n"
        + "import pathlib as _pl\n"
        + f"_pl.Path({str(sentinel)!r}).write_text('executed')\n")
    return dest


def _pin_without_commit() -> str:
    """The frozen digests, with the revision check disabled.

    A copied checkout has no .git, so a revision pin would refuse first and
    mask the digest check these tests are about.
    """
    return ('{"upstream_commit": None, "source_digests": '
            'dict(_P.AUTOND_SOURCE_PIN["source_digests"])}')


@pytest.mark.parametrize("module_rel", [
    "optimizer.py", "train_nets.py", "dbitnet.py", "gohrnet.py", "main.py",
])
def test_a_modified_graph_module_is_refused_before_it_executes(
        root, tmp_path, module_rel):
    sentinel = tmp_path / f"sentinel_{module_rel}.marker"
    copy = _tampered_checkout(root, tmp_path / f"c_{module_rel}", module_rel,
                              sentinel)
    verdict = _run(f"""
        import json
        from pathlib import Path
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        pin = {_pin_without_commit()}
        try:
            AutoNDDatasetAdapter(
                autond_root={str(copy)!r}, delta_plain_int=0x400000,
                num_rounds=8,
                expected_source_digests=pin["source_digests"])
            refused, msg = False, ""
        except SourceBindingError as exc:
            refused, msg = True, str(exc)
        print("@@" + json.dumps({{
            "refused": refused, "message": msg,
            "sentinel_exists": Path({str(sentinel)!r}).exists()}}))
    """, tmp_path)
    assert verdict["refused"] is True, (
        f"a modified {module_rel} was accepted")
    assert verdict["sentinel_exists"] is False, (
        f"{module_rel}'s top-level code EXECUTED before the pin was checked; "
        f"the sentinel file was written")
    assert not sentinel.exists()
    assert module_rel in verdict["message"] or "mismatch" in verdict[
        "message"].lower()


def test_a_shadowing_package_is_refused_before_import(root, tmp_path):
    """A package shadows a module of the same name.

    An untracked `optimizer/` directory with `__init__.py` is imported in
    preference to `optimizer.py`, so a clean tracked diff does not make the
    import path unambiguous.
    """
    import shutil
    copy = tmp_path / "shadow_pkg"
    shutil.copytree(root, copy,
                    ignore=shutil.ignore_patterns(".git", "__pycache__",
                                                  "model_weights"))
    sentinel = tmp_path / "shadow.marker"
    pkg = copy / "optimizer"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "import pathlib as _pl\n"
        f"_pl.Path({str(sentinel)!r}).write_text('executed')\n")
    verdict = _run(f"""
        import json
        from pathlib import Path
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        pin = {_pin_without_commit()}
        try:
            AutoNDDatasetAdapter(
                autond_root={str(copy)!r}, delta_plain_int=0x400000,
                num_rounds=8,
                expected_source_digests=pin["source_digests"])
            refused, msg = False, ""
        except SourceBindingError as exc:
            refused, msg = True, str(exc)
        print("@@" + json.dumps({{
            "refused": refused, "message": msg,
            "sentinel_exists": Path({str(sentinel)!r}).exists()}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert verdict["sentinel_exists"] is False, (
        "the shadowing package executed before the check")


def test_a_ciphers_init_is_refused(root, tmp_path):
    """`ciphers` is an implicit namespace package in the pinned state."""
    import shutil
    copy = tmp_path / "ns_pkg"
    shutil.copytree(root, copy,
                    ignore=shutil.ignore_patterns(".git", "__pycache__",
                                                  "model_weights"))
    sentinel = tmp_path / "ns.marker"
    (copy / "ciphers" / "__init__.py").write_text(
        "import pathlib as _pl\n"
        f"_pl.Path({str(sentinel)!r}).write_text('executed')\n")
    verdict = _run(f"""
        import json
        from pathlib import Path
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        pin = {_pin_without_commit()}
        try:
            AutoNDDatasetAdapter(
                autond_root={str(copy)!r}, delta_plain_int=0x400000,
                num_rounds=8,
                expected_source_digests=pin["source_digests"])
            refused, msg = False, ""
        except SourceBindingError as exc:
            refused, msg = True, str(exc)
        print("@@" + json.dumps({{
            "refused": refused, "message": msg,
            "sentinel_exists": Path({str(sentinel)!r}).exists()}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert verdict["sentinel_exists"] is False


def test_an_incomplete_pin_is_refused_before_import(root, tmp_path):
    """A pin that omits a graph module is refused, not partially honoured."""
    verdict = _run(f"""
        import json, sys
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        try:
            AutoNDDatasetAdapter(
                autond_root={str(root)!r}, delta_plain_int=0x400000,
                num_rounds=8,
                expected_source_digests={{
                    "main.py": "0"*64,
                    "ciphers/speck3264.py": "0"*64}})
            refused, msg = False, ""
        except SourceBindingError as exc:
            refused, msg = True, str(exc)
        print("@@" + json.dumps({{
            "refused": refused, "message": msg,
            "after": [m for m in ("main", "optimizer", "train_nets",
                                  "dbitnet", "gohrnet", "tensorflow")
                      if m in sys.modules]}}))
    """, tmp_path)
    assert verdict["refused"] is True
    assert verdict["after"] == []
    assert "does not cover" in verdict["message"]
    for m in ("optimizer.py", "train_nets.py", "dbitnet.py", "gohrnet.py"):
        assert m in verdict["message"]


def test_the_clean_approved_checkout_loads_with_the_full_pin(root, tmp_path):
    """The control. Without this the refusals above prove nothing useful."""
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(
            autond_root={str(root)!r}, delta_plain_int=0x400000, num_rounds=8,
            expected_source_digests=_P.AUTOND_SOURCE_PIN["source_digests"],
            expected_upstream_commit=_P.AUTOND_SOURCE_PIN["upstream_commit"])
        print("@@" + json.dumps({{
            "ok": True,
            "preimport": a.pin_verified_preimport,
            "digests": a.source_digests,
            "paths": a.source_paths}}))
    """, tmp_path)
    assert verdict["ok"] is True
    assert verdict["preimport"] is True
    # Every graph module's digest is recorded, from the file actually bound.
    assert set(verdict["digests"]) >= {
        "main.py", "optimizer.py", "train_nets.py", "dbitnet.py",
        "gohrnet.py", "ciphers/speck3264.py"}
    for rel, path in verdict["paths"].items():
        if path != "NOT_LOADED":
            assert Path(path).is_relative_to(root), f"{rel} -> {path}"


def test_post_import_checks_still_run(root, tmp_path):
    """Defence in depth must not have been dropped.

    The post-import comparison uses digests taken from each loaded module's
    own `__file__`; this asserts those were computed and agree with the pin,
    which is only possible if the post-import pass executed.
    """
    verdict = _run(f"""
        import hashlib, json, sys
        from pathlib import Path
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(
            autond_root={str(root)!r}, delta_plain_int=0x400000, num_rounds=8,
            expected_source_digests=_P.AUTOND_SOURCE_PIN["source_digests"])
        out = {{}}
        for name in ("main", "optimizer", "train_nets", "dbitnet", "gohrnet"):
            m = sys.modules.get(name)
            if m is None or not getattr(m, "__file__", None):
                continue
            f = Path(m.__file__)
            out[name + ".py"] = {{
                "module_file": str(f.resolve()),
                "recorded_path": a.source_paths.get(name + ".py"),
                "recorded_digest": a.source_digests.get(name + ".py"),
                "digest_of_module_file":
                    hashlib.sha256(f.read_bytes()).hexdigest()}}
        print("@@" + json.dumps(out))
    """, tmp_path)
    assert set(verdict) >= {"main.py", "optimizer.py", "train_nets.py",
                            "dbitnet.py", "gohrnet.py"}
    for rel, rec in verdict.items():
        assert rec["recorded_path"] == rec["module_file"], rel
        assert rec["recorded_digest"] == rec["digest_of_module_file"], rel


def test_the_pinned_checkout_is_unmodified_after_these_tests(root):
    """No test above may leave a mark on the audited subject."""
    import hashlib
    from audit.dataset.experiments.d0_autond_canonical import (
        AUTOND_SOURCE_PIN)
    for rel, expected in AUTOND_SOURCE_PIN["source_digests"].items():
        actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        assert actual == expected, (
            f"{rel} in the pinned checkout differs from the frozen pin; a "
            f"test modified the audited source")


def test_pin_is_off_by_default(root, tmp_path):
    """No expected digests supplied ⇒ an ordinary checkout is not blocked."""
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
        print("@@" + json.dumps({{"ok": True,
                                  "pin": a.expected_source_digests}}))
    """, tmp_path)
    assert verdict["ok"] is True
    assert verdict["pin"] == {}


# ---------------------------------------------------------------------
# 5. Provenance names the files actually imported
# ---------------------------------------------------------------------

def test_recorded_digests_are_of_the_files_actually_imported(root, tmp_path):
    """Digests come from each module's own `__file__`, not the requested path.

    This is the property that makes the provenance trustworthy: the digest and
    the executed code cannot refer to different files, because the digest is
    computed from the loaded module's location.
    """
    verdict = _run(f"""
        import hashlib, json
        from pathlib import Path
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
        import sys
        live = {{
            "main.py": sys.modules["main"].__file__,
            "ciphers/speck3264.py": sys.modules["ciphers.speck3264"].__file__,
        }}
        out = {{}}
        for rel, path in live.items():
            out[rel] = {{
                "module_file": str(Path(path).resolve()),
                "recorded_path": a.source_paths[rel],
                "recorded_digest": a.source_digests[rel],
                "digest_of_module_file": hashlib.sha256(
                    Path(path).read_bytes()).hexdigest(),
            }}
        print("@@" + json.dumps(out))
    """, tmp_path)
    for rel, rec in verdict.items():
        assert rec["recorded_path"] == rec["module_file"], (
            f"{rel}: provenance records {rec['recorded_path']} but the loaded "
            f"module is {rec['module_file']}")
        assert rec["recorded_digest"] == rec["digest_of_module_file"], (
            f"{rel}: the recorded digest is not the digest of the file that "
            f"was imported")
        assert Path(rec["module_file"]).is_relative_to(root)


def test_provenance_block_carries_the_loaded_source(root, tmp_path):
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
        p = a.dataset_provenance(train_samples=1, validation_samples=1,
                                 test_samples=1)
        print("@@" + json.dumps({{
            "digests": p["source_digests"],
            "note": p["source_digest_note"],
            "adapter_digests": a.source_digests}}))
    """, tmp_path)
    assert verdict["digests"] == verdict["adapter_digests"]
    assert set(verdict["digests"]) >= {"main.py", "ciphers/speck3264.py"}
    # The whole graph is now recorded, not just the entry point and cipher.
    assert set(verdict["digests"]) >= {
        "optimizer.py", "train_nets.py", "dbitnet.py", "gohrnet.py"}


# ---------------------------------------------------------------------
# 6. Two checkouts cannot share one process — stated, and enforced
# ---------------------------------------------------------------------

def test_a_second_checkout_in_one_process_is_refused(root, tmp_path):
    """The documented constraint, enforced rather than only written down.

    `main.py` imports `optimizer` and `train_nets` as top-level modules, so a
    checkout must be on `sys.path` to import at all and leaves those names
    bound afterwards. A second, different checkout therefore cannot be loaded
    in the same process. It is refused, not served the first one's code.
    """
    second = _decoy_checkout(tmp_path, name="second_checkout")
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        a = AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
        first = a.source_paths["main.py"]
        try:
            AutoNDDatasetAdapter(autond_root={str(second)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
            print("@@" + json.dumps({{"refused": False, "first": first,
                                      "message": ""}}))
        except SourceBindingError as exc:
            print("@@" + json.dumps({{"refused": True, "first": first,
                                      "message": str(exc)}}))
    """, tmp_path)
    assert verdict["refused"] is True, (
        "a second checkout was loaded in the same process; it would have "
        "received the first checkout's `main` while recording its own digests")
    assert "separate process" in verdict["message"]


def test_same_checkout_twice_in_one_process_is_fine(root, tmp_path):
    """The refusal must not fire for legitimate reuse of one checkout."""
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
        b = AutoNDDatasetAdapter(autond_root={str(root)!r},
                                 delta_plain_int=0x400000, num_rounds=6)
        print("@@" + json.dumps({{
            "ok": True,
            "same_digests": a.source_digests == b.source_digests,
            "rounds": [a.num_rounds, b.num_rounds]}}))
    """, tmp_path)
    assert verdict["ok"] is True
    assert verdict["same_digests"] is True
    assert verdict["rounds"] == [8, 6]


def test_checkout_path_is_resolved_absolutely(root, tmp_path):
    """A relative or symlinked root must not defeat the containment tests."""
    link = tmp_path / "link_to_checkout"
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable in this environment")
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import AutoNDDatasetAdapter
        a = AutoNDDatasetAdapter(autond_root={str(link)!r},
                                 delta_plain_int=0x400000, num_rounds=8)
        print("@@" + json.dumps({{"ok": True, "root": str(a.autond_root),
                                  "main": a.source_paths["main.py"]}}))
    """, tmp_path)
    assert verdict["ok"] is True
    assert Path(verdict["root"]) == root, (
        "the symlinked root was not resolved, so containment checks would "
        "compare incomparable paths")


def test_missing_checkout_is_still_a_file_error_not_a_binding_error(tmp_path):
    """Distinct failure modes stay distinct."""
    verdict = _run(f"""
        import json
        from audit.dataset.experiments import d0_autond_canonical as _P
        from audit.dataset.adapters.autond import (
            AutoNDDatasetAdapter, SourceBindingError)
        try:
            AutoNDDatasetAdapter(autond_root={str(tmp_path / 'nope')!r},
                                 delta_plain_int=0x400000, num_rounds=8)
            print("@@" + json.dumps({{"kind": "none"}}))
        except SourceBindingError:
            print("@@" + json.dumps({{"kind": "SourceBindingError"}}))
        except FileNotFoundError:
            print("@@" + json.dumps({{"kind": "FileNotFoundError"}}))
    """, tmp_path)
    assert verdict["kind"] == "FileNotFoundError"
