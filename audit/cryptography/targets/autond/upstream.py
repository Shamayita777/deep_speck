"""
Pinned AutoND upstream: hash verification and import isolation.

Location. The single pinned checkout (commit 8d66e0a43a9e6fbbb3624d68296c1b77aa2da779) stays at
``autond_ce1/AutoND-upstream/``, where AutoND CE1 evidence and manifests reference it. It is the
target-level upstream for every AutoND experiment; there is no second copy.

Isolation. The upstream uses generic top-level module names (``main``, ``optimizer``,
``train_nets``, ``ciphers`` ...). Importing it by putting its root on ``sys.path`` and leaving it
there would (a) let any later ``import main`` anywhere in the process resolve to AutoND, and
(b) let an unrelated module already registered under one of those names shadow the upstream.
``load`` therefore, for the duration of the import only:

1. verifies every file listed in ``upstream_sha256.json``;
2. sets aside any ``sys.modules`` entry that collides with an upstream top-level name;
3. prepends the root to ``sys.path`` and disables bytecode writing;
4. imports, and checks that every module resolved inside the pinned root;
5. moves every module that came from the root out of ``sys.modules`` into a private table,
   restores the set-aside entries, ``sys.path`` and the bytecode flag.

Module identity is stable: ``load`` caches one ``Upstream`` per resolved root, so repeated loads
return the same module objects. Upstream code that runs later only imports TensorFlow lazily
(checked by test), which is unaffected by the removal.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import threading
from pathlib import Path

CRYPTO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = CRYPTO_ROOT / "autond_ce1" / "AutoND-upstream"
HASH_FILE = CRYPTO_ROOT / "upstream_sha256.json"
UPSTREAM_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"
CHECKPOINT_RELPATH = "model_weights/speck3264_single-key_0x400000_round8_polished.h5"
CHECKPOINT_SHA256 = "a7be788dbbb760a7be9053748e6e0f37dedc5768db3a74a5c3ebd294dca4cc26"
TOP_LEVEL = ("main", "train_nets", "dbitnet", "gohrnet", "optimizer", "eval_nets", "polish_nets", "ciphers")
IMPORTS = ("ciphers.speck3264", "dbitnet", "main")

_LOCK = threading.Lock()
_CACHE: dict = {}


class UpstreamError(RuntimeError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_root(root=None) -> Path:
    return Path(root or os.environ.get("AUTOND_UPSTREAM") or DEFAULT_ROOT).resolve()


def verify(root) -> dict:
    root = Path(root)
    expected = json.loads(HASH_FILE.read_text())
    files = {}
    for rel, exp in sorted(expected.items()):
        p = root / rel
        obs = sha256_file(p) if p.is_file() else None
        files[rel] = {"expected": exp, "observed": obs, "ok": obs == exp}
    return {"root": str(root), "commit": UPSTREAM_COMMIT, "hash_file": str(HASH_FILE),
            "hash_file_sha256": sha256_file(HASH_FILE), "files": files,
            "all_verified": all(f["ok"] for f in files.values())
            and files.get(CHECKPOINT_RELPATH, {}).get("expected") == CHECKPOINT_SHA256}


def _inside(mod, root: Path) -> bool:
    f = getattr(mod, "__file__", None)
    if f:
        return root in Path(f).resolve().parents
    return any(root in Path(p).resolve().parents or Path(p).resolve() == root
               for p in getattr(mod, "__path__", []) or [])


def _is_upstream_name(name: str) -> bool:
    return name.split(".", 1)[0] in TOP_LEVEL


class Upstream:
    """Verified, isolated upstream modules. Use ``load()``; do not construct directly."""

    def __init__(self, root: Path, verification: dict, modules: dict):
        self.root = root
        self.verification = verification
        self.modules = modules
        self.cipher = modules["ciphers.speck3264"]
        self.dbitnet = modules["dbitnet"]
        self.main = modules["main"]

    @property
    def checkpoint(self) -> Path:
        return self.root / CHECKPOINT_RELPATH

    def delta_bits(self, input_difference: int = 0x400000):
        return self.main.integer_to_binary_array(input_difference, self.cipher.plain_bits)

    def delta_words(self, input_difference: int = 0x400000) -> tuple[int, int]:
        w = self.cipher.convert_from_binary(self.delta_bits(input_difference))
        return int(w[0, 0]), int(w[0, 1])


def load(root=None) -> Upstream:
    root = resolve_root(root)
    with _LOCK:
        if root in _CACHE:
            return _CACHE[root]
        ver = verify(root)
        if not ver["all_verified"]:
            bad = [k for k, v in ver["files"].items() if not v["ok"]]
            raise UpstreamError(f"pinned upstream verification failed for {bad} under {root}")
        set_aside = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_upstream_name(k)}
        saved_path, flag = list(sys.path), sys.dont_write_bytecode
        sys.path.insert(0, str(root))
        sys.dont_write_bytecode = True
        captured = {}
        try:
            for name in IMPORTS:
                importlib.import_module(name)
            for k in list(sys.modules):
                m = sys.modules[k]
                if _is_upstream_name(k) and m is not None:
                    if not _inside(m, root):
                        raise UpstreamError(f"module {k!r} resolved outside the pinned upstream {root}")
                    captured[k] = m
        finally:
            for k in [k for k in list(sys.modules) if _is_upstream_name(k)]:
                captured.setdefault(k, sys.modules[k])
                del sys.modules[k]
            sys.modules.update(set_aside)
            sys.path[:] = saved_path
            sys.dont_write_bytecode = flag
        missing = [n for n in IMPORTS if n not in captured]
        if missing:
            raise UpstreamError(f"upstream modules not captured: {missing}")
        up = Upstream(root, ver, captured)
        _CACHE[root] = up
        return up
