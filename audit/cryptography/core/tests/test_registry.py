"""The registry agrees with the source tree, the manifests and the A3 freeze."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from audit.cryptography.core import registry as REG

CRYPTO = Path(__file__).resolve().parents[2]
AUDIT = CRYPTO.parent
REPO = AUDIT.parent


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_route_paths_exist_and_one_authoritative_route_per_experiment():
    for rid, r in REG.ROUTES.items():
        assert (CRYPTO / r["entry"]).is_file(), rid
        for k in ("new_production", "implementation", "verifier"):
            if k in r:
                assert (CRYPTO / r[k]).is_file(), (rid, k)
    for exp in ("GOHR-CE1", "GOHR-CE2", "GOHR-CE3", "GOHR-CE4", "AUTOND-CE1", "AUTOND-CE2"):
        live = [k for k, r in REG.ROUTES.items() if k.startswith(exp) and r["status"] in ("AUTHORITATIVE", "NOT_RUN")]
        live += [k for k, r in REG.ROUTES.items() if k == exp and "new_production" in r]
        assert len(set(live)) <= 1, exp


def test_frozen_files_still_match_every_manifest_that_lists_them():
    v1 = {l[66:]: l[:64] for l in (CRYPTO / "PACKAGE_MANIFEST.sha256").read_text().splitlines() if l.strip()}
    c1 = {l[66:]: l[:64] for l in (CRYPTO / "ce234" / "CE234_MANIFEST.sha256").read_text().splitlines() if l.strip()}
    a1 = {l[66:]: l[:64] for l in (CRYPTO / "autond_ce1" / "package_r3.1" / "SHA256SUMS").read_text().splitlines()
          if l.strip()}
    for rel, bindings in REG.FROZEN.items():
        if "AutoND CE1 package SHA256SUMS" in bindings:
            assert a1.get(f"audit/cryptography/{rel}") == _sha(CRYPTO / rel), f"{rel} differs from AutoND CE1 SHA256SUMS"
        p = CRYPTO / rel
        assert p.is_file(), rel
        key = f"audit/cryptography/{rel}"
        if "PACKAGE_MANIFEST" in bindings:
            assert v1.get(key) == _sha(p), f"{rel} differs from PACKAGE_MANIFEST"
        if "CE234_MANIFEST v1" in bindings:
            assert c1.get(rel) == _sha(p), f"{rel} differs from CE234_MANIFEST v1"


def test_a3_binding_list_matches_the_adversarial_battery():
    battery = AUDIT / "adversarial" / "battery.py"
    if not battery.exists():
        pytest.skip("adversarial package not present")
    text = battery.read_text()
    found = set(re.findall(r'"(audit\.cryptography\.[a-z0-9_.]+)"', text[text.index("def source_hashes"):]))
    assert found == set(REG.A3_BOUND_MODULES)
    for mod in REG.A3_BOUND_MODULES:
        rel = mod.replace("audit.cryptography.", "").replace(".", "/") + ".py"
        assert rel in REG.FROZEN, rel


def _importers(module: str) -> list[str]:
    stem = module[:-3].replace("/", ".")
    pat = re.compile(rf"(audit\.cryptography\.{re.escape(stem)}\b|from \.+{re.escape(stem.split('.')[-1])} import|"
                     rf"from audit\.cryptography(\.[a-z_]+)* import [^\n#]*\b{re.escape(stem.split('.')[-1])}\b)")
    out = []
    for p in CRYPTO.rglob("*.py"):
        rel = p.relative_to(CRYPTO).as_posix()
        if rel == module or "/tests/" in f"/{rel}" or rel.startswith(("history", "autond_ce1/", "core/registry")) \
                or "/history/" in rel or "tests_authorised" in rel:
            continue
        if pat.search(p.read_text(errors="ignore")):
            out.append(rel)
    return out


@pytest.mark.parametrize("module", [m for m, c in REG.COMPONENTS.items() if not c["used_by_production"]])
def test_components_marked_unused_have_no_production_importer(module):
    importers = [i for i in _importers(module) if not i.startswith(("engine.py", "reporting.py", "evaluation.py",
                                                                      "test/", "adapters/gohr.py"))]
    assert importers == [], (module, importers)
