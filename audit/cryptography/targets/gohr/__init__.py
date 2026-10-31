"""
Gohr target reference (registry only; no code).

The Gohr implementation stays where it is. Its files are bound by ``PACKAGE_MANIFEST.sha256``,
by the Gohr CE1-CE4 certificates and, at run time, by the A3 adversarial freeze, so they are
neither moved nor copied:

    cipher / data / model / trainer / evaluator   audit/cryptography/gohr/
    CE1 reference checkpoint (historical)         audit/cryptography/evidence/ce1/
    CE2-CE4 reference checkpoint                  audit/cryptography/Archive/best5depth10.h5
    host repository (Gohr's original code)        repository root (speck.py, train_nets.py, ...)
"""

from pathlib import Path

CRYPTO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = CRYPTO_ROOT / "gohr"
CE234_REFERENCE_CHECKPOINT = CRYPTO_ROOT / "Archive" / "best5depth10.h5"
CE234_REFERENCE_CHECKPOINT_SHA256 = "256eb4a5ba93414f0a46ffd498d6121a03705698ac2a81764c2375207c5d6737"
SOURCE_FILES = ("speck.py", "dataset.py", "model.py", "trainer.py", "evaluate.py")
