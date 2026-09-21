"""Make the shared generic framework (experimental_validity/framework) importable."""
import sys
from pathlib import Path

_EV = Path(__file__).resolve().parents[2] / "experimental_validity"
if _EV.is_dir() and str(_EV) not in sys.path:
    sys.path.insert(0, str(_EV))
