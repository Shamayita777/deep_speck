#!/usr/bin/env python3
"""
Apply Holm correction across the frozen primary hypothesis family
{H-EV-SHUFFLE, H-EV-REPRESENTATION} once BOTH production certificates
exist, and write the corrected certificates + a combined family
certificate. Must be run after both scripts/run_ev.py invocations for
the two paired experiments have produced their per-experiment
certificates - never before, and family membership is fixed (exactly
these two hypotheses) regardless of what either individual certificate
says.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gohr.experiments import apply_primary_family_correction


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: apply_family_correction.py <shuffle_certificate.json> <representation_certificate.json>")
        return 1

    shuffle_path, representation_path = Path(sys.argv[1]), Path(sys.argv[2])
    shuffle_cert = json.loads(shuffle_path.read_text())
    representation_cert = json.loads(representation_path.read_text())

    combined = apply_primary_family_correction(shuffle_cert, representation_cert)

    shuffle_path.write_text(json.dumps(combined["shuffle_certificate"], indent=2, sort_keys=True, default=str))
    representation_path.write_text(
        json.dumps(combined["representation_certificate"], indent=2, sort_keys=True, default=str)
    )
    combined_path = shuffle_path.parent / "primary_family_certificate.json"
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True, default=str))

    print(f"H-EV-SHUFFLE: adjusted_p={combined['shuffle_certificate']['statistics']['adjusted_p_value']} "
          f"decision={combined['shuffle_certificate']['decision']}")
    print(f"H-EV-REPRESENTATION: adjusted_p={combined['representation_certificate']['statistics']['adjusted_p_value']} "
          f"decision={combined['representation_certificate']['decision']}")
    print(f"Combined family certificate written to {combined_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
