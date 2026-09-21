#!/usr/bin/env python3
"""
Run all four EV experiments in SMOKE mode.

Every result produced by this script is NON_EVIDENTIARY: it verifies
that the pipeline works end-to-end (dataset generation, model
construction, training, evaluation, statistics, multiplicity
correction, manifest/certificate generation) on a drastically reduced
problem. It must never be used to populate the scientific evidence
tables, and every certificate it writes carries "non_evidentiary": true
and is written under results/smoke/, never results/production/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gohr.experiments import (
    apply_primary_family_correction,
    run_ev_baseline,
    run_ev_noise,
    run_h_ev_representation,
    run_h_ev_shuffle,
)
from gohr.representation import generate_candidate1_permutation, run_full_validation
from gohr.baseline import BASELINE
from gohr import speck
from framework.firewall import FreezeRecord, TestSetFirewall
from framework.provenance import config_hash, utc_timestamp


def main() -> int:
    output_root = Path(__file__).resolve().parent.parent / "results" / "smoke"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EV SMOKE RUN - ALL RESULTS ARE NON_EVIDENTIARY")
    print("=" * 70)

    print("\n[1/4] EV-BASELINE (smoke)...")
    baseline_cert = run_ev_baseline(
        run_mode="smoke", requested_replicates=3, minimum_valid_replicates=2,
        output_dir=output_root / "ev_baseline",
    )
    _write(output_root / "ev_baseline_certificate.json", baseline_cert)
    print(f"    decision={baseline_cert['decision']} non_evidentiary={baseline_cert['non_evidentiary']}")
    print(f"    valid_replicates={baseline_cert['sufficiency_summary']['valid_replicates']}"
          f"/{baseline_cert['sufficiency_summary']['requested_replicates']}")

    print("\n[2/4] EV-NOISE (smoke)...")
    noise_cert = run_ev_noise(
        run_mode="smoke", requested_reruns=3, minimum_valid_reruns=2,
        output_dir=output_root / "ev_noise",
    )
    _write(output_root / "ev_noise_certificate.json", noise_cert)
    print(f"    decision={noise_cert['decision']} non_evidentiary={noise_cert['non_evidentiary']}")

    print("\n[3/4] H-EV-SHUFFLE (smoke)...")
    shuffle_firewall = TestSetFirewall(experiment_id="H-EV-SHUFFLE")
    shuffle_firewall.freeze(FreezeRecord(
        frozen_config_hash=config_hash({"experiment": "H-EV-SHUFFLE", "mode": "smoke"}),
        frozen_permutation_hash=None,
        frozen_replicate_plan_hash=config_hash({"requested_pairs": 3, "minimum_valid_pairs": 2}),
        frozen_statistical_plan_hash=config_hash({"alpha": 0.05}),
        frozen_at_utc=utc_timestamp(),
    ))
    shuffle_cert = run_h_ev_shuffle(
        run_mode="smoke", requested_pairs=3, minimum_valid_pairs=2,
        output_dir=output_root / "h_ev_shuffle", firewall=shuffle_firewall,
    )
    _write(output_root / "h_ev_shuffle_certificate.json", shuffle_cert)
    print(f"    decision={shuffle_cert['decision']} non_evidentiary={shuffle_cert['non_evidentiary']}")

    print("\n[4/4] H-EV-REPRESENTATION (smoke)...")
    permutation = generate_candidate1_permutation(np.random.default_rng(999))
    X_probe, Y_probe = speck.make_train_data(2000, BASELINE.rounds, diff=BASELINE.differential)
    validation = run_full_validation(X_probe, Y_probe, permutation)
    assert validation["all_passed"], validation
    permutation.save(output_root / "candidate1_permutation.json")

    representation_firewall = TestSetFirewall(experiment_id="H-EV-REPRESENTATION")
    representation_firewall.freeze(FreezeRecord(
        frozen_config_hash=config_hash({"experiment": "H-EV-REPRESENTATION", "mode": "smoke"}),
        frozen_permutation_hash=permutation.hash,
        frozen_replicate_plan_hash=config_hash({"requested_pairs": 3, "minimum_valid_pairs": 2}),
        frozen_statistical_plan_hash=config_hash({"alpha": 0.05}),
        frozen_at_utc=utc_timestamp(),
    ))
    representation_cert = run_h_ev_representation(
        run_mode="smoke", requested_pairs=3, minimum_valid_pairs=2,
        output_dir=output_root / "h_ev_representation", permutation=permutation,
        firewall=representation_firewall,
    )
    _write(output_root / "h_ev_representation_certificate.json", representation_cert)
    print(f"    decision={representation_cert['decision']} non_evidentiary={representation_cert['non_evidentiary']}")
    print(f"    permutation_hash={permutation.hash}")

    print("\n[+] Applying Holm correction across the primary family "
          "{H-EV-SHUFFLE, H-EV-REPRESENTATION} (smoke)...")
    combined = apply_primary_family_correction(shuffle_cert, representation_cert)
    _write(output_root / "primary_family_certificate.json", combined)
    print(f"    H-EV-SHUFFLE: raw_p={shuffle_cert['statistics']['raw_p_value']} "
          f"adjusted_p={shuffle_cert['statistics']['adjusted_p_value']} decision={shuffle_cert['decision']}")
    print(f"    H-EV-REPRESENTATION: raw_p={representation_cert['statistics']['raw_p_value']} "
          f"adjusted_p={representation_cert['statistics']['adjusted_p_value']} decision={representation_cert['decision']}")

    print("\n" + "=" * 70)
    print("SMOKE RUN COMPLETE. ALL RESULTS ABOVE ARE NON_EVIDENTIARY.")
    print(f"Certificates written under: {output_root}")
    print("=" * 70)
    return 0


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    sys.exit(main())
