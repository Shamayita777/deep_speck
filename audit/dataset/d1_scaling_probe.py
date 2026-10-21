"""Measure D1's cost as a function of partition size.

NOT an audit and NOT evidence. This exists to answer one engineering question
before a production run is launched: what wall-clock time and peak memory does
the UNCHANGED generic D1 engine need at the declared case-study scale
(10^7 / 10^6 / 10^6), so that the run is sized rather than guessed at.

The engine is called exactly as the production experiment calls it. Nothing is
optimised, replaced or bypassed; if the measurement says the declared scale
does not fit a given machine, that is the finding, not a reason to change the
engine.

Method
------
D1's per-sample work is a SHA-256 over the row's raw bytes, accumulated in
Python dicts/sets keyed by 64-character hex strings. Both time and memory are
therefore expected to be linear in the number of samples. The probe measures
several sizes and reports the observed scaling exponent, so the extrapolation
to 10^7 is checked against data rather than assumed to be linear.

Peak memory is read from /proc/self/status VmHWM (peak resident set size),
which is a high-water mark for the whole process and so includes the arrays
themselves. Each size runs in a FRESH subprocess, because VmHWM never
decreases within a process and a shared process would report the largest
previous size for every subsequent measurement.

Usage
    python d1_scaling_probe.py --sizes 100000 200000 400000      # parent
    python d1_scaling_probe.py --child 400000                    # one point
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (
            (candidate / "audit" / "dataset").is_dir()
            and (candidate / "audit" / "dataset" / "canonical.py").is_file()
        ):
            return candidate

    raise RuntimeError(
        f"Could not locate repository root from {start}"
    )


REPO = find_repo_root(Path(__file__).resolve().parent)

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


#: How peak memory was obtained, recorded in the output so a reader knows
#: which measurement they are looking at rather than assuming.
MEMORY_SOURCE = "unknown"


def peak_rss_kib() -> int:
    """Peak resident set size in KiB, with a portable fallback.

    `/proc/self/status` VmHWM is a true process high-water mark and is
    preferred, but it exists only on Linux. `resource.getrusage` is available
    on Linux and macOS and also reports a high-water mark, with one trap: it
    is in KILOBYTES on Linux and in BYTES on macOS. That is handled here
    rather than silently reporting a figure 1024x wrong on one platform.

    Returns -1 when no measurement is available, and the caller records that
    rather than substituting a plausible-looking number.
    """
    global MEMORY_SOURCE
    status = Path("/proc/self/status")
    if status.is_file():
        try:
            for line in status.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    MEMORY_SOURCE = "/proc/self/status VmHWM"
                    return int(line.split()[1])
        except OSError:
            pass
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            MEMORY_SOURCE = "resource.ru_maxrss (bytes, macOS)"
            return int(raw) // 1024
        MEMORY_SOURCE = "resource.ru_maxrss (KiB)"
        return int(raw)
    except Exception:                                          # noqa: BLE001
        MEMORY_SOURCE = "unavailable"
        return -1


def current_rss_kib() -> int:
    """Current RSS in KiB, or -1 where it cannot be read.

    Only /proc exposes a current (as opposed to peak) figure portably enough
    to be worth reading; elsewhere this returns -1 and the field is reported
    as unavailable instead of being filled with the peak.
    """
    status = Path("/proc/self/status")
    if not status.is_file():
        return -1
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


def child(n: int, feature_bits: int, val_frac: float) -> dict:
    """One measurement point, in its own process."""
    from audit.dataset import d1_duplicate_detection as d1

    n_val = max(1, int(n * val_frac))
    rng = np.random.default_rng(20261019)

    # Synthetic uint8 {0,1} data with AutoND's feature width. The probe is
    # measuring the ENGINE's cost per row, which depends on row width and row
    # count, not on the generator. Using synthetic data keeps the probe off
    # the audited subject entirely and makes it reproducible from a seed,
    # which the real AutoND generator is not.
    X_tr = rng.integers(0, 2, size=(n, feature_bits), dtype=np.uint8)
    Y_tr = rng.integers(0, 2, size=n, dtype=np.uint8)
    X_va = rng.integers(0, 2, size=(n_val, feature_bits), dtype=np.uint8)
    Y_va = rng.integers(0, 2, size=n_val, dtype=np.uint8)
    X_te = rng.integers(0, 2, size=(n_val, feature_bits), dtype=np.uint8)
    Y_te = rng.integers(0, 2, size=n_val, dtype=np.uint8)

    data_bytes = sum(a.nbytes for a in (X_tr, Y_tr, X_va, Y_va, X_te, Y_te))
    rss_before = current_rss_kib()
    gc.collect()

    t0 = time.perf_counter()
    results = d1.audit_duplicates(
        X_tr, X_va, X_te, Y_tr, Y_va, Y_te,
        expected_feature_schema={"ndim": 2, "columns": feature_bits,
                                 "dtype": "uint8", "value_set": [0, 1]},
        expected_label_schema={"ndim": 1, "dtype": "uint8",
                               "value_set": [0, 1]},
        representation_convention="probe-synthetic",
        feature_encoding="binary-bits-uint8",
    )
    elapsed = time.perf_counter() - t0
    verdict = d1.evaluate_d1(results)

    return {
        "train_samples": n,
        "validation_samples": n_val,
        "test_samples": n_val,
        "total_rows": n + 2 * n_val,
        "feature_bits": feature_bits,
        "array_bytes": data_bytes,
        "elapsed_s": round(elapsed, 3),
        "us_per_row": round(1e6 * elapsed / (n + 2 * n_val), 3),
        "peak_rss_kib": peak_rss_kib(),
        "memory_source": MEMORY_SOURCE,
        "rss_before_audit_kib": rss_before,
        "engine_overhead_kib": peak_rss_kib() - (data_bytes // 1024),
        "verdict": str(verdict),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--child", type=int, default=None)
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[50_000, 100_000, 200_000, 400_000])
    p.add_argument("--feature-bits", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="val and test each this fraction of train, matching "
                        "the 10:1:1 ratio of the declared case study")
    p.add_argument("--out", type=Path, default=None,
                   help="where to write the JSON result. Defaults to "
                        "<repo>/artifacts/d1_scaling_probe.json; pass an "
                        "explicit path to write elsewhere. No machine-specific "
                        "path is baked in.")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = REPO / "artifacts" / "d1_scaling_probe.json"

    if args.child is not None:
        rec = child(args.child, args.feature_bits, args.val_frac)
        print("@@JSON@@" + json.dumps(rec))
        return 0

    points = []
    for n in args.sizes:
        # No -I/-S/-E here: -I and -E discard PYTHONPATH, and -S skips `site`,
        # which removes site-packages and so numpy. The child must import both
        # the repo (via PYTHONPATH) and the installed packages.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO)
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--child", str(n), "--feature-bits", str(args.feature_bits),
               "--val-frac", str(args.val_frac)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        tag = [l for l in proc.stdout.splitlines() if l.startswith("@@JSON@@")]
        if not tag:
            print(f"n={n}: FAILED\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            return 1
        rec = json.loads(tag[0][len("@@JSON@@"):])
        points.append(rec)
        print(f"  n_train={rec['train_samples']:>9,}  "
              f"rows={rec['total_rows']:>9,}  "
              f"t={rec['elapsed_s']:>8.2f}s  "
              f"{rec['us_per_row']:>6.2f} us/row  "
              f"peakRSS={rec['peak_rss_kib']/1024:>8.1f}MB  "
              f"arrays={rec['array_bytes']/2**20:>7.1f}MB  "
              f"{rec['verdict']}")

    # ------------------------------------------------------------------
    # Analysis.
    #
    # A log-log exponent is the WRONG model here and an earlier version of
    # this probe reported one. Both series carry a large fixed offset -- the
    # interpreter, numpy and the import graph are ~34 MB of resident set
    # before a single row is hashed -- so a power-law fit through the raw
    # points reads as sublinear (exponent ~0.66 for memory) when the marginal
    # cost per row is in fact near-constant. Fitting `y = a + b*rows` instead
    # separates the offset from the per-row cost, and the MARGINAL cost
    # between consecutive points is printed so the reader can see directly
    # whether it is constant, rising or falling rather than trusting a fit.
    # ------------------------------------------------------------------
    rows = np.array([pt["total_rows"] for pt in points], dtype=float)
    secs = np.array([pt["elapsed_s"] for pt in points], dtype=float)
    peak = np.array([pt["peak_rss_kib"] for pt in points], dtype=float)
    out: dict = {"points": points,
                 "memory_source": points[0].get("memory_source", "unknown")}
    if any(int(pt["peak_rss_kib"]) < 0 for pt in points):
        out["memory_measurement_unavailable"] = (
            "peak RSS could not be measured on this platform; the memory "
            "projection below is meaningless and must not be used for sizing.")

    marginals = []
    for i in range(1, len(points)):
        dr = rows[i] - rows[i - 1]
        marginals.append({
            "from_rows": int(rows[i - 1]),
            "to_rows": int(rows[i]),
            "marginal_us_per_row": round(1e6 * (secs[i] - secs[i - 1]) / dr, 2),
            "marginal_bytes_per_row": round(
                1024 * (peak[i] - peak[i - 1]) / dr, 1),
        })
    out["marginal_costs"] = marginals

    if len(points) >= 3:
        bt, at = np.polyfit(rows, secs, 1)          # secs = at + bt*rows
        bm, am = np.polyfit(rows, peak, 1)          # KiB  = am + bm*rows
        target_rows = 12_000_000                    # 10^7 + 10^6 + 10^6
        out["target_rows"] = target_rows
        out["affine_time"] = {
            "offset_s": round(float(at), 3),
            "us_per_row": round(1e6 * float(bt), 3)}
        out["affine_memory"] = {
            "offset_mib": round(float(am) / 1024, 1),
            "bytes_per_row": round(1024 * float(bm), 1)}
        out["affine_projection"] = {
            "seconds": round(float(at + bt * target_rows), 1),
            "peak_rss_gib": round(float(am + bm * target_rows) / 2**20, 2)}
        # The per-row TIME cost is not constant -- dict growth and cache
        # misses make it rise -- so also project using the LAST observed
        # marginal rate, which is the pessimistic end of what was measured.
        last = marginals[-1]
        out["last_marginal_projection"] = {
            "basis": f"marginal rate measured from {last['from_rows']:,} to "
                     f"{last['to_rows']:,} rows",
            "seconds": round(
                float(secs[-1]) + last["marginal_us_per_row"] * 1e-6
                * (target_rows - rows[-1]), 1),
            "peak_rss_gib": round(
                (float(peak[-1]) * 1024 + last["marginal_bytes_per_row"]
                 * (target_rows - rows[-1])) / 2**30, 2)}
        out["extrapolation_caveat"] = (
            "Both figures are PROJECTIONS, not measurements. The per-row time "
            "cost was observed to RISE with size, so the affine fit is "
            "optimistic and the last-marginal figure is the better planning "
            "number. Memory can depart from either once dict resizing or "
            "swapping begins, and neither projection models swap.")

        print()
        print("  marginal cost between consecutive points"
              "  (watch whether it is constant):")
        for m in marginals:
            print(f"    {m['from_rows']:>9,} -> {m['to_rows']:>9,} rows:  "
                  f"{m['marginal_us_per_row']:>6.2f} us/row   "
                  f"{m['marginal_bytes_per_row']:>7.1f} B/row")
        print()
        print(f"  affine fit:  time = {out['affine_time']['offset_s']} s + "
              f"{out['affine_time']['us_per_row']} us/row")
        print(f"               mem  = "
              f"{out['affine_memory']['offset_mib']} MiB + "
              f"{out['affine_memory']['bytes_per_row']} B/row")
        print()
        print(f"  PROJECTED at {target_rows:,} rows (1.2e7 = the declared "
              f"case study):")
        print(f"    affine fit          "
              f"{out['affine_projection']['seconds']/60:6.1f} min   "
              f"{out['affine_projection']['peak_rss_gib']:5.2f} GiB")
        print(f"    last marginal rate  "
              f"{out['last_marginal_projection']['seconds']/60:6.1f} min   "
              f"{out['last_marginal_projection']['peak_rss_gib']:5.2f} GiB"
              f"   <- planning number")
        print("    (projections, not measurements; per-row time was observed "
              "to rise with size)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
