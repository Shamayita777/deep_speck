"""
Frozen, balanced A/B execution order for the paired AutoND EV hypotheses.

Within a pair both arms share data, seed and verified initial weights and run in one process on one GPU,
one after the other. Training them always A-then-B would confound B - A with execution position (cache,
autotuning or thermal state of the second training). Protocol `balanced-ab-v1` therefore assigns each pair
an order, "AB" or "BA", with exactly floor(n/2) "BA" pairs and the rest "AB":

    tag  = "autond-ev-ab-order-v1|<experiment>|design=<design_hash>|pairs=<n>"
    seed = int(SHA-256(tag)[:16 hex digits], 16)
    order = numpy.random.Generator(PCG64(seed)).permutation(["AB"] * ceil(n/2) + ["BA"] * floor(n/2))

The schedules for calibration and production are frozen in `configs/autond_ev_order_schedule.json`, whose
SHA-256 is pinned below; the runner never draws a schedule at execution time for those tiers. It reads the
frozen file, checks it against the pin and against the derivation, and refuses on any mismatch. Smoke runs
are non-evidentiary and use the derivation directly for their (tiny) unit counts.

The protocol identifier is recorded in every paired unit, certificate, variance artifact and power plan, so
a variance estimate obtained under one execution order can never size a production design run under another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL = "balanced-ab-v1"
LEGACY_PROTOCOL = "fixed-ab-legacy"          # what pre-protocol calibrations did: A then B in every pair
SCHEDULE_PATH = Path(__file__).resolve().parents[1] / "configs" / "autond_ev_order_schedule.json"
SCHEDULE_ARTIFACT = "autond-ev-order-schedule-1"
SCHEDULE_SHA256 = "72ae907d278b424408c8bde882f2166988a1d909d20d1334d96509d5cb3e8696"
ORDERS = ("AB", "BA")


class OrderError(RuntimeError):
    pass


def derivation_tag(experiment_id: str, design: str, pairs: int) -> str:
    return f"autond-ev-ab-order-v1|{experiment_id}|design={design}|pairs={pairs}"


def derive(experiment_id: str, design: str, pairs: int) -> list[str]:
    if not (isinstance(pairs, int) and pairs >= 1):
        raise OrderError(f"pairs must be a positive integer, got {pairs!r}")
    seed = int(hashlib.sha256(derivation_tag(experiment_id, design, pairs).encode()).hexdigest()[:16], 16)
    base = ["AB"] * (pairs - pairs // 2) + ["BA"] * (pairs // 2)
    return [str(x) for x in np.random.Generator(np.random.PCG64(seed)).permutation(base)]


def schedule_sha256(order: list[str]) -> str:
    return hashlib.sha256("|".join(order).encode()).hexdigest()


def load_frozen(path: Path = SCHEDULE_PATH, expected_sha256: str = SCHEDULE_SHA256) -> dict:
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != expected_sha256:
        raise OrderError(f"{path} has SHA-256 {got}; the pinned schedule is {expected_sha256}")
    frozen = json.loads(data)
    if frozen.get("artifact") != SCHEDULE_ARTIFACT or frozen.get("protocol") != PROTOCOL:
        raise OrderError(f"{path} is not a {SCHEDULE_ARTIFACT} for protocol {PROTOCOL}")
    return frozen


def schedule_for(experiment_id: str, mode: str, pairs: int, design: str, *, frozen: dict | None = None) -> list[str]:
    """The execution order of every pair of one paired experiment in one tier; refuses any inconsistency."""
    expected = derive(experiment_id, design, pairs)
    if mode == "smoke":
        return expected
    frozen = frozen if frozen is not None else load_frozen()
    if frozen.get("design_hash") != design:
        raise OrderError(f"the frozen schedule was derived for design {frozen.get('design_hash')}, not {design}")
    entry = (frozen.get("schedules") or {}).get(mode, {}).get(experiment_id)
    if entry is None:
        raise OrderError(f"no frozen {mode} schedule for {experiment_id}")
    order = entry.get("order")
    if not (isinstance(order, list) and len(order) == pairs and all(o in ORDERS for o in order)):
        raise OrderError(f"frozen {mode} schedule for {experiment_id} does not cover exactly {pairs} pairs")
    if order != expected or entry.get("tag") != derivation_tag(experiment_id, design, pairs) \
            or entry.get("schedule_sha256") != schedule_sha256(order):
        raise OrderError(f"frozen {mode} schedule for {experiment_id} differs from its documented derivation")
    if order.count("BA") != pairs // 2:
        raise OrderError(f"frozen {mode} schedule for {experiment_id} is not balanced")
    return order


def render(design: str, plans: dict[str, dict[str, int]]) -> dict:
    """The frozen artifact for {mode: {experiment: pairs}} (used once, to write the schedule file)."""
    return {"artifact": SCHEDULE_ARTIFACT, "protocol": PROTOCOL, "design_hash": design,
            "derivation": ("seed = int(SHA-256(tag)[:16 hex], 16); order = numpy PCG64(seed).permutation("
                           "['AB'] * ceil(n/2) + ['BA'] * floor(n/2)); tag = "
                           "'autond-ev-ab-order-v1|<experiment>|design=<design_hash>|pairs=<n>'"),
            "meaning": {"AB": "condition_a trains (and is evaluated) first, then condition_b",
                        "BA": "condition_b first, then condition_a"},
            "schedules": {mode: {e: {"pairs": n, "tag": derivation_tag(e, design, n),
                                     "order": derive(e, design, n),
                                     "schedule_sha256": schedule_sha256(derive(e, design, n))}
                                 for e, n in plan.items()}
                          for mode, plan in plans.items()}}
