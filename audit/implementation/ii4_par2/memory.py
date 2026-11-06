"""
Cgroup-v2-aware host-memory availability for the II-4 readiness gate (PAR2).

WHY THIS EXISTS. The frozen preflight (audit/implementation/preflight.py,
memory_report) reads os.sysconf("SC_AVPHYS_PAGES"). On Linux glibc that is
sysinfo().freeram, the kernel's count of free pages (MemFree). It leaves out
page cache the kernel reclaims on demand, and it knows nothing of the
container's cgroup limit. On Kaggle it reported 0.24 GB while MemAvailable
was about 29 GiB, so a correct machine was refused. This module replaces that
one measurement; it does not change the requirement it is compared with.

THE MEASUREMENT (all quantities in bytes)

    U(phi) = min(H, G(phi))
    H      = MemAvailable from /proc/meminfo
    G(phi) = min over every visible cgroup level with a finite limit of
               max(0, limit - memory.current)
             + phi * max(0, inactive_file - file_dirty - file_writeback - file_mapped)
    limit  = min(memory.max, memory.high); the literal "max" is no limit

The gate clears when U(PHI) >= the frozen requirement and no blocking rule
fires. PHI = 0.5 is a stated conservative POLICY, not a measured quantity:

  * only INACTIVE file cache is credited; active file cache, reclaimable slab
    and swap are never counted;
  * dirty, under-writeback and mapped file pages are subtracted first, because
    they cannot be dropped without I/O or unmapping;
  * the remainder counts at half value, allowing for cache that is the run's
    own block data (re-read from disk if it is reclaimed) or otherwise costly
    to reclaim;
  * H and every cgroup level are taken at their minimum, and memory.high
    counts as a limit because reclaim throttling starts there.

U(0) and U(1) are always recorded as a sensitivity analysis; only U(PHI) gates.

RULES THAT OVERRIDE THE CREDIT OR THE DECISION (any level, any file present)

  block      memory.events oom > 0 or oom_kill > 0    (the container already killed)
  block      memory.pressure "full" avg10/avg60/avg300 > 0
             (every runnable task stalled on memory while the machine is idle)
  no credit  memory.events max > 0 or high > 0        (the limit is already binding)
  no credit  memory.pressure "some" avg60 > 0         (reclaim is already costing time)
  no credit  memory.stat, memory.events or memory.pressure unreadable

FAIL-CLOSED CASES

  block      MemAvailable absent or unparseable (never falls back to MemFree or MemTotal)
  block      a cgroup v1 memory controller (outside this measurement's scope)
  block      a v2 cgroup path in /proc/self/cgroup but no v2 hierarchy at the mount
  block      a finite limit whose memory.current is absent or unparseable
  block      any interface file present but unparseable
  disclosed  /proc/self/cgroup absent: no cgroup is visible; H alone governs

Units follow the frozen preflight: a "GB" is 10^9 bytes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

MEASUREMENT_VERSION = "ii4-par2-memory-1"
PHI = 0.5
SENSITIVITY_PHIS = (0.0, 0.5, 1.0)

CGROUP_FILES = ("memory.max", "memory.high", "memory.current", "memory.stat", "memory.events",
                "memory.pressure", "memory.peak", "memory.swap.max", "memory.swap.current",
                "memory.min", "memory.low", "cgroup.controllers")
RECLAIM_SUBTRACT = ("file_dirty", "file_writeback", "file_mapped")


class Malformed(ValueError):
    pass


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None


def read_sources(proc_root: str = "/proc", cgroup_root: str = "/sys/fs/cgroup") -> dict[str, Any]:
    """Raw text of every file the measurement uses, from this process's own cgroup up to the mount."""
    proc, cg = Path(proc_root), Path(cgroup_root)
    raw: dict[str, Any] = {
        "proc_root": str(proc), "cgroup_root": str(cg),
        "proc_meminfo": _read(proc / "meminfo"),
        "proc_self_cgroup": _read(proc / "self" / "cgroup"),
        "proc_pressure_memory": _read(proc / "pressure" / "memory"),
        "boot_id": (_read(proc / "sys" / "kernel" / "random" / "boot_id") or "").strip() or None,
        "cgroup_v2_mounted": (cg / "cgroup.controllers").is_file(),
        "levels": [],
    }
    rel = _v2_path(raw["proc_self_cgroup"])
    if rel is not None and raw["cgroup_v2_mounted"]:
        parts = [p for p in rel.split("/") if p]
        for depth in range(len(parts), -1, -1):
            sub = "/" + "/".join(parts[:depth])
            d = cg.joinpath(*parts[:depth])
            raw["levels"].append({"path": sub, "visible": d.is_dir(),
                                  "files": {f: _read(d / f) for f in CGROUP_FILES} if d.is_dir() else {}})
    return raw


def _v2_path(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        hid, _, rest = line.partition(":")
        ctrl, _, path = rest.partition(":")
        if hid == "0" and ctrl == "":
            return path.strip() or "/"
    return None


def cgroup_mode(raw: dict[str, Any]) -> str:
    text = raw.get("proc_self_cgroup")
    if not text:
        return "none"
    v1_memory = any(line.split(":")[0] != "0" and "memory" in line.split(":")[1].split(",")
                    for line in text.splitlines() if line.count(":") >= 2)
    if v1_memory:
        return "v1"
    if _v2_path(text) is None:
        return "none"
    return "v2" if raw.get("cgroup_v2_mounted") else "v2-unmounted"


# --- parsers: each raises Malformed rather than guessing ---------------------------------

def parse_meminfo(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        key, sep, rest = line.partition(":")
        fields = rest.split()
        if not sep or not fields or not fields[0].isdigit():
            raise Malformed(f"/proc/meminfo line {line!r}")
        unit = fields[1] if len(fields) > 1 else ""
        out[key.strip()] = int(fields[0]) * (1024 if unit == "kB" else 1)
    return out


def parse_limit(text: str) -> Optional[int]:
    s = text.strip()
    if s == "max":
        return None
    if not s.isdigit():
        raise Malformed(f"limit {s!r}")
    return int(s)


def parse_int(text: str) -> int:
    s = text.strip()
    if not s.isdigit():
        raise Malformed(f"integer {s!r}")
    return int(s)


def parse_keyed(text: str, what: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        f = line.split()
        if len(f) != 2 or not f[1].lstrip("-").isdigit():
            raise Malformed(f"{what} line {line!r}")
        out[f[0]] = int(f[1])
    return out


def parse_pressure(text: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        kind, *fields = line.split()
        if kind not in ("some", "full"):
            raise Malformed(f"memory.pressure line {line!r}")
        vals = {}
        for f in fields:
            k, sep, v = f.partition("=")
            try:
                vals[k] = float(v)
            except ValueError:
                raise Malformed(f"memory.pressure field {f!r}") from None
            if not sep:
                raise Malformed(f"memory.pressure field {f!r}")
        out[kind] = vals
    return out


# --- the assessment ---------------------------------------------------------------------

def _level(lv: dict[str, Any], blocks: list, notes: list) -> Optional[dict[str, Any]]:
    """One cgroup level's terms, or None when it imposes no limit."""
    files, path = lv.get("files") or {}, lv["path"]
    out: dict[str, Any] = {"path": path, "credit_allowed": True, "no_credit_because": []}

    def get(name, parser, *args):
        text = files.get(name)
        if text is None:
            return None
        try:
            return parser(text, *args)
        except Malformed as exc:
            blocks.append(f"cgroup {path}: {name} unparseable ({exc})")
            return None

    mx, hi = get("memory.max", parse_limit), get("memory.high", parse_limit)
    out["memory_max"], out["memory_high"] = mx, hi
    out["memory_peak"] = get("memory.peak", parse_int)
    out["swap_max"] = get("memory.swap.max", parse_limit)
    out["swap_current"] = get("memory.swap.current", parse_int)
    out["memory_min"], out["memory_low"] = get("memory.min", parse_int), get("memory.low", parse_int)
    events = get("memory.events", parse_keyed, "memory.events")
    pressure = get("memory.pressure", parse_pressure)
    stat = get("memory.stat", parse_keyed, "memory.stat")
    out["events"], out["pressure"], out["stat"] = events, pressure, stat

    if events is None:
        out["no_credit_because"].append("memory.events unreadable")
    else:
        if events.get("oom", 0) > 0 or events.get("oom_kill", 0) > 0:
            blocks.append(f"cgroup {path}: memory.events records oom={events.get('oom', 0)} "
                          f"oom_kill={events.get('oom_kill', 0)}")
        if events.get("max", 0) > 0 or events.get("high", 0) > 0:
            out["no_credit_because"].append(
                f"memory.events max={events.get('max', 0)} high={events.get('high', 0)}: the limit is binding")
    if pressure is None:
        out["no_credit_because"].append("memory.pressure unreadable")
    else:
        full = pressure.get("full", {})
        if any(full.get(k, 0.0) > 0.0 for k in ("avg10", "avg60", "avg300")):
            blocks.append(f"cgroup {path}: memory.pressure full {full}")
        if pressure.get("some", {}).get("avg60", 0.0) > 0.0:
            out["no_credit_because"].append(f"memory.pressure some avg60={pressure['some']['avg60']}")
    if stat is None:
        out["no_credit_because"].append("memory.stat unreadable")

    finite = [v for v in (mx, hi) if v is not None]
    if not finite:
        notes.append(f"cgroup {path}: no finite memory.max or memory.high")
        return None
    out["limit"] = min(finite)
    cur = get("memory.current", parse_int)
    if cur is None:
        if files.get("memory.current") is None:
            blocks.append(f"cgroup {path}: finite limit {out['limit']} but memory.current is unreadable")
        return None
    out["current"] = cur
    out["free"] = max(0, out["limit"] - cur)
    if stat is not None:
        if "inactive_file" not in stat:
            out["no_credit_because"].append("memory.stat has no inactive_file")
            out["reclaimable"] = 0
        else:
            sub = {k: stat.get(k, 0) for k in RECLAIM_SUBTRACT}
            missing = [k for k in RECLAIM_SUBTRACT if k not in stat]
            if missing:
                out["no_credit_because"].append(f"memory.stat lacks {missing}")
            out["subtracted"] = sub
            out["reclaimable"] = max(0, stat["inactive_file"] - sum(sub.values()))
    else:
        out["reclaimable"] = 0
    out["credit_allowed"] = not out["no_credit_because"]
    out["usable"] = {str(p): out["free"] + (p * out["reclaimable"] if out["credit_allowed"] else 0)
                     for p in SENSITIVITY_PHIS}
    out["usable_formula_only"] = {str(p): out["free"] + p * out["reclaimable"] for p in SENSITIVITY_PHIS}
    return out


def assess(raw: dict[str, Any], *, required_bytes: int, phi: float = PHI) -> dict[str, Any]:
    """The corrected measurement and its decision. Pure: depends only on `raw`."""
    if phi not in SENSITIVITY_PHIS:
        raise ValueError(f"phi {phi} is not one of the recorded values {SENSITIVITY_PHIS}")
    blocks: list[str] = []
    notes: list[str] = []
    out: dict[str, Any] = {"measurement": MEASUREMENT_VERSION, "phi": phi,
                           "required_bytes": int(required_bytes), "required_gb": required_bytes / 1e9,
                           "boot_id": raw.get("boot_id")}

    host = None
    if raw.get("proc_meminfo") is None:
        blocks.append("/proc/meminfo unreadable")
    else:
        try:
            mi = parse_meminfo(raw["proc_meminfo"])
            out["meminfo"] = {k: mi.get(k) for k in ("MemTotal", "MemFree", "MemAvailable", "Buffers",
                                                     "Cached", "Inactive(file)", "Active(file)", "Dirty",
                                                     "Writeback", "Shmem", "SReclaimable")}
            host = mi.get("MemAvailable")
            if host is None:
                blocks.append("/proc/meminfo has no MemAvailable; MemFree is not a substitute")
        except Malformed as exc:
            blocks.append(f"/proc/meminfo unparseable ({exc})")
    out["host_mem_available"] = host

    mode = cgroup_mode(raw)
    out["cgroup_mode"] = mode
    levels = []
    if mode == "v1":
        blocks.append("cgroup v1 memory controller: outside this measurement's scope")
    elif mode == "v2-unmounted":
        blocks.append("a v2 cgroup is named in /proc/self/cgroup but no v2 hierarchy is mounted at "
                      f"{raw.get('cgroup_root')}; a limit may exist and cannot be read")
    elif mode == "none":
        notes.append("no cgroup is visible; MemAvailable alone governs")
    else:
        for lv in raw.get("levels") or []:
            if not lv.get("visible"):
                notes.append(f"cgroup {lv['path']} is not visible from this namespace")
                continue
            res = _level(lv, blocks, notes)
            if res is not None:
                levels.append(res)
    out["cgroup_levels"] = levels

    def u(p: float, key: str = "usable") -> Optional[int]:
        g = [lv[key][str(p)] for lv in levels]
        cands = [x for x in [host, min(g) if g else None] if x is not None]
        return min(cands) if cands else None

    out["usable_bytes"] = {str(p): u(p) for p in SENSITIVITY_PHIS}
    out["usable_bytes_formula_only"] = {str(p): u(p, "usable_formula_only") for p in SENSITIVITY_PHIS}
    out["usable_gb"] = {k: (v / 1e9 if v is not None else None) for k, v in out["usable_bytes"].items()}
    out["sensitivity_clear"] = {k: (v is not None and v >= required_bytes and not blocks)
                                for k, v in out["usable_bytes"].items()}
    gating = out["usable_bytes"][str(phi)]
    if gating is None and not blocks:
        blocks.append("no memory quantity could be measured")
    elif gating is not None and gating < required_bytes:
        blocks.append(f"usable memory {gating / 1e9:.3f} GB at phi={phi} is below the requirement "
                      f"{required_bytes / 1e9:.3f} GB")
    out["blocks"], out["notes"] = blocks, notes
    out["clear"] = not blocks
    return out


def measure(*, required_bytes: int, proc_root: str = "/proc",
            cgroup_root: str = "/sys/fs/cgroup") -> dict[str, Any]:
    raw = read_sources(proc_root, cgroup_root)
    result = assess(raw, required_bytes=required_bytes)
    result["raw"] = raw
    result["sysconf_avphys_bytes"] = _sysconf_avphys()
    return result


def _sysconf_avphys() -> Optional[int]:
    """What the frozen preflight reads, recorded beside the corrected value for comparison."""
    try:
        if "SC_AVPHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    return None
