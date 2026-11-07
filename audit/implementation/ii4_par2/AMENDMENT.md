# II-4-GOHR-IMPL-V1-PAR2: readiness-gate memory correction (gate-only amendment)

**Amends:**
- `ii-4-gohr-impl-v1.frozen.json` (`ab02d471…`);
- PAR1, `ii-4-gohr-impl-v1-par1.frozen.json` (`457fa3c7…`).

**Record:** `audit/implementation/preregistrations/ii-4-gohr-impl-v1-par2.frozen.json`.

## The defect

**Where:** the frozen `audit/implementation/preflight.py`, `memory_report`. It measures available memory as `os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")`.

**What that measures:** on Linux glibc, `sysinfo().freeram`, which is the kernel's free-page count (`MemFree`). It leaves out page cache that the kernel reclaims on demand, and it ignores the container's cgroup limit.

**What the operator reported on Kaggle:**

| quantity | value |
|---|---|
| preflight "available" | 0.24 GB |
| `MemFree` | about 237 MiB |
| `MemAvailable` | about 29 GiB |
| psutil available | about 29.85 GiB |
| cgroup v2 `memory.max` | 30 GiB |
| cgroup v2 `memory.current` | about 25.88 GiB |
| `inactive_file` | about 19.08 GiB |
| `memory.pressure` | zero |
| `memory.events` oom / max | zero |

**Effect:** the readiness gate refused a machine that had memory.

## What PAR2 changes

PAR2 changes only the readiness gate.

**The gate (`ii4_par2/gate.py`):**
1. It calls the frozen `run_ii4_impl.readiness_gate` exactly as the PAR1 launcher does.
2. It removes only the problems that the frozen memory report produced.
3. It requires the corrected measurement (`ii4_par2/memory.py`) in their place, against the same frozen requirement of 8.0 GB (`DESIGN_RATIONALE.compute_estimate.memory_measured.recommended_free_host_gb`).

Every other frozen refusal stands. The frozen memory report is kept verbatim.

**The measurement**, in bytes:

```
U(phi) = min(MemAvailable,
             min over visible cgroup levels with a finite limit of
               max(0, min(memory.max, memory.high) - memory.current)
               + phi * max(0, inactive_file - file_dirty - file_writeback - file_mapped))
```

**The gating value, φ = 0.5:**
- It is a conservative policy fixed before use. It is not estimated from data.
- Active file cache, reclaimable slab and swap are never credited.
- U(0) and U(1) are recorded as a sensitivity analysis.

**Blocking and no-credit rules:**

| condition | effect |
|---|---|
| `oom` or `oom_kill` > 0 | blocks |
| any pressure `full` average > 0 | blocks |
| `MemAvailable` absent | blocks |
| cgroup v1 | blocks |
| v2 named but not mounted | blocks |
| a limit without `memory.current` | blocks |
| any unparseable interface file | blocks |
| `max` or `high` events > 0 | withdraws the cache credit |
| pressure `some` avg60 > 0 | withdraws the cache credit |
| an unreadable `memory.stat`, `memory.events` or `memory.pressure` | withdraws the cache credit |

**The concurrent memory probe (`ii4_par2/memprobe.py`):**
- **Why it exists:** the 8.0 GB requirement came from a 4.0 GB peak estimate for one process. PAR1 runs two workers at once, and the PAR1 amendment never revisited memory.
- **What it runs:** one child per GPU, concurrently. Each child runs the frozen adapter's real path at scale 1.0 (block data generation and loading, then one epoch of `train_arm` for both arms).
- **What it records:**
  - each child's `VmHWM` and its sampled RSS peak;
  - cgroup `memory.peak` before and after;
  - the sampled peak of `memory.current`;
  - events and pressure.
- **When it passes:** when both children complete and no oom or oom_kill count rises.
- **Execution requires** a passing probe from the same boot.
- **The requirement is not changed.**

**Records:** every gate decision, probe and execution start is appended to `<run dir>.par2_gate.jsonl`, a SHA-256 hash chain kept beside the run directory. `ii4_par2/verify_par2.py` (standard library only) checks it, then runs the frozen PAR1 verifier.

## What PAR2 does not change

PAR2 leaves untouched:
- every PAR1 and original frozen source, record and manifest;
- the scheduler, worker, runner and adapter;
- the plan, design, seeds and decision rules;
- the configuration fingerprint (`62273336…`);
- the PAR1 experiment-fingerprint rule.

A run executed through PAR2 therefore has exactly the PAR1 execution identity, and the frozen PAR1 verifier applies to it unchanged. PAR2 is bound by its own record and its gate chain, not by the experiment fingerprint.

**Not covered:** the original sequential `run_ii4_impl --execute` path still uses the frozen memory measurement. PAR2 corrects only the PAR1 parallel launcher path.
