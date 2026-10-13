#!/usr/bin/env python3
"""
Pre-production resource verification - GENERIC.

Answers, by measurement rather than assumption, the questions that
decide whether a 53-84 GPU-hour run can complete at all:

  WHICH DEVICE ACTUALLY TRAINS? Not which devices exist. A tiny
  training step is executed and its device placement is read back, so
  the answer is observed rather than inferred from
  `list_physical_devices`.

  HOW MANY DEVICES ARE USED? Reported from the runner's actual
  behaviour, which is SEQUENTIAL single-device training with no
  distribution strategy. This module will not describe a second GPU as
  contributing when it is idle.

  IS THERE ENOUGH DISK AND MEMORY? Checked against sizes the caller
  measures, not against sizes anyone guessed.

  WILL THE FILESYSTEM KEEP THE PROMISES THE RECOVERY DESIGN RELIES ON?
  The two-slot checkpoint protocol, the atomic commit manifest and the
  firewall's persistence all rest on two POSIX guarantees: `os.replace`
  is atomic within a filesystem, and `fsync` makes bytes durable. On a
  FUSE mount - Google Drive under Colab above all - neither is
  guaranteed: a rename can be a non-atomic copy-then-delete against a
  remote API, and a successful `fsync` can return before anything has
  left the machine. The SIGKILL tests measure these guarantees on the
  filesystem they run on and say nothing about any other, so the
  filesystem backing the output directory is identified and classified
  here, and an unrecognized one is REFUSED rather than assumed sound.

NO SPEEDUP IS CLAIMED and no training strategy is changed. Declaring a
`MirroredStrategy` to use a second GPU would alter the effective batch
size and the gradient-averaging semantics of a FROZEN protocol (batch
5000, Adam, MSE). That is a protocol change, not an optimization, and it
is out of scope for a run whose configuration is already registered.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

PREFLIGHT_VERSION = "ii4-preflight-v1"

#: What the runner actually does, stated so a reader does not have to
#: infer it from the absence of a strategy declaration.
TRAINING_TOPOLOGY = {
    "arms": "SEQUENTIAL - arm A trains to completion, then arm B",
    "blocks": "SEQUENTIAL - one block at a time",
    "distribution_strategy": None,
    "devices_used_per_fit": 1,
    "second_gpu": (
        "NOT USED. The runner declares no distribution strategy, so TensorFlow places the "
        "whole training step on one device. A second GPU sits idle for the entire run."),
    "no_speedup_claimed": (
        "This run takes the same wall-clock time on a two-GPU machine as on a one-GPU machine "
        "of the same model. Any schedule built on an assumed 2x speedup is wrong."),
    "why_not_changed": (
        "Using both GPUs would require a distribution strategy, which splits the global batch "
        "across replicas and averages gradients. At the frozen batch size of 5000 that changes "
        "the per-replica batch and the gradient semantics - a change to a registered protocol, "
        "not an optimization. It is therefore out of scope here."),
    "legitimate_parallelism": (
        "Two SEPARATE blocks could be run as two independent processes, one pinned to each GPU "
        "with CUDA_VISIBLE_DEVICES, because blocks are independent experimental units. That "
        "halves wall-clock without touching the protocol. It is NOT implemented here: two "
        "processes would write to one run directory and one state file, and making that safe "
        "needs per-block locking that has not been built or tested. Attempting it with the "
        "current runner risks corrupting the run state."),
}


def gpu_report() -> dict[str, Any]:
    """
    Which device actually performs a training step.

    A tiny model is built and fitted for one step with device placement
    logging read back from the op, because `list_physical_devices`
    reports what EXISTS, and a GPU can exist while training silently
    falls back to CPU - a 40x slowdown that turns 60 GPU-hours into
    weeks and is invisible until the run is far too late to restart.
    """
    out: dict[str, Any] = {"tensorflow": None, "keras": None, "physical_gpus": [],
                           "logical_gpus": [], "cuda_built": None,
                           "training_device": None, "training_is_on_gpu": None,
                           "problems": [], "warnings": [], "requires_acknowledgement": []}
    try:
        import tensorflow as tf
    except Exception as exc:                                     # noqa: BLE001
        out["problems"].append(f"TensorFlow is not importable: {type(exc).__name__}: {exc}")
        return out

    out["tensorflow"] = tf.__version__
    try:
        out["keras"] = tf.keras.__version__
    except Exception:                                            # noqa: BLE001
        pass
    try:
        out["cuda_built"] = bool(tf.test.is_built_with_cuda())
    except Exception:                                            # noqa: BLE001
        pass
    try:
        out["physical_gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
        out["logical_gpus"] = [d.name for d in tf.config.list_logical_devices("GPU")]
    except Exception as exc:                                     # noqa: BLE001
        out["problems"].append(f"could not enumerate GPUs: {exc}")

    # Observe where a real variable update lands.
    try:
        import numpy as np
        with tf.device(None):
            v = tf.Variable([[1.0, 2.0]], dtype=tf.float32)
            w = tf.Variable([[1.0], [1.0]], dtype=tf.float32)
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(tf.matmul(v, w))
            grads = tape.gradient(loss, [v, w])
            out["training_device"] = grads[0].device or v.device
        out["training_is_on_gpu"] = "GPU" in (out["training_device"] or "").upper()
    except Exception as exc:                                     # noqa: BLE001
        out["problems"].append(f"could not observe the training device: {exc}")

    if out["physical_gpus"] and out["training_is_on_gpu"] is False:
        out["problems"].append(
            f"a GPU is present ({out['physical_gpus']}) but the gradient step was placed on "
            f"{out['training_device']}. Training would run on CPU at roughly 1/40 of GPU "
            "speed, turning a 60 GPU-hour run into weeks. Resolve before launching.")
    if not out["physical_gpus"]:
        out["problems"].append(
            "no GPU is visible to TensorFlow. The compute estimate assumes GPU training.")
    if len(out["physical_gpus"]) > 1:
        # A WARNING, not a blocker. The idle GPU is an intentional
        # consequence of a frozen single-device protocol, and an earlier
        # version blocked launch on it - which would have blocked the
        # lab's actual multi-GPU machine for a condition that is by
        # design. It still requires explicit acknowledgement, so it
        # cannot pass unnoticed.
        out["warnings"].append(
            f"{len(out['physical_gpus'])} GPUs are visible, but the runner trains on ONE "
            "device sequentially and declares no distribution strategy. The other(s) will "
            "idle for the whole run and GPU-hours equal wall-clock hours. This is intentional: "
            "a distribution strategy would change the gradient semantics of a frozen protocol. "
            "Acknowledge with --acknowledge-idle-gpus. See TRAINING_TOPOLOGY.")
        out["requires_acknowledgement"].append("idle_gpus")
    out["topology"] = dict(TRAINING_TOPOLOGY)
    out["topology"]["devices_visible"] = len(out["physical_gpus"])
    return out


def memory_report(*, required_gb: Optional[float] = None) -> dict[str, Any]:
    """Host memory, and whether it clears a caller-supplied requirement."""
    out: dict[str, Any] = {"total_gb": None, "available_gb": None, "problems": []}
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        out["total_gb"] = round(page * os.sysconf("SC_PHYS_PAGES") / 1e9, 2)
        if hasattr(os, "sysconf_names") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            out["available_gb"] = round(page * os.sysconf("SC_AVPHYS_PAGES") / 1e9, 2)
    except (ValueError, OSError, AttributeError) as exc:
        out["problems"].append(f"host memory could not be read: {exc}")
    if required_gb is not None:
        out["required_gb"] = required_gb
        have = out["available_gb"] if out["available_gb"] is not None else out["total_gb"]
        if have is not None and have < required_gb:
            out["problems"].append(
                f"{have} GB of memory available against an estimated requirement of "
                f"{required_gb} GB. The training arrays are cast to float32 for training, "
                "which is four times their on-disk uint8 size.")
    return out


def disk_report(path, *, required_gb: Optional[float] = None) -> dict[str, Any]:
    """
    Free space where the run WILL write.

    The output directory usually does not exist yet, so the nearest
    existing ancestor is measured instead - which is the filesystem the
    run will actually land on. Reporting "unreadable" for a path that
    simply has not been created yet would make the check useless exactly
    when it is needed.
    """
    probe = Path(path).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    out: dict[str, Any] = {"path": str(path), "measured_at": str(probe), "problems": []}
    try:
        usage = shutil.disk_usage(str(probe))
        out.update(total_gb=round(usage.total / 1e9, 2),
                   used_gb=round(usage.used / 1e9, 2),
                   free_gb=round(usage.free / 1e9, 2))
    except OSError as exc:
        out["problems"].append(f"disk usage could not be read for {probe}: {exc}")
        return out
    if required_gb is not None:
        out["required_gb"] = required_gb
        if out["free_gb"] < required_gb:
            out["problems"].append(
                f"{out['free_gb']} GB free against an estimated requirement of "
                f"{required_gb} GB.")
        elif out["free_gb"] < required_gb * 2:
            out["problems"].append(
                f"{out['free_gb']} GB free against {required_gb} GB required: under 2x "
                "headroom. A full disk mid-run fails an arm and invalidates its block.")
    return out


#: Filesystems on which `os.replace` is atomic and `fsync` is durable,
#: and on which the SIGKILL recovery tests have therefore been measured
#: or are the standard the POSIX guarantees were written for. `overlay`
#: is included because it is what a container (Kaggle, Colab, Docker)
#: presents for local scratch, and its upper layer is an ordinary local
#: filesystem; `/kaggle/working` and Colab's `/content` are overlay.
DURABLE_FILESYSTEMS: frozenset[str] = frozenset({
    "ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs", "jfs", "reiserfs",
    "overlay", "overlayfs", "apfs", "hfs", "ufs",
})

#: Filesystems the run must not write to, with the reason each one
#: breaks the recovery design. The key is matched against the reported
#: type and against its prefix before the first '.', so every
#: `fuse.<something>` lands on the `fuse` entry.
REJECTED_FILESYSTEMS: dict[str, str] = {
    "fuse": ("a FUSE mount: `os.replace` may be implemented as a non-atomic "
             "copy-then-delete and a successful fsync may return before the bytes "
             "have left the machine, so neither the two-slot protocol nor the "
             "atomic commit manifest holds"),
    "fuseblk": ("a FUSE block mount (ntfs-3g and similar): rename atomicity is not "
                "guaranteed by the driver"),
    "fusectl": "a FUSE control filesystem, not a place to write run output",
    "drive": "a Google Drive mount: renames are remote API calls and are not atomic",
    "drivefs": "a Google Drive mount: renames are remote API calls and are not atomic",
    "gcsfuse": "a Google Cloud Storage FUSE mount: object stores have no atomic rename",
    "s3fs": "an S3 FUSE mount: object stores have no atomic rename",
    "rclone": "an rclone FUSE mount: remote-backed, no atomic rename",
    "sshfs": "an sshfs mount: rename atomicity depends on the remote and the link",
    "nfs": "a network filesystem: rename atomicity and fsync durability are not guaranteed",
    "nfs4": "a network filesystem: rename atomicity and fsync durability are not guaranteed",
    "cifs": "an SMB/CIFS mount: rename semantics differ from POSIX",
    "smb3": "an SMB mount: rename semantics differ from POSIX",
    "smbfs": "an SMB mount: rename semantics differ from POSIX",
    "afs": "a distributed filesystem: not verified for atomic rename",
    "9p": "a 9p mount (VM host sharing): rename and fsync are passed through a host agent",
    "virtiofs": "a virtiofs host share: durability depends on the host's cache policy",
    "davfs": "a WebDAV mount: no atomic rename",
    "ceph": "a distributed filesystem: not verified for atomic rename",
    "glusterfs": "a distributed filesystem: not verified for atomic rename",
    "lustre": "a distributed filesystem: not verified for atomic rename",
    "vboxsf": "a VirtualBox shared folder: rename and fsync are not POSIX-faithful",
    "vmhgfs": "a VMware shared folder: rename and fsync are not POSIX-faithful",
    "exfat": "no fsync-backed metadata journaling; a crash can lose a completed rename",
    "vfat": "no fsync-backed metadata journaling; a crash can lose a completed rename",
    "msdos": "no fsync-backed metadata journaling; a crash can lose a completed rename",
}

#: Memory-backed filesystems. Rename IS atomic here, but nothing
#: survives a host restart, which defeats the purpose of a recovery
#: design for a 53-84 GPU-hour run.
EPHEMERAL_FILESYSTEMS: dict[str, str] = {
    "tmpfs": ("a memory-backed filesystem: the whole run is lost on a host restart, "
              "so the committed checkpoints cannot be recovered"),
    "ramfs": "a memory-backed filesystem: nothing survives a host restart",
    "devtmpfs": "a device filesystem, not a place to write run output",
}

#: Paths that are a Google Drive mount under Colab even when the kernel
#: reports a generic type. Checked as a path prefix, case-insensitively.
DRIVE_PATH_MARKERS: tuple[str, ...] = (
    "/content/drive", "/content/gdrive", "/gdrive", "/mnt/gdrive",
    "/mydrive", "/my drive", "/googledrive", "/google drive",
)


def _unescape_mount_field(field: str) -> str:
    """mountinfo octal-escapes space, tab, newline and backslash."""
    for code, char in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        field = field.replace(code, char)
    return field


def _mounts_from_mount_command() -> list[dict[str, str]]:
    """
    BSD and macOS fallback: parse the output of `mount`.

    There is no /proc there, so without this the mount table reads empty
    and every path is classified UNKNOWN - a fail-closed refusal, but a
    WRONG one on an ordinary local APFS disk. A check that cannot
    distinguish "unverifiable" from "unsupported platform" is not doing
    its job.

    The format is:
        /dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled)
    with the filesystem type as the first parenthesised field and the
    mount options following it.
    """
    try:
        proc = subprocess.run(["mount"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    mounts: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        match = re.match(r"^(?P<dev>.+?) on (?P<point>.+?) \((?P<opts>.*)\)\s*$", line)
        if not match:
            continue
        fields = [f.strip() for f in match.group("opts").split(",") if f.strip()]
        if not fields:
            continue
        mounts.append({
            "mount_point": match.group("point"),
            "options": ",".join(fields[1:]),
            "fstype": fields[0],
            "device": match.group("dev"),
        })
    return mounts


def read_mounts(source: str = "/proc/self/mountinfo") -> list[dict[str, str]]:
    """
    The mount table, as (mount_point, fstype, device, options) records.

    `/proc/self/mountinfo` is preferred because it reports mount points
    for THIS process's namespace, which is what a container's paths
    actually resolve through. Where it does not exist - macOS, the BSDs -
    the `mount` command is parsed instead.
    """
    try:
        text = Path(source).read_text(encoding="utf-8")
    except OSError:
        return _mounts_from_mount_command()
    mounts: list[dict[str, str]] = []
    for line in text.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        lf, rf = left.split(), right.split()
        if len(lf) < 6 or len(rf) < 2:
            continue
        mounts.append({
            "mount_point": _unescape_mount_field(lf[4]),
            "options": lf[5],
            "fstype": rf[0],
            "device": _unescape_mount_field(rf[1]),
        })
    return mounts or _mounts_from_mount_command()


def mount_for(path, mounts: list[dict[str, str]]) -> Optional[dict[str, str]]:
    """
    The mount that backs `path`: the longest mount point that is a path
    prefix of it. Longest wins, because `/` is a prefix of everything
    and would otherwise mask every nested mount.
    """
    target = Path(path).resolve()
    best: Optional[dict[str, str]] = None
    best_len = -1
    for m in mounts:
        mp = Path(m["mount_point"])
        if target == mp or mp in target.parents:
            depth = len(mp.parts)
            if depth > best_len:
                best, best_len = m, depth
    return best


def classify_filesystem(fstype: str, *, resolved_path: str = "",
                        device: str = "") -> tuple[str, Optional[str]]:
    """
    (verdict, reason) for one filesystem type. Verdict is ACCEPTED,
    REJECTED or UNKNOWN.

    UNKNOWN is a refusal, not a pass. The instruction this implements is
    that crash durability is not claimed for a filesystem that has not
    been verified, and an unrecognized type has not been.
    """
    kind = (fstype or "").strip().lower()
    family = kind.split(".", 1)[0]
    haystack = f"{resolved_path} {device}".lower()

    for marker in DRIVE_PATH_MARKERS:
        if marker in haystack:
            return "REJECTED", (f"the path is a Google Drive mount ({marker!r}): renames "
                                f"are remote API calls and are not atomic, so a kill "
                                f"during a checkpoint commit can leave no valid slot")
    if kind in REJECTED_FILESYSTEMS:
        return "REJECTED", f"{kind} is {REJECTED_FILESYSTEMS[kind]}"
    if family in REJECTED_FILESYSTEMS:
        return "REJECTED", f"{kind} is {REJECTED_FILESYSTEMS[family]}"
    if kind in EPHEMERAL_FILESYSTEMS:
        return "REJECTED", f"{kind} is {EPHEMERAL_FILESYSTEMS[kind]}"
    if kind in DURABLE_FILESYSTEMS:
        return "ACCEPTED", None
    if not kind:
        return "UNKNOWN", ("the filesystem backing the output directory could not be "
                           "identified")
    return "UNKNOWN", (f"{kind} is not in the verified set "
                       f"{sorted(DURABLE_FILESYSTEMS)}; the SIGKILL recovery tests were "
                       f"not measured on it, so crash durability is not claimed for it")


def filesystem_report(path, *, mounts: Optional[list[dict[str, str]]] = None,
                      mounts_source: str = "/proc/self/mountinfo") -> dict[str, Any]:
    """
    Identify and classify the filesystem the run WILL write to.

    The output directory usually does not exist yet, so the nearest
    existing ancestor is resolved first - the same path `disk_report`
    measures, and the filesystem the run will land on.
    """
    requested = Path(path)
    resolved = requested.resolve()
    # The nearest EXISTING ancestor, for continuity with `disk_report`,
    # which has to call statvfs on something that exists.
    probe = resolved
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    # Mount matching uses the RESOLVED REQUESTED path, not the existing
    # ancestor. A mount point is a path prefix and does not care whether
    # the leaf exists; walking up first would discard the very component
    # that identifies the mount - `/content/drive/MyDrive/out` would be
    # classified as `/` and a Drive-backed output directory that has not
    # been created yet would be reported as local disk.
    table = read_mounts(mounts_source) if mounts is None else list(mounts)
    out: dict[str, Any] = {
        "path": str(requested),
        "resolved_path": str(resolved),
        "measured_at": str(probe),
        "problems": [],
        "warnings": [],
        "mount_table_entries": len(table),
    }
    if not table:
        out["verdict"] = "UNKNOWN"
        out["problems"].append(
            f"the mount table could not be read ({mounts_source}), so the filesystem "
            f"backing {resolved} is unidentified. Crash durability is not claimed for an "
            f"unverified filesystem; this refusal is deliberate.")
        return out

    mount = mount_for(resolved, table)
    if mount is None:
        out["verdict"] = "UNKNOWN"
        out["problems"].append(
            f"no mount in the table backs {resolved}; the filesystem is unidentified.")
        return out

    out.update(mount_point=mount["mount_point"], fstype=mount["fstype"],
               device=mount["device"], options=mount["options"])
    verdict, reason = classify_filesystem(
        mount["fstype"], resolved_path=str(resolved), device=mount["device"])
    out["verdict"] = verdict
    if verdict != "ACCEPTED":
        out["problems"].append(
            f"output directory {resolved} is on {mount['mount_point']} ({mount['fstype']}): "
            f"{reason}. Write the run output to local disk instead.")
    options = {o.strip() for o in mount["options"].split(",")}
    if "ro" in options or "read-only" in options:
        out["problems"].append(
            f"{mount['mount_point']} is mounted read-only; the run cannot write there.")
    return out


#: What a userspace probe can and cannot establish. Stated as data so
#: the preflight report can print it verbatim instead of paraphrasing,
#: and so a reader can see the limits without reading this module.
WHAT_THE_PROBE_ESTABLISHES = {
    "established_when_every_operation_succeeds": [
        "os.replace within a single directory is SUPPORTED on this filesystem and, "
        "immediately afterwards, the destination holds the new bytes and the source name "
        "is gone (observed, single-threaded)",
        "fsync on a regular file descriptor is SUPPORTED and returns success",
        "fsync on a DIRECTORY file descriptor is SUPPORTED and returns success - the "
        "operation the commit protocol uses to make a rename itself durable, and the one "
        "most likely to fail or be silently ignored on a non-local filesystem",
        "the filesystem type reported for this exact path is one on which these "
        "operations have POSIX semantics",
    ],
    "not_established_by_this_probe_or_by_any_test_in_this_suite": [
        "DURABILITY ACROSS POWER LOSS OR KERNEL PANIC. fsync returning success is a "
        "promise made by the filesystem, its driver, the block layer and the drive's own "
        "cache. None of that is auditable from userspace. A process-kill test cannot "
        "establish it either: SIGKILL leaves the page cache intact, so data that was "
        "never flushed to the device still reads back correctly afterwards.",
        "ATOMICITY UNDER CONCURRENT ACCESS. The probe is single-threaded. It observes "
        "that a rename completed, not that no reader could ever see a torn intermediate "
        "state.",
        "BEHAVIOUR WHEN THE DEVICE IS FULL. ENOSPC during a checkpoint write is not "
        "exercised.",
        "WRITE BARRIERS AND DRIVE CACHE POLICY. Whether the drive honours a flush "
        "request rather than acknowledging it immediately cannot be determined from here.",
    ],
    "consequence": (
        "The recovery design is therefore verified against ORDERED, ATOMIC-RENAME "
        "semantics on a supported filesystem, and against process death at every "
        "boundary. It is NOT verified against sudden power loss, and no claim to that "
        "effect should appear in the certificate."
    ),
}


def durability_probe(path) -> dict[str, Any]:
    """
    Actually perform the operations the commit protocol depends on, in
    the filesystem that will hold the run.

    Classifying a filesystem by the type string in the mount table says
    what SHOULD be true. This performs the operations and reports what
    IS true: a same-directory `os.replace`, an `fsync` on a file, and an
    `fsync` on a directory descriptor. The last is the one that quietly
    fails or is ignored outside local filesystems, and it is the one the
    atomic commit manifest relies on.

    The probe writes only inside a temporary subdirectory it creates
    itself, and removes only the entries it created. It never writes to,
    renames or deletes anything that was already present, so it is safe
    to run against a directory that holds evidence.
    """
    requested = Path(path)
    probe_at = requested.resolve()
    while not probe_at.exists() and probe_at != probe_at.parent:
        probe_at = probe_at.parent

    out: dict[str, Any] = {
        "requested_path": str(requested),
        "probed_at": str(probe_at),
        "operations": {},
        "problems": [],
        "established": [],
        "unverified": [],
        "cannot_be_proven": list(
            WHAT_THE_PROBE_ESTABLISHES[
                "not_established_by_this_probe_or_by_any_test_in_this_suite"]),
    }

    def record(name: str, ok: bool, detail: str) -> None:
        out["operations"][name] = {"supported": ok, "detail": detail}
        if not ok:
            out["problems"].append(f"{name}: {detail}")

    if not probe_at.is_dir():
        out["probe_verdict"] = "NOT RUN"
        out["problems"].append(
            f"{probe_at} is not an existing directory, so the operations the commit "
            f"protocol depends on could not be exercised.")
        return out

    workdir: Optional[Path] = None
    created: list[Path] = []
    try:
        workdir = Path(tempfile.mkdtemp(prefix=".durability-probe-", dir=str(probe_at)))
        src = workdir / "probe.inflight"
        dst = workdir / "probe.committed"

        # 1. write + fsync a regular file
        try:
            with src.open("wb") as fh:
                fh.write(b"durability probe payload")
                fh.flush()
                os.fsync(fh.fileno())
            created.append(src)
            record("file_write_and_fsync", True, "fsync on a file descriptor succeeded")
        except OSError as exc:
            record("file_write_and_fsync", False, f"fsync on a file failed: {exc}")
            raise

        # 2. same-directory atomic replace, onto a name that already exists,
        #    because replacing an EXISTING file is what the commit protocol does
        try:
            dst.write_bytes(b"previous committed payload")
            created.append(dst)
            os.replace(src, dst)
            if src.exists():
                record("same_directory_replace", False,
                       "os.replace left the source name in place")
            elif dst.read_bytes() != b"durability probe payload":
                record("same_directory_replace", False,
                       "os.replace did not leave the new bytes at the destination")
            else:
                created.remove(src)
                record("same_directory_replace", True,
                       "os.replace over an existing file left the new bytes and removed "
                       "the source name")
        except OSError as exc:
            record("same_directory_replace", False, f"os.replace failed: {exc}")
            raise

        # 3. fsync the DIRECTORY - the operation that makes the rename durable
        try:
            fd = os.open(str(workdir), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            record("directory_fsync", True,
                   "fsync on a directory descriptor succeeded")
        except OSError as exc:
            record("directory_fsync", False,
                   f"fsync on a directory descriptor failed: {exc}. The commit protocol "
                   f"cannot make a rename durable on this filesystem.")

        # 4. read back after the sync
        try:
            record("read_back_after_sync", dst.read_bytes() == b"durability probe payload",
                   "the renamed file read back with the expected content")
        except OSError as exc:
            record("read_back_after_sync", False, f"read-back failed: {exc}")

    except OSError as exc:
        out["problems"].append(f"the probe could not complete: {exc}")
    finally:
        # Remove ONLY what this probe created.
        for leftover in created:
            try:
                leftover.unlink()
            except OSError:
                pass
        if workdir is not None:
            for stray in list(workdir.glob("*")):
                try:
                    stray.unlink()
                except OSError:
                    pass
            try:
                workdir.rmdir()
            except OSError:
                out["problems"].append(
                    f"the probe's temporary directory {workdir} could not be removed; "
                    f"remove it by hand before launching.")

    required = ("file_write_and_fsync", "same_directory_replace", "directory_fsync",
                "read_back_after_sync")
    missing = [op for op in required if not out["operations"].get(op, {}).get("supported")]
    if missing:
        out["probe_verdict"] = "FAILED"
        out["unverified"] = list(
            WHAT_THE_PROBE_ESTABLISHES["established_when_every_operation_succeeds"])
        out["problems"].append(
            f"the filesystem at {probe_at} did not support {missing}. The two-slot "
            f"checkpoint protocol and the atomic commit manifest cannot be relied on "
            f"there, so crash recovery is not claimed and execution is refused.")
    else:
        out["probe_verdict"] = "PASSED"
        out["established"] = list(
            WHAT_THE_PROBE_ESTABLISHES["established_when_every_operation_succeeds"])
    return out


def _disk_budget(items: Optional[list[tuple[str, float]]],
                 free_gb: Optional[float]) -> dict[str, Any]:
    """
    An ITEMISED disk requirement rather than one number.

    A single "needs ~5 GB" cannot be checked by a reader: it does not
    say whether the dataset, the rotating slots, the best and terminal
    checkpoints, the temporary inflight files and the logs were all
    counted, nor how much of it is margin. Each line is named and the
    margin is shown as its own line, so a disagreement can be located
    instead of guessed at.
    """
    if not items:
        return {}
    subtotal = round(sum(size for _, size in items), 3)
    out: dict[str, Any] = {
        "items": [{"component": name, "gb": size} for name, size in items],
        "subtotal_gb": subtotal,
    }
    if free_gb is not None:
        out["free_gb"] = free_gb
        out["headroom_multiple"] = (round(free_gb / subtotal, 1) if subtotal else None)
    return out


def preflight(*, output_path, disk_required_gb: Optional[float] = None,
              memory_required_gb: Optional[float] = None,
              extra: Optional[dict] = None,
              acknowledged: tuple[str, ...] = (),
              mounts: Optional[list[dict[str, str]]] = None,
              probe: bool = True,
              disk_items: Optional[list[tuple[str, float]]] = None,
              estimate_notes: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    All checks, with BLOCKERS separated from disclosed WARNINGS.

    A blocker is a condition that would make the run wrong or impossible:
    training on CPU when the estimate assumes GPU, no disk, no memory. A
    warning is a condition that is intentional but must not pass
    unnoticed - an idle second GPU being the case that matters here. A
    warning blocks only until it is explicitly acknowledged, so the
    operator states that they know, rather than the tool pretending the
    condition does not exist or refusing a correct configuration.
    """
    gpu = gpu_report()
    mem = memory_report(required_gb=memory_required_gb)
    disk = disk_report(output_path, required_gb=disk_required_gb)
    fs = filesystem_report(output_path, mounts=mounts)
    # Only probe if the type check did not already refuse: there is no
    # point writing to a filesystem we have decided not to use, and on a
    # remote mount the probe itself can be slow.
    if fs["verdict"] == "ACCEPTED" and probe:
        dur = durability_probe(output_path)
    else:
        dur = {"probe_verdict": "NOT RUN",
               "requested_path": str(output_path),
               "operations": {}, "problems": [], "established": [],
               "unverified": list(WHAT_THE_PROBE_ESTABLISHES[
                   "established_when_every_operation_succeeds"]),
               "cannot_be_proven": list(WHAT_THE_PROBE_ESTABLISHES[
                   "not_established_by_this_probe_or_by_any_test_in_this_suite"]),
               "skipped_because": ("the filesystem type check already refused this path"
                                   if fs["verdict"] != "ACCEPTED"
                                   else "probing was disabled by the caller")}
    problems = [*gpu["problems"], *mem["problems"], *disk["problems"], *fs["problems"],
                *dur["problems"]]
    warnings = [*gpu.get("warnings", []), *mem.get("warnings", []),
                *disk.get("warnings", []), *fs.get("warnings", [])]
    needed = list(gpu.get("requires_acknowledgement", []))
    unacknowledged = [n for n in needed if n not in acknowledged]
    return {
        "artifact": PREFLIGHT_VERSION,
        "output_path": str(output_path),
        "gpu": gpu, "memory": mem, "disk": disk, "filesystem": fs, "durability": dur,
        "disk_budget": _disk_budget(disk_items, disk.get("free_gb")),
        "estimate_notes": dict(estimate_notes or {}),
        "verified": list(dur.get("established", [])),
        "unverified": list(dur.get("unverified", [])),
        "cannot_be_proven": list(dur.get("cannot_be_proven", [])),
        "extra": dict(extra or {}),
        "problems": problems,
        "warnings": warnings,
        "requires_acknowledgement": needed,
        "acknowledged": list(acknowledged),
        "unacknowledged": unacknowledged,
        "clear_to_launch": not problems and not unacknowledged,
    }


def format_report(report: dict[str, Any]) -> str:
    g, m, d = report["gpu"], report["memory"], report["disk"]
    fs = report.get("filesystem") or {}
    dur = report.get("durability") or {}
    lines = [
        "PREFLIGHT",
        f"  tensorflow / keras   {g['tensorflow']} / {g['keras']}",
        f"  built with CUDA      {g['cuda_built']}",
        f"  GPUs visible         {len(g['physical_gpus'])} {g['physical_gpus']}",
        f"  gradient step ran on {g['training_device']}",
        f"  training on GPU      {g['training_is_on_gpu']}",
        f"  devices per fit      {TRAINING_TOPOLOGY['devices_used_per_fit']} "
        f"(strategy: {TRAINING_TOPOLOGY['distribution_strategy']})",
        f"  arms / blocks        {TRAINING_TOPOLOGY['arms']}",
        f"  memory               {m.get('available_gb')} GB available of "
        f"{m.get('total_gb')} GB"
        + (f"; need ~{m['required_gb']} GB" if "required_gb" in m else ""),
        f"  disk at {d.get('measured_at', d['path'])}",
        f"                       {d.get('free_gb')} GB free of {d.get('total_gb')} GB"
        + (f"; need ~{d['required_gb']} GB" if "required_gb" in d else ""),
        f"  filesystem           {fs.get('fstype', 'UNIDENTIFIED')} on "
        f"{fs.get('mount_point', '?')} -> {fs.get('verdict', 'UNKNOWN')}",
        f"  durability probe     {dur.get('probe_verdict', 'NOT RUN')} at "
        f"{dur.get('probed_at', dur.get('requested_path', '?'))}",
    ]
    for name, op in (dur.get("operations") or {}).items():
        lines.append(f"    {'verified  ' if op['supported'] else 'FAILED    '} {name}")
    budget = report.get("disk_budget") or {}
    if budget:
        lines.append("  disk budget (itemised)")
        for item in budget["items"]:
            lines.append(f"    {item['gb']:>8.3f} GB  {item['component']}")
        lines.append(f"    {budget['subtotal_gb']:>8.3f} GB  SUBTOTAL")
        if budget.get("headroom_multiple") is not None:
            lines.append(f"    {budget['free_gb']:>8.2f} GB  free "
                         f"({budget['headroom_multiple']}x the subtotal)")
    for key, value in (report.get("extra") or {}).items():
        lines.append(f"  {key:20} {value}")
    notes = report.get("estimate_notes") or {}
    if notes:
        lines.append("\n  COMPUTE ESTIMATE (an estimate, not a guarantee)")
        for key, value in notes.items():
            lines.append(f"    {key}: {value}")
    if report["problems"]:
        lines.append("\n  BLOCKERS")
        lines.extend(f"    - {p}" for p in report["problems"])
    if report.get("warnings"):
        lines.append("\n  WARNINGS (disclosed, not blocking)")
        lines.extend(f"    - {w}" for w in report["warnings"])
    if report.get("unacknowledged"):
        lines.append(f"\n  AWAITING ACKNOWLEDGEMENT: {report['unacknowledged']}")
        lines.append("    pass --acknowledge-idle-gpus to proceed")

    # Exactly what has been verified, what has not, and what no test in
    # this suite can establish. Printed on EVERY report, pass or fail,
    # because the limits of the evidence do not depend on the outcome.
    if report.get("verified"):
        lines.append("\n  VERIFIED ON THIS FILESYSTEM, BY MEASUREMENT")
        lines.extend(f"    + {v}" for v in report["verified"])
    if report.get("unverified"):
        lines.append("\n  NOT VERIFIED HERE (the probe did not run or did not pass)")
        lines.extend(f"    ? {v}" for v in report["unverified"])
    if report.get("cannot_be_proven"):
        lines.append("\n  CANNOT BE PROVEN BY ANY TEST IN THIS SUITE")
        lines.extend(f"    - {v}" for v in report["cannot_be_proven"])
        lines.append(f"    => {WHAT_THE_PROBE_ESTABLISHES['consequence']}")

    permitted = not report["problems"] and not report.get("unacknowledged")
    lines.append(f"\n  EXECUTION PERMITTED: {'YES' if permitted else 'NO'}")
    if permitted:
        lines.append("  clear to launch")
    return "\n".join(lines)