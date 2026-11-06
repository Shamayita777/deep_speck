"""
One GPU worker of the II-4-GOHR-IMPL-V1-PAR1 scheduler.

Started by scheduler.execute_parallel with CUDA_VISIBLE_DEVICES already set, so
the process sees exactly one device. It walks the pending block list, claims each
unfinished block with an exclusive-create file, and trains that block's two arms
with the frozen runner methods. It never writes state.json or firewall.json.

Exit codes: 0 every claimed block finished training or was recorded as a frozen
technical failure; 2 an injected interruption (tests only); 3 an identity or
integrity refusal.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

from audit.common.outcomes import ExecutionMode
from audit.common.provenance import utc_timestamp
from audit.implementation.arm_recovery import InjectedInterruption
from audit.implementation.ii4_comparison_runner import ComparisonExecutionError
from audit.implementation.ii4_parallel.scheduler import (
    TRAINING_COMPLETE,
    BlockTrainingRunner,
    claim_block,
)


def load_factory(spec: str):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def log(event: dict) -> None:
    print(json.dumps({"utc": utc_timestamp(), **event}, sort_keys=True), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--factory", required=True, help="module:function returning (plan, adapter)")
    ap.add_argument("--factory-kwargs", default="{}")
    ap.add_argument("--session", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--interrupt-after-epoch", type=int, default=None)
    ap.add_argument("--stop-after-blocks", type=int, default=None,
                    help="tests only: exit 0 after this many trained blocks, leaving the rest")
    a = ap.parse_args(argv)

    plan, adapter = load_factory(a.factory)(root=str(a.root), **json.loads(a.factory_kwargs))
    if Path(plan.output_dir) != a.root:
        log({"event": "refused", "reason": f"factory plan writes to {plan.output_dir}, not {a.root}"})
        return 3
    production = plan.execution_mode is ExecutionMode.PRODUCTION
    if production and (a.interrupt_after_epoch is not None or a.stop_after_blocks is not None):
        log({"event": "refused", "reason": "fault injection is refused in production"})
        return 3
    log({"event": "worker_start", "gpu": a.gpu, "session": a.session,
         "devices_seen": adapter.environment_details().get("gpus")})

    trained = 0
    for bid in [b for b in a.blocks.split(",") if b]:
        if (a.root / "blocks" / bid / TRAINING_COMPLETE).exists():
            continue
        if not claim_block(a.root, bid, a.session, a.gpu):
            continue
        log({"event": "claimed", "block": bid, "gpu": a.gpu})
        runner = BlockTrainingRunner(plan=plan, adapter=adapter, session_id=a.session,
                                     interrupt_after_epoch=a.interrupt_after_epoch,
                                     block_id=bid, gpu_label=a.gpu)
        try:
            summary = runner.train_block(int(bid[5:]))
        except InjectedInterruption as exc:
            log({"event": "injected_interruption", "block": bid, "detail": str(exc)})
            return 2
        except ComparisonExecutionError as exc:
            log({"event": "refused", "block": bid, "reason": str(exc)[:2000]})
            return 3
        except Exception as exc:                                # noqa: BLE001
            log({"event": "error", "block": bid, "type": type(exc).__name__,
                 "detail": traceback.format_exc()[-3000:]})
            return 3
        log({"event": "block_done", **summary})
        trained += 1
        if a.stop_after_blocks is not None and trained >= a.stop_after_blocks:
            log({"event": "stopped_by_test_flag", "after_blocks": trained})
            return 0
    log({"event": "worker_finished", "gpu": a.gpu, "blocks_trained_this_session": trained})
    return 0


if __name__ == "__main__":
    sys.exit(main())
