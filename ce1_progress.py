# READ-ONLY CE1 progress check. Imports no TensorFlow, touches no GPU, writes nothing.
# usage (from the repo root):  python ce1_progress.py audit/cryptography/evidence_current/ce1/production_20260930
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path

run = Path(sys.argv[1])
lines = (run / "ledger.jsonl").read_text().splitlines()
ev = []
for l in lines:
    try: ev.append(json.loads(l))
    except json.JSONDecodeError: pass          # a half-written last line while training runs
now = datetime.now(timezone.utc)
ts = lambda e: datetime.fromisoformat(e["timestamp"])
print(f"ledger: {len(ev)} events; last event {(now - ts(ev[-1])).total_seconds():.0f}s ago "
      f"({ev[-1]['event']} {ev[-1].get('block_id')}/{ev[-1].get('arm')})")
arms = {}
for e in ev:
    if e.get("arm"): arms.setdefault((e["block_id"], e["arm"]), []).append(e)
for (b, a), es in sorted(arms.items()):
    ck = [e for e in es if e["event"] == "CHECKPOINT"]
    last = es[-1]
    if ck:
        first, lastc = ck[0], ck[-1]
        n = lastc["epoch"] - first["epoch"]
        spe = (ts(lastc) - ts(first)).total_seconds() / n if n else float("nan")
        rem = (200 - lastc["epoch"]) * spe / 3600
        print(f"{b}/{a:9} epoch {lastc['epoch']:3d}/200  {spe:6.1f}s/epoch  "
              f"~{rem:4.1f}h left for this arm  (last checkpoint {(now-ts(lastc)).total_seconds():.0f}s ago)")
    else:
        print(f"{b}/{a:9} last event: {last['event']}")
for e in ev:
    if e["event"] in ("FAILURE", "PAUSED", "REFUSED", "INTERRUPTED", "BLOCK_NOT_COUNTED"):
        print("!!", e["event"], e.get("block_id"), e.get("arm"), e.get("reason") or e.get("code"), (e.get("detail") or "")[:160])
