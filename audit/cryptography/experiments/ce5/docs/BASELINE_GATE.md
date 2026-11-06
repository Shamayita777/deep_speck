# G-CE5 baseline gate (r3)

## What is reproduced

The gate reproduces P1 Table 7, row 5 (p. 19): accuracy **92.98%**, TPR **90.76%**, TNR **95.22%**. These come from the average-key-rank distinguisher with mask 0xff8f/ff8f.

The plain DDT (D5: 0.911 / 0.877 / 0.947, P1 Table 1 p. 9) is used only as a separation reference. Gohr's N5 and M-ODT are not gate inputs (SOURCE_VERIFICATION.md §5).

## Why the r2 interval was replaced

The r2 interval was [0.9204, 0.9392], the midpoint to D5 on either side. It shows **separation** from the plain DDT. It does not show **reproduction** of 92.98%: an implementation at 92.1%, about 0.9 points low, would have passed.

r3 therefore has two components and requires both of the first two.

| component | condition | role |
|---|---|---|
| reproduction, accuracy | Clopper–Pearson 95% interval inside [0.9298 − τ, 0.9298 + τ] | required |
| reproduction, rates | CP intervals of TPR and TNR inside 0.9076 ± τ_r and 0.9522 ± τ_r | required |
| separation | CP lower limit of accuracy > (0.9298 + 0.911)/2 = 0.9204 | reported (adequacy) |

**Decisions.**
- **PASS** iff reproduction holds. This is the only outcome under which the baseline may be called Benamira's aKR distinguisher, and the only one under which CE5 proceeds.
- **Separation without reproduction** is reported as an adequacy result, and CE5 is **UNADJUDICATED**.
- **No substitute baseline** is run under this design.

## Tolerance

P1 does not state the evaluation size for Table 7, so the published value's own sampling error is unknown. (Table 8 uses 10^5 pairs; at that size the SE of an accuracy near 0.93 is about 0.0008.)

The other source of variation is the table itself. A 10^7-pair aDDT is an estimate, and a fresh table gives a slightly different distinguisher.

**Rule, fixed before approval.**
- τ = max(0.005, 3 σ_t) and τ_r = max(0.010, 3 σ_t,rate).
- σ_t comes from the non-confirmatory `variability` stage: 5 independent tables, each built from fresh pairs and scored on its own fresh 10^5 + 10^5 pairs. No network, trail proxy or replicate is involved.
- **The floors are proposals, not published or approved values.** They are about 6 SEs of a 10^5-pair evaluation for accuracy and about 7 SEs of a 5×10^4-per-class evaluation for the rates. They were confirmed unchanged on 2026-11-05 (design `tolerance_decision`).
- The `variability` stage has been run twice (non-confirmatory; no network output, so it cannot inform the CE5 hypothesis). The second run stores the evaluation ciphertexts so that its scores can be re-scored independently.

| run | accuracy (5 tables) | TPR | TNR | SD accuracy / TPR / TNR | 3 SD accuracy / rate | verified |
|---|---|---|---|---|---|---|
| `validation/r3_final_20261105` | 0.9284–0.9295 | 0.9072–0.9088 | 0.9491–0.9507 | 0.00040 / 0.00079 / 0.00070 | 0.0012 / 0.0024 | hashes, sums, confusion matrices, SDs and tolerances recompute; ciphertexts were not stored, so scores cannot be re-scored |
| `validation/r3_final_20261105_variability_v2` | 0.9284–0.9294 | 0.9060–0.9082 | 0.9495–0.9513 | 0.00049 / 0.00087 / 0.00069 | 0.0015 / 0.0026 | VERIFIED, including 80 scalar re-scorings |

**What these runs imply for the tolerance.**
- Under the registered formula, the floors dominate in both runs: τ = 0.005 and τ_r = 0.010.
- Observed against Table 7:
  - mean accuracy is about 0.929 (published 0.9298);
  - TPR is about 0.907 (published 0.9076);
  - TNR is about 0.950 (published 0.9522), about 0.2 points low.
- With the proposed floors, all three would lie well inside their acceptance intervals. With a bare 3 SD rule, without floors, TNR would be at the margin.
- These values were observed before the tolerance decision. The formula and the floors were written into the design before either run. Confirming them as written keeps that decision free of outcome information; tightening them now would be informed by these results, and must be recorded as such.
- `gate.tolerance` remains unset.

At 10^5 pairs per class the CP half-widths are about 0.0011 (accuracy) and 0.0018 (TPR), so τ = 0.005 is not consumed by sampling alone.

## Data, independence, failure

- **Gate data:** 100,000 real and 100,000 random pairs, generated inside the production command after approval and the timestamp check. They are independent of the tables and of every replicate.
- **Frozen inputs:** the tables are fixed by their registered hashes, and there is no smoothing.
- **Recorded:** the confusion matrix, rates, CP intervals and both components. The verifier recomputes all of them, including the strict decisions, from the stored scores, and spot-checks the scores with the literal scalar Algorithm 2.
- **On FAIL:** UNADJUDICATED, no replicate is run, and the result is reported. A test confirms that no raw replicate directory is created.
- **Crash recovery.** The production manifest is written before the gate.
  - If `gate_raw.npz` exists without a recorded gate, it is adopted only if it has the registered size, and every stored score re-derives exactly from its stored ciphertexts. The adopted gate is marked `recovered`.
  - Otherwise production refuses, keeps the file and requires a new output directory. Nothing is overwritten.
