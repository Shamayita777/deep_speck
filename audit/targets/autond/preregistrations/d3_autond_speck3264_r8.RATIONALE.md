# D3 AutoND preregistration — rationale for the draft

Draft: `d3_autond_speck3264_r8.json` (schema `d3-preregistration-3`). **Status: UNAPPROVED DRAFT.**
Nothing in the draft is an approval. The approver has answered the Gohr-results attestation in `data_examined_before_approval` (item 5): they examined those results before approval. The approval fields are still blank.

## 1. What this audit is

A **bound-reporting audit with a FAIL screen** on the existing canonical instance `29cd55c91cb27686…`. It is not an equivalence certification.

- **No PASS or equivalence claim is made.** The thresholds are operational FAIL thresholds, not validated equivalence margins. At these sample sizes their design resolution makes PASS unreachable for every confirmatory family.
- **Guarantees.** With probability ≥ 1 − 0.005, every reported upper and lower bound of every confirmatory hypothesis holds simultaneously (G1). With probability ≥ 1 − 0.005, no confirmatory exact detection is false (G2). Jointly, P(any G1 or G2 error) ≤ 0.01 by the union bound.
- **FAIL** means some confirmatory hypothesis has a lower bound above its threshold: a confirmed deviation.
- **INCONCLUSIVE** is the expected outcome when nothing FAILs. It certifies nothing at the thresholds; the information is in the reported simultaneous bounds.
- **Every guarantee is conditional on i.i.d. rows within each partition and independent partitions** (§5).
- **Scope.** Each hypothesis concerns one bit or one bit pair. Nothing is claimed about classifiers that use more than two bits, the 64-bit model, or the full joint law.

## 2. Design

| Item | Value | Basis / status |
|---|---|---|
| Global α | 0.01, **joint** over G1 and G2 | Approver decision (2026-10-22). Enforced by one engine rule, `joint_alpha_problem`: 2 × Σ(allocation) ≤ global α. The preregistration validator, the calibration verifier, the decision gate and the runner all apply it. |
| Confirmatory families | R2, A, B, C, D; 0.001 each (Σ = 0.005) | Equal split; approver decision. No evidence-based weighting exists. |
| R1 | Exploratory, non-gating, α 0.01 within R1 | Its sufficient statistic, the label counts, was examined in the D0 manifest before approval. Disclosure alone would not make R1 preregistered. |
| E | Exploratory, non-gating, α 0.01 within E | Unchanged. It characterises cross-class pair structure (the distinguisher signal). |
| Thresholds | `class_ratio` m, `class_conditional_bit` 2m, `pairwise_tvd` m, `cross_class_bit` 2m, with m = 0.00144 | Operational FAIL thresholds from the frozen `autond_ii_v2` margin (commit `fc46c57`, 2026-10-16, before this instance existed). They were approved for checkpoint-accuracy replication and are **not validated for D3**. |

Exploratory α is set by the runner (`exploratory_alpha` = global α within each exploratory family). The certificate records it, together with the runner's hash. The preregistration schema has no field for it.

### Design resolution

Minimum certifiable effect: the upper bound obtained when the observed table equals its expectation. It uses exact Clopper–Pearson bounds at the per-hypothesis α. Realised bounds are somewhat wider (calibration S4).

| Family | Per-hypothesis α | Train | Validation / test | Threshold |
|---|---|---|---|---|
| R1 (exploratory) | 0.01 / 3 | 0.00046 | 0.00147 | 0.00144 |
| R2 | 0.001 / 384 | 0.00105 | 0.00332 | 0.00288 |
| D | 0.001 / 192 | 0.00210 | 0.00665 | 0.00288 |
| A | 0.001 / 6048 | 0.0064 (train vs held-out) | 0.0097 (validation vs test) | 0.00144 |
| B / C | 0.001 / 6048 | 0.0091 (train vs held-out) | 0.0137 (validation vs test) | 0.00144 |

The calibration recomputes these values in S5, and `test_d3_autond_preregistration.py` checks them against it. Each confirmatory family includes validation/test hypotheses whose resolution exceeds the threshold, so no confirmatory family can PASS. A, B and C exceed it even with unlimited validation/test data, because the train-sample term alone is above m.

### What each threshold means

- **`class_ratio` (R1, non-gating).** A positive ratio of ½ + κ lets a constant classifier gain κ accuracy.
- **`class_conditional_bit` (R2).** This checks conformity to the generator's exact reference ½. **It does not bound a label shortcut on its own:** if the two classes deviate in opposite directions by 2m each, a single-bit classifier gains up to 2m. Only D bounds the cross-class difference.
- **`cross_class_bit` (D).** With balanced classes, "predict 1 iff bit j = 1" has accuracy ½ + (p₁ − p₀)/2. So 2m is the cross-class difference worth an advantage of m to that single-bit classifier.
- **`pairwise_tvd` (A, B, C).** For a classifier that uses only one bit pair, |E_P f − E_Q f| ≤ TVD. So m is a split-consistency shift of m for a pair-restricted classifier. For pooled A the reading is weaker, because class-swapping shifts can cancel.
- **None of these bounds the 64-bit model.** Many sub-threshold coordinate or pair shifts can combine in a full-width classifier.

## 3. Comparison with the Gohr D3 procedure

- **The same as Gohr:**
  - scope (class balance, bit marginals, all 2016 pair laws across partitions);
  - TVD as the pair effect size;
  - Bonferroni correction;
  - the D1 and D2 scope exclusions.
- **Different, and why:**
  - **Bound to the D0 instance.** Gohr regenerated its data in-run and has no D0 instance.
  - **Class-stratified families.** Pooled checks cannot see opposite-direction class biases.
  - **Exact tests and simultaneous bounds** instead of the asymptotic chi-square.
  - **PASS requires precision.** Gohr's legacy PASS means only non-detection, as the legacy code's own docstring states.
  - **One joint α across families.** Gohr's per-partition and per-comparison α allowed up to 6× the stated α.
  - **Thresholds traced to a frozen margin.** Gohr's 0.005 / 0.01 / 0.01 carry no recorded derivation.

Gohr's completed status validates none of the AutoND choices. Gohr's code and evidence are unchanged.

## 4. Data examined before approval

`data_examined_before_approval` lists it in full:
- the D0 manifest, including the label counts (hence R1's exposure);
- D1 PASS, and D2 INCONCLUSIVE with its descriptive row-pair values;
- synthetic calibration evidence only;
- two blinded D3 computations. These computed the complete audit in memory under an earlier engine version and recorded only seals, with no statistic displayed;
- the legacy Gohr D3 results (the consistency audit with outcome PASS and its statistics, and the DESCRIPTIVE_ONLY report), on Gohr's five-round dataset under the legacy method. The drafting assistant and the approver both examined them. They validate no AutoND threshold or design choice.

This plan was fixed after D0, D1 and D2 had been run on this instance. It is not a preregistration made before the data existed.

## 5. D2 and row independence

**D2 remains INCONCLUSIVE** for A, B and C, with no failures. Its only warnings are the radius-8 near-duplicate components.

- **Monte Carlo resolution explains the warning.** D2 estimated the radius-8 null pair probability as 0.0 from 100,000 simulated pairs, so its excess ratio was reported as UNBOUNDED. The exact value for two independent uniform 64-bit rows is P(Hamming ≤ 8) = 2.7813 × 10⁻¹⁰.
- **Stratum C (Y = 0 rows, replaced by independent uniform bits by construction):**
  - Expected neighbour hits among 20,480 queried rows *at full retrieval*: 28.45 (train), 2.85 (validation), 2.85 (test).
  - D2's retrieval is non-exhaustive (8 tables × 16 sampled bit positions), so these are upper bounds on what D2 could observe.
  - **Approximate, assumption-dependent recall adjustment.** This is the drafting assistant's arithmetic, not a D2 result and not a test:
    - The tables are assumed to sample positions independently, and the 10,000-candidate bucket cap is assumed not to bind (about 76 rows per bucket at these sizes).
    - Under those assumptions, mass-weighted recall within radius 8 is ≈ 0.53.
    - Expected counts are then ≈ 15.1 / 1.5 / 1.5, against observed counts of 14 / 2 / 1.
  - The counts are therefore **consistent with** the independent-uniform null. This is **not a formal test of independence**: query hits share a reference set and are dependent, and the recall figure is approximate.
- **A and B** (which involve Y = 1 rows) have no exact null.
- **This explains the warning; it does not show that rows are independent.** D2 records `not_proof_of_independence: true`. D2.5 (sensitivity calibration) was not run, because no authorised AutoND protocol exists.
- **An unconditional independence claim would require** all of the following, and even then it would cover only the mechanisms tested plus the generator argument (rows i.i.d. if `os.urandom` is ideal, `main.py:15-25`):
  - an exact or adequately resolved radius-8 null;
  - calibrated or exhaustive retrieval recall;
  - an authorised D2.5 sensitivity calibration.
- **Consequence for D3.** Every D3 statement is conditional on i.i.d. rows. Dependent rows would make the Clopper–Pearson bounds too narrow and the exact tests anti-conservative.

## 6. What requires the approver (not done, and must not be done by anyone else)

1. Done: the Gohr-results attestation in `data_examined_before_approval` (item 5) is answered. No `PENDING APPROVER ANSWER` marker remains.
2. Review §1–5 and the JSON.
3. In the JSON, set:
   - `"approved": true`;
   - `"approved_by"`: your name;
   - `"approved_utc"`: an ISO-8601 time with timezone, before the commit;
   - `"approval_statement"`: exactly the engine's `APPROVAL_STATEMENT`:
     > I approve these D3 tolerances as operational FAIL thresholds, not validated equivalence margins, and the joint global alpha, the family alpha allocation and the family roles, for this audit. I fixed them before examining any D3 statistic of a confirmatory family on the audited data, and data_examined_before_approval records what was examined before approval.
4. Commit the file, together with the code under audit (engine, runner, calibration tool and tests), so that the certificate's git HEAD contains it.
5. Record the file's SHA-256 outside the file. This is the anchor passed as `--preregistration-sha256`.
6. Then run:
   ```
   python -m audit.dataset.experiments.d3_autond_distribution --mode confirmatory \
     --instance-dir audit/dataset/evidence/d0_autond_speck3264_r8_production \
     --out-dir audit/dataset/evidence/d3_autond_speck3264_r8_confirmatory_<id> \
     --preregistration audit/targets/autond/preregistrations/d3_autond_speck3264_r8.json \
     --preregistration-sha256 <anchor> \
     --calibration-dir audit/dataset/evidence/d3_method_validity_20261022_c39726e5
   ```
