# SDB-V2 design — clean draft for review

**Status.** Unapproved design draft. No V2 code exists. Nothing is registered, executed, committed or pushed.

**This document supersedes, for review purposes only:**
- `SDB_V2_DESIGN.md`, revision 1 (`8d08d843…`);
- revision 0 (`e088b47a…`, copy at `~/ce5_runs/SDB_V2_DESIGN.rev0_e088b47a.md`).

Both are kept unchanged as historical drafts. Nothing from them is authoritative unless it is restated here.

**V1 stays the registered conformance pilot and is not modified:**

| item | file | SHA-256 |
|---|---|---|
| registration | `registration_sdb_v1.json` | `d9df4816…` |
| result | `results/sdb_v1_20261106T104733Z.json` | `e8fc3e19…` |
| benchmark source | `seeded_defects.py` | `c4d5ffd8…` |
| engine source | `claim_engine.py` | `821c8e29…` |

**Companion file:** `SDB_V2_H2_ALLOWLIST_DRAFT.json` (`f7c2bde8…`). It lists the H2 archive members and their hashes (§5.2).

**Manuscript of reference:** `~/Downloads/IEEE_SP_Claim_Centered_Audit_v2.tex`, SHA-256 `190209a7…`. Line numbers below refer to this file.

---

## 1. Scope and fixed decisions

1. **No new training.** No CPU or GPU model training of any kind. Only existing checkpoints, stored predictions and historical evidence are used. Where a required model or prediction artifact does not exist, the result is NOT_ASSESSED (or NOT_APPLICABLE, as defined in §4.3). No model or training evidence is manufactured to raise coverage.
2. **Engine outcomes.**
   - **Dimension outcomes:** exactly PASS, FAIL, INCONCLUSIVE and N/A (l.927–933).
   - **Absent evidence:** treated as the manuscript says: "a missing prerequisite gives INCONCLUSIVE" (l.856; also l.932). Absence is recorded only as a *reason code* attached to INCONCLUSIVE, never as a verdict or outcome.
   - **Claim outcomes:** exactly SUPPORTED, PARTIALLY SUPPORTED, NOT SUPPORTED and INCONCLUSIVE (l.1089–1091).
   - **No other category exists in V2.** The historical v1 label for G-II4 is excluded from every rule, registry, engine state, oracle, test and enumeration. It is not converted into a manuscript rule.
3. **Rule gaps are not filled.** The manuscript has no written rule for:
   - resolving conflicting evidence (stage 10, l.851, only "keeps unresolved discrepancies on record");
   - how a refused (invalid) evidence record enters a claim decision;
   - how a D4 progression failure under a SUPPORTED engine outcome maps to DI.

   No such rule is created. The affected cases are scored at detector level only, or removed (§2.3).
4. **H2** is one historical A-D4 evidence replay. It is not a seeded defect and is scored only as replay MATCH or MISMATCH (§3).

---

## 2. Case registry (23 cases)

### 2.1 Structure and totals

| group | IDs | n | scored as |
|---|---|---|---|
| historical defect | H1 | 1 | detection |
| historical evidence replay | H2 | 1 | replay MATCH/MISMATCH only |
| synthetic defect fixtures | S1–S6 | 6 | detection |
| clean component controls | K2–K8 | 7 | false alarms |
| positive decision-rule control | P1 | 1 | decision-rule conformance |
| abstention and decision-rule controls | U1–U7 | 7 | abstention |
| **total** | | **23** | |

1 + 1 + 6 + 7 + 1 + 7 = 23.

- **Detection** is scored on 7 cases (H1, S1–S6).
- **False alarms** are scored on 7 cases (K2–K8).
- **ID K1 is retired,** not reused: the former K1 is now P1.

### 2.2 Registry (one row per case)

**Columns:**
- **Level** says what the case can establish:
  - **C** is component conformance: the real component returns the registered output on this input.
  - **D** is dimension-rule conformance: the manuscript's roll-up rule is applied to that component output, *within the fixture only*.
  - **R** is claim-rule conformance: the decision procedure is applied to declared or derived dimension inputs.
- **No case establishes real dimension evidence or real claim-level evidence about Gohr or AutoND.** H2 reproduces recorded historical values; it adds none.
- **Expected response** is the registered output at the levels marked.

| ID | Condition (input) | Real component | Level | Expected response |
|---|---|---|---|---|
| H1 | existing checkpoint `audit/cryptography/evidence/ce1/best5depth10 (10).h5` (`de630afc…`), presented with the published description "depth 10" | `implementation.gohr_adapter.probe_checkpoint_architecture`; `cryptography.ce234.production.verify_reference_model` | C | probe reports 11 Conv1D and 5 Add (realized depth 5 ≠ declared 10); binding gate refuses |
| H2 | archived A-D4 evidence (§3) | `adapters.autond_d4.progression`; `experiments.d4_autond_label_shuffle.pipeline_progression`, `progression_criterion`, `autond_decision(role="co_primary", starting_round=5)`; stored-prediction rescoring | replay | every recorded value in §3.3 reproduced exactly |
| S1 | released SPECK32/64 reference module with α changed from 7 to 8 | `implementation.ii2_independent.verify_m1.test_published_vector`, `compare_key_schedules`, `compare_round_traces` | C, D, R | C: test vector mismatch, round traces diverge. D: in the fixture claim, the registered II criterion "published vector reproduces" is violated, so II FAIL (l.929). R: fixture claim with R_c = {II}, so NOT SUPPORTED (Table claim-decisions, row 1) |
| S2 | registered checkpoint `256eb4a5…` replaced by the existing file `audit/cryptography/evidence/ce1/signal_destroyed.h5` (`a110fad1…`) | `ce234.production.verify_reference_model` | C | hash gate refuses |
| S3 | synthetic partitions: 200 test rows copied from train | `dataset.d1_duplicate_detection.audit_duplicates`, `evaluate_d1` | C, D, R | C: D1 outcome FAIL, 200 overlaps. D: "DI is FAIL if any experiment gives FAIL" (l.2002), so DI FAIL within the fixture. R: fixture claim with R_c = {DI}, so NOT SUPPORTED |
| S4 | synthetic D2-N fixture, reduced scale (§5.1) | `dataset.d2_sample_dependence.near_neighbor_exact_screen`; `dataset.experiments.d2n_fresh_draw_null.rank_p`, `classify` | C | `classify` outcome EXCESS. **Not confirmatory DI evidence and not rolled up to DI** |
| S5 | synthetic per-epoch records: shuffled-label fits with round-5 max val_acc 0.507, round-6 max ≤ 0.505 | `adapters.autond_d4.progression`; `progression_criterion` | C | best_round 5, stop_round 6, `met = false`. **Progression-rule conformance only**; no DI or claim outcome registered |
| S6 | synthetic paired per-unit accuracies, 6 units: a declared single-factor "non-essential" change costing 0.05 | `experimental_validity.framework` `paired_analysis`, `assess_practical_equivalence`, `decide_difference_detection`, `decide_final` | C, D, R | C: the framework's final decision is a difference beyond the declared threshold. D: a "meaningful effect … with valid interventional attribution" (l.929–930; the single-factor design is declared, l.936–942), so EV FAIL within the fixture. R: fixture claim with R_c = {EV}, so NOT SUPPORTED |
| K2 | synthetic paired per-unit accuracies, change with no effect | EV framework (as S6) | C | equivalence established within the declared margin. **Not EV PASS**: one experiment does not cover every required EV criterion |
| K3 | existing registered checkpoint `audit/cryptography/Archive/best5depth10.h5` (`256eb4a5…`) | `verify_reference_model`, `probe_checkpoint_architecture` | C | hash, architecture (depth 10) and behavioural binding accepted |
| K4 | unmutated reference SPECK32/64 module | `verify_m1` functions (as S1) | C | test vector, key schedule and round traces agree. **Not II PASS**: II needs the independent-implementation comparison (l.1111–1112) |
| K5 | synthetic disjoint partitions | D1 (as S3) | C | D1 outcome PASS. **Not DI PASS**: DI PASS needs every PASS-capable experiment (l.2002) |
| K6 | synthetic D2-N fixture with no planting (§5.1) | D2-N functions (as S4) | C | NO_EXCESS. Reduced scale; not confirmatory |
| K7 | untampered PAR1 fake-adapter run directory | `implementation.ii4_parallel.verify.verify` | C | VERIFIED |
| K8 | synthetic per-epoch records: every shuffled fit stops at round 5 with best_round None; every baseline passes round 5 | `progression`; `progression_criterion` | C | `met = true`. No DI outcome registered |
| P1 | declared inputs: R_c = {II, DI, EV}, all PASS (asserted); reproduction COMPLETE and predictive advantage (asserted); one CE2 test SUPPORTED, preregistered and adequately powered (asserted); "criterion not satisfied by an executed control", sourced from `e6_certificate.json` field `interpretation.ce2_admission = "ADMITTED"` (`f071aae2…`); L* = L3 | engine only | R | L3; SUPPORTED. **A positive decision-rule control, not a clean empirical claim** |
| U1 | synthetic per-fit test counts: shuffled fits at accuracy 0.5053, n = 10⁶, band 0.5 ± 0.005 | `dataset.d4_controlled_perturbation.evaluate_near_chance_decision` | C, D, R | C: fits not shown inside the band, so INCONCLUSIVE. D: "anything else is INCONCLUSIVE" (l.2002), so DI INCONCLUSIVE. R: fixture claim with R_c = {DI}, so INCONCLUSIVE (row 2) |
| U2 | real A3 certificate (`d90ec393…`): `dimensions.CE3.scientific_decision = "SUPPORTED"`, `dimensions.CE3.adversarial_outcome = "NOT_CAUGHT"`, i.e. the CE3 criterion is satisfied by an executed adversarial control. Other inputs declared: R_c all PASS, reproduction COMPLETE, L* = L3, no other CE test | engine | R | CE3 does not qualify for L3 (l.1012–1015); level L2 < L3, so INCONCLUSIVE (row 3) |
| U3 | declared inputs: CE2 qualifying; R_c = {DI}; DI INCONCLUSIVE; L* = L3 | engine | R | INCONCLUSIVE (row 2); CE cannot lift the level above L1 |
| U4 | PAR1 fake-adapter run directory with the certified accuracy edited after the run | `ii4_parallel.verify.verify` | C | verifier refuses. **Detector-level verifier refusal only**; no claim consequence registered |
| U5 | declared inputs: R_c = {EV}; no EV record supplied; other dimensions PASS | engine | R | EV enters as INCONCLUSIVE with reason code `missing_prerequisite` (l.856, l.932), so INCONCLUSIVE (row 2) |
| U6 | synthetic EV per-unit records: 6 units, 2 valid, registered minimum 5 | EV framework (`failures.enforce_minimum_valid_replicates` via the certificate path) | C, D, R | C: INCONCLUSIVE, insufficient valid units. D: "a shortfall below the registered minimum of valid units gives INCONCLUSIVE" (l.918–920). R: fixture claim, so INCONCLUSIVE |
| U7 | declared inputs, two variants scored as one case: (a) CE2 SUPPORTED but not preregistered; (b) CE2 SUPPORTED, preregistered, not adequately powered. R_c all PASS; reproduction COMPLETE; L* = L3 | engine | R | both variants: CE2 not qualifying, so L2 and INCONCLUSIVE (row 3). The case is MATCH only if both variants match |

**Asserted inputs** are listed field by field in the registration:
- **P1:** every input except the E6 field.
- **U2:** every input except the A3 fields.
- **U3, U5 and U7:** all inputs.

**Fixture claims** in S1, S3, S6, U1 and U6 exist only to exercise the manuscript's roll-up and precedence rules on a real component output. They are not claims about any published work.

**H1 versus S2:**
- **H1** is a real historical artifact whose *description* is wrong. It is caught by reading the architecture.
- **S2** is an injected *file substitution*. It is caught by the hash gate.

### 2.3 Excluded case

| former ID | reason for exclusion |
|---|---|
| U8 (conflicting admissible records) | the manuscript has no conflict-resolution rule. Excluded until an approved rule exists. Not scored, not registered |

---

## 3. H2: historical A-D4 evidence replay

### 3.1 Claim-mapping history (preserved as recorded)

| protocol | assignment of A-D4 | when recorded | source |
|---|---|---|---|
| v1 | "Every AutoND DI and CE experiment bears on A1 and A2" | the original v1 protocol | manuscript Table revisions, row 4 (l.1566) |
| v2 | "Retraining experiments (A-D4, A-D5, A-CE1) bear on A1 only" | **after the A-D4 result was known.** First recorded in the D2-N registration, commit `5fb47d3`, 2026-11-05 04:12 UTC. A-D4 evidence was committed 2026-11-04 (`ad8c5d8`) | l.1566, l.1624–1626 |

- The v2 assignment is **not preregistered**, and V2 does not describe it as preregistered.
- V2 creates no A2 reassignment and alters neither row.
- H2 replays the evidence and reports its bearing on A1 as recorded under each protocol.
- H2 makes no new statement about A2.

### 3.2 Inputs

All inputs are existing and read-only.
- **Certificate:** `audit/dataset/evidence/d4_autond_speck3264_3ad26073acf0/result/d4_certificate.json`, SHA-256 `b4527041ff788e53f9781095897841aadb6bfaa4e0287aa3627c3bfbffbbd244` (tracked).
- **Archive:** `~/Downloads/results (1).zip`, 4,702,757,023 bytes, SHA-256 `3be7c5d5cc69e4b47fad73e0e0c28cdc88d4d13fc1f955829b477a9e0daa0be4`, 1,525 members.
- **Allowlisted members (671),** all under `deep_speck/audit/dataset/evidence/d4_autond_speck3264_3ad26073acf0/`:
  - 330 `fits/replicate_0k/{baseline,perturbed}/stage_rRR/checkpoint_epoch_EEE.metrics.json`;
  - 330 matching `.keras` files, **hashed only, never loaded**;
  - 10 `fits/predictions/replicate_0{1..5}_{baseline,perturbed}.npy`;
  - `dataset/test_y.npy`.
- **Verified read-only on 2026-11-06**, recorded in the allowlist:
  - every member path is relative, with no `..` and no symlink;
  - all 10 prediction files match the certificate's SHA-256;
  - all 330 records' `checkpoint_sha256` values equal the SHA-256 of the archived `.keras` member they name;
  - 330 = 5 baselines × 5 rounds × 10 epochs (250), plus perturbed fits of 10 + 20 + 20 + 20 + 10 epochs (80).

### 3.3 Recorded values the replay must reproduce

Threshold 0.505; starting round 5; role co_primary. Three distinct quantities:
- **selected_stage_round** (= `comparison_round`) is the round whose minimum-val_loss checkpoint is reloaded and scored. It is 5 for every fit by design. It says nothing about passing.
- **best_round** is the last round whose max val_acc exceeds 0.505, or None.
- **stop_round** is the first round whose max val_acc is ≤ 0.505. It is None if the cap is reached.

| fit | selected_stage_round | selected epoch | best_round | stop_round | rounds_trained | max val_acc by round |
|---|---|---|---|---|---|---|
| perturbed rep 1 | 5 | 10 | None | 5 | [5] | 0.5048 |
| perturbed rep 2 | 5 | 10 | **5** | **6** | [5, 6] | 0.5061, 0.5022 |
| perturbed rep 3 | 5 | 10 | **5** | **6** | [5, 6] | 0.5053, 0.5010 |
| perturbed rep 4 | 5 | 10 | **5** | **6** | [5, 6] | 0.5053, 0.5037 |
| perturbed rep 5 | 5 | 10 | None | 5 | [5] | 0.5002 |
| baseline reps 1–5 | 5 | 10 | 8 | 9 | [5 … 9] | rep 1: 0.9185, 0.7738, 0.6046, 0.5086, 0.5013 |

The registration will hold all ten fits' full values, copied from the certificate by script rather than retyped.

**What the replay must show:**
- **Exactly 3 of 5 shuffled fits (reps 2, 3, 4) passed round 5 and stopped at round 6.**
- **The co-primary criterion requires every shuffled fit to stop at round 5 with best_round None,** and every baseline to have a best_round. So:
  - `perturbed_stop_at_starting_round_without_best_round = false`;
  - `baseline_pass_the_starting_round = true`;
  - `met = false`.

### 3.4 Registered H2 outputs: three levels, kept separate

| level | registered output | source of the expected value |
|---|---|---|
| detector | `progression` recomputed from the 330 records equals each fit's recorded `best_round`, `stop_round`, `rounds_trained`, `rounds_max_val_acc` and `status`; `progression_criterion` as in §3.3; `autond_decision`: `engine_overall_d4_outcome = INCONCLUSIVE`, `autond_overall_outcome = INCONCLUSIVE`. PROGRESSION_NOT_CONFIRMED does not arise, because it applies only to a SUPPORTED engine outcome (`d4_autond_label_shuffle.py:293`). `baseline_adequacy_gate = PASSED` | certificate |
| DI (as recorded) | A-D4 gives DI INCONCLUSIVE for A1 under both protocols | l.1433–1434; l.2002 ("anything else is INCONCLUSIVE") |
| claim (as recorded) | A1 NOT SUPPORTED under v1 (II FAIL, DI INCONCLUSIVE, EV FAIL) and under v2 (II INCONCLUSIVE, DI INCONCLUSIVE, EV FAIL). The blocking reason is the admissible EV FAIL, plus II FAIL in v1; it is not A-D4 | l.1535, l.1586 |

**Stored-prediction rescoring (also part of the replay):**
- For each of the 10 arrays, `correct = Σ[(pred ≥ 0.5) == test_y]`. The ≥ 0.5 convention is from `d4_controlled_perturbation.py:423`.
- The counts must equal the recorded `baseline_correct` and `perturbed_correct` (for example, rep 1: 918,209 and 500,785 out of 10⁶) and the per-class counts.
- This is rescoring of stored outputs. **It is not model re-execution,** and it is reported in its own row (§4.2).

**H2 scoring.**
- MATCH at a level means every value at that level is reproduced exactly; otherwise MISMATCH.
- H2 does not enter detection, false-alarm, abstention or claim-outcome counts.

---

## 4. Baselines (one authoritative specification)

**No baseline outcome is equated with, or mapped to, claim support.**

### 4.1 B1: accuracy-only reproduction by fresh-data inference

- **Applies only** where an existing model and its released data generator both exist:
  - model: the case's existing `.h5`;
  - generator: root `speck.py:make_train_data` (`n`, `nr = 5`, default input difference), called for inference data only. No training function is called.
- **Procedure:**
  - generate a fresh seeded test set of n = 2×10⁵ pairs (seed registered);
  - run inference with the frozen stack (Keras 3.13.2, TF 2.20.0);
  - compute accuracy â with the prediction ≥ 0.5 convention.
- **Reported value** is the published Gohr 5-round depth-10 accuracy, 0.9291, as recorded in `implementation-certificate-V5.json` field `observation.findings[0].declared.reference_reported_best_val_accuracy` from `reference/train_5_rounds.py`. The same value is used for H1, S2 and K3, because each is presented as that published artifact.
- **Output (descriptive):**
  - â; Δ = â − 0.9291;
  - SE = √(â(1−â)/n);
  - the 95% Wilson interval for the accuracy;
  - the interval for Δ treating 0.9291 as fixed.
- **The reported value carries no published uncertainty,** and that is stated with the result.

**Tolerance.**
- No REPRODUCED/NOT_REPRODUCED threshold is used unless it is approved separately.
- The manuscript defines reproduction status (COMPLETE, SUBSTANTIAL, PARTIAL, NON-REPRODUCIBLE; l.1102–1108) by completeness and fidelity, not by a numeric accuracy tolerance.
- The earlier draft's max(0.005, 3·SE) has no manuscript basis:
  - 0.005 is the half-width of the *chance band* (l.1932), a different quantity;
  - at n = 2×10⁵, 3·SE ≈ 0.0017, so the term would always reduce to 0.005;
  - the published 0.9291 is a single best validation accuracy with unknown run-to-run spread.
- If a threshold is wanted, it must be proposed with its own rationale and approved before registration. Otherwise B1 stays descriptive.

| case | B1 status | input |
|---|---|---|
| H1 | EXECUTED (descriptive) | `de630afc…` |
| S2 | EXECUTED (descriptive) | `a110fad1…` (the substituted file) |
| K3 | EXECUTED (descriptive) | `256eb4a5…` |
| H2 | not B1: see B1-S (§4.2) | — |
| S1, S3, S4, S5, S6, K2, K4, K5, K6, K7, K8 | NOT_ASSESSED: the bundle describes a published pipeline with a reported accuracy, but no existing model or predictions exist for it | — |
| P1, U1–U7 | NOT_APPLICABLE: the case declares no reported accuracy; its inputs are decision-rule declarations or evidence records | — |

Totals: 3 executed + 1 B1-S + 11 NOT_ASSESSED + 8 NOT_APPLICABLE = 23.

### 4.2 B1-S: stored-prediction score recomputation (H2 only)

- This is the H2 rescoring of §3.4, reported in its own row.
- It is **not** counted as B1 coverage and **not** described as model re-execution.

### 4.3 B2: ordinary artifact and reproducibility checklist (executed)

**Item results:**
- **PASS / FAIL:** the item was executed.
- **NOT_ASSESSED:** the item's subject is part of the case, but the needed artifact does not exist or is excluded by policy (the reason is recorded).
- **NOT_APPLICABLE:** the item's subject does not exist in the case by construction.

**Checklist output:**
- FLAGGED if any item FAILs;
- NO_ISSUE_FOUND if at least one item PASSes and every other item is PASS or NOT_APPLICABLE;
- otherwise NOT_ASSESSED.

| item | executed check |
|---|---|
| C1 artifacts present | every file the bundle lists exists |
| C2 loadable | the model loads (h5py read plus Keras load, frozen stack), or the arrays load with `numpy.load(allow_pickle=False)`, or the released module imports |
| C3 environment documented | the bundle's environment fields are present |
| C4 accuracy | NOT_ASSESSED in every case while no B1 tolerance is approved (B1's Δ is reported descriptively beside it) |
| C5 published test vector | the SPECK32/64 vector (key `1918 1110 0908 0100`, plaintext `6574 694c`, ciphertext `a868 42f2`) through the released cipher code |
| C6 no exact train/test overlap | hash-set intersection of the released partitions (its own code, not D1) |
| C7 checksums | file SHA-256 equals the bundle's published checksum |
| C8 architecture and parameter count | Conv1D and Add counts and total parameter count equal the published description |

**Applicability** (E = executed, NS = NOT_ASSESSED, NA = NOT_APPLICABLE):

| case | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | notes |
|---|---|---|---|---|---|---|---|---|---|
| H1 | E | E | E | NS | E | NA | NS | E | C7: no published checksum exists for the historical file |
| H2 | E | E | E | NS | NS | NS | E | NS | C5, C6: cipher code and partitions not allowlisted. C8: `.keras` not loaded by policy. C7: certificate hashes |
| S1 | E | E | E | NS | E | NA | E | NS | no model |
| S2 | E | E | E | NS | E | NA | E | E | published checksum is `256eb4a5…` |
| S3 | E | E | E | NS | NA | E | E | NS | |
| S4 | E | E | E | NS | NA | E | E | NS | |
| S5 | E | E | E | NS | NA | NA | E | NS | |
| S6 | E | E | E | NS | NA | NA | E | NS | |
| K2 | E | E | E | NS | NA | NA | E | NS | |
| K3 | E | E | E | NS | E | NA | E | E | |
| K4 | E | E | E | NS | E | NA | E | NS | |
| K5 | E | E | E | NS | NA | E | E | NS | |
| K6 | E | E | E | NS | NA | E | E | NS | |
| K7 | E | E | E | NS | NA | NA | E | NS | |
| K8 | E | E | E | NS | NA | NA | E | NS | |
| P1, U3, U5, U7 | E | NA | NA | NA | NA | NA | NA | NA | declarations only |
| U1, U6 | E | E | E | NA | NA | NA | E | NA | |
| U2 | E | E | NA | NA | NA | NA | E | NA | certificate hash |
| U4 | E | E | E | NA | NA | NA | E | NA | |

C3 is constructed by us for every synthetic bundle, so its PASS carries no information about any real publication. This is reported.

### 4.4 B1-naïve (illustrative only)

- A stated inference policy, "a reproduced accuracy means the claim is accepted".
- Described in the report to illustrate the gap between reproduction and claim support, for example, H2's stored predictions rescore exactly while A1 is NOT SUPPORTED.
- **It is not executed, not a baseline and not scored.**

### 4.5 Metrics (per method, kept separate)

| metric | cases | framework | B2 | B1 |
|---|---|---|---|---|
| detection | H1, S1–S6 (7) | registered C-level output reached | FLAGGED | not scored (descriptive) |
| false alarms | K2–K8 (7) | any C-level output other than the registered clean one | FLAGGED | not scored |
| positive control | P1 (1) | MATCH/MISMATCH | NA | NA |
| abstention | U1–U7 (7) | MATCH/MISMATCH at each registered level | reported, not scored | NA |
| coverage | all 23 | — | items E vs NS vs NA per case | executed / NS / NA counts |
| replay | H2 (1) | MATCH/MISMATCH per level | reported separately | B1-S row |

No method is forced to a verdict. Results are reported per case, alongside the counts.

---

## 5. Fixture and evidence specifications

### 5.1 S4 and K6 (D2-N fixtures)

- **Functions:** imported read-only from `audit/dataset/experiments/d2n_fresh_draw_null.py` (SHA-256 `069c0bc5…`).
- **Rank p:** `rank_p(observed, null) = (1 + #{v ∈ null : v ≥ observed}) / (len(null) + 1)`.
  - Ties count against the observation (≥).
  - With 59 nulls, the smallest attainable p is 1/60.
- **Statistical gate:** `p ≤ α_per + 1e-12`, with α_per = 0.05 / 3 = 1/60. The 1e-12 guard exists so that the equality 1/60 = 0.05/3 is not decided by floating-point rounding.
  - Consequence: the gate passes **only** when the observed count is strictly greater than all 59 null counts.
  - A single tie with the maximum null gives p = 2/60 > α_per, so the gate fails.
- **Practical gate:** ratio = observed / mean(null) ≥ 1.25. If mean(null) = 0, the ratio is +∞ when observed > 0 and 1.0 when observed = 0.
- **Outcome:**
  - EXCESS if both gates pass;
  - INCONCLUSIVE if only the statistical gate passes;
  - otherwise NO_EXCESS.
- **S4:** n = 20,000, 2% one-bit near-duplicates planted, radius 8, 59 fresh null draws. The registered outcome is EXCESS. The fixture is built, and its observed and null counts are recorded, **before registration**. If the built fixture does not give strict exceedance of all 59 nulls, it is rebuilt with a registered change before registration, never after.
- **K6:** the same construction without planting. The registered outcome is NO_EXCESS.
- **Both** are reduced-scale algorithm-conformance fixtures. Neither is D2-N evidence or confirmatory DI evidence.

### 5.2 H2 allowlist preservation

- **`SDB_V2_H2_ALLOWLIST_DRAFT.json`** (`f7c2bde8…`) records, for each of the 671 members:
  - path, size, CRC-32 and SHA-256;
  - the archive's size, SHA-256 and member count;
  - the certificate hash;
  - the path-safety result and the hash-consistency results.
- **The builder** (`~/ce5_runs/sdb_v2_h2_allowlist_builder.py`) only streams member bytes into SHA-256 and parses the `.metrics.json` records with `json`. Nothing is extracted, unpickled, imported, loaded as a model or executed.
- **At run time,** each member is read again from the archive and refused unless its SHA-256 equals the allowlist entry:
  - `.npy` with `allow_pickle=False`;
  - `.metrics.json` with `json`;
  - `.keras` hashed only.
- **Preservation copy (needs your instruction):** if wanted, the 341 non-model members (330 records, 10 prediction files, `test_y.npy`) can be copied, unchanged, into a hashed V2 evidence directory. That requires your explicit instruction, because the archive is treated as untrusted and is not extracted without one. Until then, the archive itself plus the allowlist is the preserved form.

---

## 6. Decision engine V2 (specification; no code yet)

### 6.1 Rules implemented (exact manuscript text only)

| rule | text implemented | source |
|---|---|---|
| component | c = (q, H, R_c, L*_c), R_c ⊆ {II, DI, EV} | l.772 |
| dimension outcomes | PASS, FAIL, INCONCLUSIVE, N/A; N/A means "excluded, with a reason, at claim decomposition", so a dimension in R_c cannot be N/A (registration error) | l.927–933 |
| admissibility | a FAIL counts only if admissible; an associational observation "cannot independently produce a dimension-level FAIL" | l.935–944, l.1047–1049 |
| units | below the registered minimum of valid units, INCONCLUSIVE | l.918–920 |
| absent prerequisite | INCONCLUSIVE, with reason code `missing_prerequisite` | l.856, l.932 |
| L0 | predictive advantage or required reproduction not established | Table evidence-levels |
| L1 | reproducible predictive evidence exceeding a declared baseline | same |
| L2 | L1 and PASS on every applicable dimension in R_c | same |
| L3 | L2 and ≥ 1 CE test that is (i) preregistered, (ii) adequately powered, (iii) SUPPORTED, and (iv) whose criterion is not satisfied by any executed adversarial control. Tests failing (iv) "corroborate but cannot establish L3" | Table evidence-levels, l.1012–1015 |
| L4 | L3 and ≥ 2 independent evidentiary lines supporting the same interpretation, differing in inferential method **and** independent evidence source, with no shared unverified component capable of explaining the result | Table evidence-levels, l.1017–1020 |
| precedence | (1) a required dimension has a valid FAIL, or a required CE test has a valid NOT SUPPORTED, gives NOT SUPPORTED; (2) a required prerequisite is non-reproducible, absent or INCONCLUSIVE, gives INCONCLUSIVE; (3) level below L*_c gives INCONCLUSIVE; (4) all required evidence and thresholds satisfied gives SUPPORTED | Table claim-decisions |
| composite | SUPPORTED iff all components SUPPORTED; NOT SUPPORTED iff all NOT SUPPORTED; PARTIALLY SUPPORTED iff all resolved and both occur; else INCONCLUSIVE | l.1078–1087 |
| reproduction-only | R_c = ∅, L* = L1: SUPPORTED when reproduction status is COMPLETE or SUBSTANTIAL | l.1095–1097 |
| DI roll-up (used by D-level fixtures only) | FAIL if any experiment FAILs; PASS iff every PASS-capable experiment PASSes and no screen FAILs; otherwise INCONCLUSIVE; D4 band and floor mapping | l.2002 |

**How the L3 and L4 qualifiers are represented and tested:**
- **Each CE test record** carries:
  - `preregistered`, with a reference to a registration hash recorded before its data;
  - `adequately_powered`, with a reference to the registered power analysis or valid-unit minimum;
  - `satisfied_by_executed_control`, with a reference to the control certificate field (for example, U2's A3 field).
- **The engine checks the four L3 conditions conjunctively,** exactly as written. Tests vary each condition alone, holding the other three true, and all 16 combinations.
- **Each evidentiary line** carries: `supports`, `interpretation`, `method`, `source`, and `unverified_components`, the set of unverified components *registered as capable of explaining the result*.
- **L4 holds** iff L3 holds and some pair of supporting lines has the same interpretation, different method, different source, and an empty intersection of `unverified_components`.
  - Tests vary each of these five conditions alone, and over all 32 combinations for a pair.
  - Lines are a separate input. They are not assumed to be CE tests.
- **What the engine cannot judge:**
  - whether a power analysis is adequate;
  - whether a component is "capable of explaining the result".
  
  It applies the registered attestations. The verifier checks that each cited reference exists and has the cited hash.
- **The L3 admission condition is itself a revision** made after the CE results were known (l.1025–1027). That is reported as a limitation (§9).

### 6.2 Text the manuscript leaves undecided

The engine and the oracle return `UNSPECIFIED` here rather than a verdict. Each item needs your approval as a labelled specification addition, or it stays UNSPECIFIED.

| item | what is undecided |
|---|---|
| X1 | whether reproduction status SUBSTANTIAL establishes L1 for a component with R_c ≠ ∅ (the text says so only for reproduction-only components) |
| X2 | whether PARTIAL establishes L1, or is a "non-reproducible" prerequisite under row 2 |
| X3 | a required dimension with an inadmissible FAIL when L*_c ≤ L1: rows 1–3 do not fire, and row 4's "all required evidence satisfied" is not met |

- **No paper verdict depends on X1–X3.** The 7 regression rows use COMPLETE and admissible outcomes.
- **None of the 23 cases depends on X1–X3.** No case input uses SUBSTANTIAL, PARTIAL or an inadmissible FAIL.

### 6.3 Oracle

- **`oracle_v2.py`** is a separate module written from the manuscript text in §6.1–6.2, structured as an explicit predicate table (one predicate per table row and level condition).
- **It shares no code with the engine,** and imports neither engine (V1 or V2) nor any benchmark module.
- **Expected values in tests come from the oracle or from hand-written literals.** Never from an engine function.

### 6.4 State-space coverage (stated exactly)

| block | coverage | factors | size |
|---|---|---|---|
| E1 | exhaustive | per dimension (3): in R_c or not (2) × outcome {PASS, FAIL-admissible, FAIL-inadmissible, INCONCLUSIVE, N/A} (5), giving 10³ = 1,000 (combinations with a required N/A must raise a registration error); × reproduction status (4) × predictive flag (2) × L*_c ∈ {0..4} (5); × CE configuration: none, or one test over outcome (3) × required (2) × preregistered (2) × powered (2) × control-satisfied (2), i.e. 49; no lines | 1,000 × 40 × 49 = 1,960,000 |
| E2 | exhaustive | R_c = {II, DI, EV}, all PASS; COMPLETE; predictive; one qualifying CE test; L* ∈ {3, 4}; lines: 0, 1 (supports ∈ {T, F}), or 2 lines (supports 2 × 2 × same interpretation 2 × method differs 2 × source differs 2 × unverified intersection empty 2 = 64) | 2 × 67 = 134 |
| E3 | exhaustive | composite rule over every sequence of 0–4 component verdicts from {SUPPORTED, NOT SUPPORTED, INCONCLUSIVE} | 121 |
| P | pairwise (strength-2 covering array, deterministic construction, generator seed registered) | all E1 factors, plus 2–3 CE tests and 0–3 lines with every line attribute | reported at registration |
| R | random | 100,000 uniform draws over the P factor space, seed registered | 100,000 |
| B | boundary | each level transition L0→L1→…→L4, each with its one-condition-removed neighbour; CI-at-margin inputs for the D4 band and EV thresholds | enumerated in registration |

- **Agreement is required on every case** in E1, E2, E3, P, R and B. That includes agreeing on `UNSPECIFIED` and on the raised registration errors.
- **Only E1, E2 and E3 are exhaustive.** P, R and B are not, and the report says so.

**Paper-verdict regression:**
- The 7 manuscript verdicts are run as a **consistency check only**: G0; G1 v1/v2; A1 v1/v2; A2 v1/v2.
- Those verdicts were produced by applying these same rules, so agreement with them is not independent evidence that the rules are correct, and it is not presented as such.
- Each row states its encoding:
  - **G1 v1:** Gohr II is encoded as PASS. This is read from Table protocols (l.1534), whose v1 column lists only DI and EV as open for G1; the claim register at l.1585 gives the v2 reading (II INCONCLUSIVE). It is re-run with II INCONCLUSIVE, to check the manuscript's statement that G1's verdict is the same under either reading (l.1322–1324). No category beyond PASS/FAIL/INCONCLUSIVE/N/A is used.
  - **A1 rows** also check the blocking reasons (§3.4).

---

## 7. Input manifest (schema, bound into the registration)

```
manifest.json
  schema: "sdb-v2-manifest-1"
  created_at_utc
  source_state: { repo, branch: "audit-framework", head_commit, git_status_bound_paths[],
                  uncommitted_v2_files: [{path, sha256}] }
  environment: { python, numpy, scipy, h5py, keras, tensorflow, platform, pip_freeze_sha256,
                 pip_freeze_file }
  sources: [{path, sha256, role: "v2" | "frozen-component" | "generator", imported_by[]}]
  transitive_import_check: { closed: true, missing: [] }
  evidence: [{id, path, sha256, kind: "checkpoint" | "certificate" | "archive" | "archive-member",
              archive_member_of?, read_mode: "hash-only" | "npy-no-pickle" | "json" | "h5-inference"}]
  fixtures: [{case_id, path, sha256, builder, builder_sha256, seed}]
  configuration: { case_id: {parameters, seeds, thresholds, tolerances} }
  cases: [{case_id, group, level_marks, expected: {C?, D?, R?, replay?}, asserted_inputs: [{field, value}],
           real_inputs: [{field, evidence_id, json_pointer}]}]
  baselines: { B1: {cases, seed, n, reported_accuracy_source}, B1S: {case: "H2"}, B2: {applicability_table} }
  oracle: { path, sha256 }
  raw_record_paths: [...]
```

- **Transitive imports:** registration refuses to proceed if any imported project module is absent from `sources`.

**`sources`, frozen components (at minimum):**
- `implementation/gohr_adapter.py`;
- `implementation/ii2_independent/{verify_m1,speck_independent}.py`;
- `cryptography/ce234/{production,frozen_plan}.py` and `cryptography/gohr/speck.py`;
- `dataset/{d1_duplicate_detection,d2_sample_dependence,d4_controlled_perturbation}.py`;
- `dataset/experiments/{d2n_fresh_draw_null,d4_autond_label_shuffle}.py` and `dataset/adapters/autond_d4.py`;
- `experimental_validity/framework/{certificate,statistics,experiment,failures,replication,power}.py`;
- `implementation/ii4_parallel/{verify,scheduler,worker}.py` and its fixtures;
- `tests/test_ii4_comparison.py` (FakeAdapter);
- root `speck.py` (generator only).

**`evidence`:**
- checkpoints `de630afc…`, `256eb4a5…`, `a110fad1…`;
- certificates A3 `d90ec393…`, E6 `f071aae2…`, A-D4 `b4527041…`, implementation V5 `0ce25b3b…` (B1 reported-accuracy source);
- the archive `3be7c5d5…` and its 671 allowlisted members.

**Raw records kept:**
- per-unit accuracies; count vectors; full null lists;
- per-epoch histories (H2: the 330 records);
- layer counts;
- B1 per-example correctness vectors (packed bits) and their hashes;
- B2 per-item evidence;
- verifier reports;
- fake-adapter run directories.

---

## 8. Independent verifier (`verify_sdb_v2.py`, specification)

- **Imports:** Python standard library, numpy, scipy and h5py only. **No project module:** not `seeded_defects.py`, `seeded_defects_v2.py`, `claim_engine.py`, `claim_engine_v2.py`, `oracle_v2.py`, and none of the frozen framework components.
- **Fails closed** (non-zero exit, with the first discrepancy named) on:
  - registration or manifest hash mismatch;
  - any source, evidence or fixture hash mismatch;
  - an allowlist member hash mismatch;
  - any recomputed value that differs from the recorded one.
- **Recomputes, with its own code:**
  - **D1:** overlap and duplicate counts from the partitions.
  - **D2-N fixtures:** rank p, ratio and the gates from the full null list, using the convention of §5.1.
  - **D4 (U1):** per-fit binomial band tests from the counts.
  - **EV (S6, K2, U6):** paired difference, CI, TOST, threshold decision and valid-unit count, from the per-unit records.
  - **Progression (S5, K8, H2):** best_round and stop_round from the per-epoch records, using the stopping rule written out independently: per round, max val_acc against 0.505, stopping at the first round at or below it.
  - **H2:** the record chain (`previous_record_sha256`); `checkpoint_sha256` against the allowlist; the 10 rescored correct counts and per-class counts from the prediction arrays and `test_y.npy`.
  - **Architecture (H1, K3):** Conv1D and Add counts from the h5 layer groups.
  - **Binding (S2, K3):** the hash, plus the recorded behavioural accuracies.
  - **PAR1 (K7, U4):** re-reading the archived verifier report and checking the run directory's hashes.
  - **Claim decisions (P1, U2, U3, U5, U7 and the fixture claims):** its own precedence implementation.
  - **Baselines:** B1 â, Δ, SE and intervals from the stored correctness vectors; B2 items from the bundles.
  - **Aggregates:** every count in §4.5.

---

## 9. Limitations (to be reported with V2)

1. **Constructed cases.** 23 cases, designed with knowledge of the detectors.
   - This is a registered conformance benchmark. It estimates no population detection or false-alarm rate and supports no general accuracy claim.
2. **No real evidence is established.** No case establishes real dimension or claim evidence:
   - the D- and R-level checks use fixture claims or declared inputs;
   - H2 reproduces recorded values only.
3. **Asserted inputs** are confined to P1, U2, U3, U5 and U7, and are listed field by field.
4. **Reduced scale.** S4 and K6 are reduced-scale fixtures.
5. **Coverage gaps are left open:**
   - B1 executes on 3 of 23 cases and is descriptive;
   - B1-S covers H2 only;
   - B1-naïve is illustrative and unscored;
   - B2's C3 is uninformative on constructed bundles.
6. **H2's A1-only assignment** was recorded after the A-D4 result. H2 validates no prospective decision.
7. **Post-result revision.** The L3 admission condition was revised after the CE results were known (l.1025–1027).
8. **Unfilled gaps:**
   - X1–X3 return UNSPECIFIED;
   - conflict resolution and invalid-evidence handling have no rule;
   - U8 is excluded;
   - U4 and S5 are detector-level only.
9. **What conformance shows.** It shows the engine implements the written rules, not that the rules are right. The paper-verdict regression is a consistency check, not independent proof.

---

## 10. Decisions needed before registration

1. **X1–X3 (§6.2):** approve a written specification addition for each, or keep UNSPECIFIED.
2. **B1 tolerance (§4.1):** keep B1 descriptive (the default in this draft), or approve a tolerance with a stated rationale.
3. **H2 preservation copy (§5.2):** copy the 341 allowlisted non-model members into a hashed V2 evidence directory, or keep the archive plus allowlist as the preserved form.
4. **Anchoring:** commit the registration, manifest, fixtures and allowlist to `audit-framework` before execution, or keep local-only ordering evidence.
