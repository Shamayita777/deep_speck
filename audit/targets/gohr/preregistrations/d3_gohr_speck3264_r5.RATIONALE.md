# D3 Gohr preregistration — rationale for the draft

Draft: `d3_gohr_speck3264_r5.json` (schema `d3-preregistration-3`). **Status: UNAPPROVED DRAFT.**
Nothing in the draft is an approval. The prior-exposure attestation for the D1/D2 records (item 3 of `data_examined_before_approval`) is answered, and so is the attestation for the Gohr D2 calibration results (item 3b). The approval fields remain blank.

Runner: `audit/dataset/experiments/d3_gohr_confirmatory.py`. Calibration tool: `audit/dataset/tools/d3_gohr_method_validity_calibration.py`. D0 build attestation: `audit/dataset/tools/d0_gohr_attested_build.py`.

## 1. What this audit is

A **bound-reporting audit** on one persisted canonical Gohr instance (Speck32/64, 5 rounds, input difference (0x0040, 0x0000), 10,000,000 / 1,000,000 / 1,000,000 rows), preregistered **before that instance is generated**.

- **Guarantees.** With probability ≥ 1 − 0.005, every reported upper and lower bound of every confirmatory hypothesis holds simultaneously (G1). With probability ≥ 1 − 0.005, no confirmatory exact detection is false (G2). Jointly, P(any G1 or G2 error) ≤ 0.01 (union bound).
- **Conditions.** Every guarantee is conditional on the assumptions in §5.
- **Scope.** Each hypothesis concerns one bit or one bit pair. Nothing is claimed about classifiers that use more than two bits, the 64-bit model, or the full 64-bit joint distribution.

## 2. Decision semantics (frozen engine; nothing here changes it)

The engine is `audit/dataset/d3_distribution_statistics.py` (method `d3-confirmatory-5`, decision rule `d3-decision-4`). Outcomes are produced in three steps.

**Step 1 — each hypothesis** (`classify_hypothesis`, lines 1317–1350). Inputs: the threshold t, the simultaneous lower and upper bounds, the point effect, and whether the exact test detected. The rules are applied in this order:

| Condition | Category | Outcome |
|---|---|---|
| lower > t (strict) | `CONFIRMED_DEVIATION` | **FAIL** |
| upper ≤ t and an exact detection | `STATISTICAL_ONLY` | **CONDITIONAL_PASS** |
| upper ≤ t and no detection | `EQUIVALENT_WITHIN_TOLERANCE` | **PASS** |
| bounds straddle t, with a detection | `SIGNIFICANT_PRACTICALITY_UNRESOLVED` | INCONCLUSIVE |
| bounds straddle t, no detection, point effect > t | `PRACTICAL_UNSUPPORTED` | INCONCLUSIVE |
| bounds straddle t, no detection, point effect ≤ t | `INSUFFICIENT_PRECISION` | INCONCLUSIVE |

The category-to-outcome map is `CATEGORY_OUTCOME` (line 930). Non-finite or inverted bounds are refused with an error, never classified.

**Step 2 — each family** (line 1380). A confirmatory family's outcome is the first of FAIL, INCONCLUSIVE, CONDITIONAL_PASS, PASS (`_OUTCOME_ORDER`, line 940; `_combine_outcomes`, line 1353) that occurs among its hypotheses. One FAIL makes the family FAIL; otherwise one inconclusive hypothesis makes it INCONCLUSIVE. The exploratory family E is `NOT_GATED`.

**Step 3 — the audit** (`evaluate_confirmatory_decision`, lines 2252–2338). The provisional outcome combines the six confirmatory family outcomes the same way (line 2283). It becomes the reported outcome only if every gate holds: preregistration, calibration, input provenance, design equality, and approval before the audit. Otherwise the outcome is INCONCLUSIVE and every failing gate is listed (line 2338).

**What each outcome states:**
- **FAIL** rests on G1: with the stated confidence, the effect exceeds t.
- **PASS** rests on G1 and the absence of an exact detection. It states only that every declared single-bit or bit-pair effect has a simultaneous upper bound at or below its operational threshold, under §5's assumptions, at these sample sizes. It does **not** state that the threshold is a validated practical-equivalence margin. It does **not** state equality of the full 64-bit joint distribution.
- **CONDITIONAL_PASS** adds that an exact test detected a difference smaller than t.
- **INCONCLUSIVE** certifies nothing at t; the information is in the reported bounds.
- **`PRACTICAL_UNSUPPORTED` is not a deviation.** The plug-in TVD is biased upward. In the AutoND run, such counts were driven by ordinary single-bit marginal noise.

**On the approval statement.** The engine's fixed `APPROVAL_STATEMENT` approves the tolerances "as operational FAIL thresholds, not validated equivalence margins". For this protocol the same thresholds also define the PASS bound statement above. The statement stays accurate in what it asserts: the thresholds are operational and are not validated equivalence margins. It is not altered here, because that would change the engine and invalidate every calibration binding. The approver should sign it with this reading.

## 3. Why the generator implies the null laws

Source: repository-root `speck.py` (`make_train_data`, lines 114–127; `convert_to_binary`, lines 83–90). SHA-256 `59296f7b7990ec3886fc524ff49256005d360eb3748d3b1570351d669acbb8ce`, git blob `28b2f836c2642b44a7bb3a445adfec2e99b36521`, unchanged since Gohr's 2019 upload (commit `fba2fd7`).

**Assumption throughout:** `os.urandom` behaves as an ideal source of independent uniform bytes, and the code runs as written.

- **Labels.** `Y = urandom byte & 1` is Bernoulli(½), independent per row, so R1's null (ratio ½) is exact.
- **Keys.** Each row has its own uniform 64-bit key (four uniform 16-bit words), independent of everything else in the row.
- **Y = 1 rows.** For a fixed key k, Speck encryption E_k is a permutation of {0,1}³². P is uniform and independent of k, so C0 = E_k(P) is uniform. P ⊕ (0x0040, 0) is also uniform, so C1 = E_k(P ⊕ Δ) is uniform. Hence every bit is ½ and every within-half pair is uniform. C0 and C1 are **not** independent: the cross-half joint law carries the differential that the distinguisher learns.
- **Y = 0 rows.** Both plaintexts P0 and P1 are independent and uniform, and both are encrypted under the **same** uniform key. Given k, E_k(P0) and E_k(P1) are images of independent uniform inputs under one permutation, so they are independent and uniform, with a law that does not depend on k. The 64-bit row is therefore exactly uniform. AutoND reaches the same law by overwriting the row with random bits; Gohr reaches it by encryption.
- **Layout.** `convert_to_binary([c0l, c0r, c1l, c1r])` uses 16-bit words, MSB first: columns 0–31 are C0 and 32–63 are C1 (the runner's `FEATURE_GROUPS`).
- **Consequences.** Under these assumptions the following nulls are all **exactly true**:
  - R2: every class-conditional bit is ½;
  - D: bit marginals are equal across classes;
  - A/B/C: pair laws are identical across partitions, since each partition is an independent `make_train_data` call from the same law.
- **Not claimed:** a uniform 64-bit law for Y = 1; independence of C0 and C1; uniform cross-half pairs given Y = 1.

## 4. Design (computed from the code, not transcribed)

| Item | Value | Basis |
|---|---|---|
| Global α | 0.01, joint over G1 and G2 | `joint_alpha_problem`: 2 × Σ(allocation) ≤ 0.01 |
| Confirmatory families | R1, R2, A, B, C, D | R1 is confirmatory because the instance did not exist when the protocol was fixed |
| Allocation | 0.005 / 6 = 8.333×10⁻⁴ each (Σ = 0.005; 2Σ = 0.01) | Equal split; approver decision; no evidence-based weighting exists |
| E | Exploratory, α 0.01 within E (6,048 hypotheses, 1.653×10⁻⁶ each) | Never gates; outside the budget |
| Thresholds | `class_ratio` m, `class_conditional_bit` 2m, `pairwise_tvd` m, `cross_class_bit` 2m, with m = 0.01 | II-4 margin (§6) |

Each confirmatory family's α serves both its bounds (G1) and its tests (G2).

| Family | Count | α per hypothesis | Clopper–Pearson α per interval | Exact-test α |
|---|---|---|---|---|
| R1 | 3 | 2.7778×10⁻⁴ | 2.7778×10⁻⁴ | binomial 2.7778×10⁻⁴ |
| R2 | 384 | 2.17014×10⁻⁶ | 2.17014×10⁻⁶ | binomial 2.17014×10⁻⁶ |
| A, B, C | 6,048 each (2,016 pairs × 3 comparisons) | 1.37787×10⁻⁷ | 1.7223×10⁻⁸ (8 cells) | Fisher cell 3.4447×10⁻⁸ (4 live cells) |
| D | 192 | 4.34028×10⁻⁶ | 2.17014×10⁻⁶ (2 proportions) | Fisher 4.34028×10⁻⁶ |

### Expected uniform-cell design-resolution illustrations

The engine's `design_resolution()` computes the simultaneous upper bound for **observed tables equal to their expected cell proportions** under the reference law: cells ½ for single bits, ¼ for bit pairs, and classes at exactly the stated fraction. These figures are **illustrations under the intended reference scenario**. They are not bounds over all possible observed tables and not a power analysis. The calibration's S5 recomputes them, and `test_d3_gohr_preregistration.py` checks them against it.

| Family | Train | Validation / test | Threshold |
|---|---|---|---|
| R1 | 0.00057 | 0.00182 | 0.01 |
| R2 (per class, 50%) | 0.00106 | 0.00335 | 0.02 |
| D (50%) | 0.00212 | 0.00670 | 0.02 |
| A (pooled) | 0.00644 (train vs held-out) | 0.00977 (validation vs test; 0.0097669) | 0.01 |
| B / C (50% class fraction) | 0.00911 (train vs held-out) | 0.01381 (validation vs test; 0.0138135) | 0.01 |
| B / C (45% class fraction) | — | 0.014561 | 0.01 |

**Why B/C precision is inadequate for m under the reference scenario:**
- At expected uniform cells, the B/C validation-vs-test upper bound (0.0138 at 50%, 0.0146 at 45%) exceeds m = 0.01.
- More generally, for **every** table the upper bound ½ Σ_c max(U_ac − L_bc, U_bc − L_ac) is at least ¼ Σ_c (width_ac + width_bc), because max(x, y) ≥ (x + y)/2.
- That floor depends only on the interval widths, not on the observed differences. Evaluated numerically on a grid of near-uniform tables (cell fractions 0.20–0.30, class strata of 450,000–550,000 rows), it lies between 0.0122 and 0.0154, above m.
- So, under the reference law, sampling noise alone is not expected to let a B/C validation-vs-test hypothesis reach PASS at m. B and C, and with them the audit, are expected to be INCONCLUSIVE unless something FAILs.

**This is not a universal impossibility proof.** Tables far from the reference law have much narrower intervals. For example, a pair whose rows all fall in one cell has an upper bound of about 7×10⁻⁵ at these sizes. A's validation-vs-test figure (0.00977) is just under m, and realised bounds are usually wider than the expected-table value.

**These facts follow from sample sizes, alphas and a threshold fixed in advance. The thresholds were not chosen, and will not be changed, to make PASS more likely.**

**Illustration only** (not a power analysis or a prediction): at identical sizes, the AutoND run's realised A validation-vs-test upper bounds ranged from 0.0098 to 0.0122, at a slightly smaller α.

## 5. Assumptions, by test

**Precise conditional statement.** Every guarantee assumes i.i.d. rows within each sample. The exact tests additionally assume that the two compared samples are independent. **D1 and D2 do not prove these assumptions.** They probe specified duplicate and dependence patterns only, and the existing Gohr D1/D2 results concern other instances, so they say nothing about the new D0 instance. The assumptions rest on the generator argument of §3 and are stated, not established.

- **One-sample tests and bounds.** R1 (class ratio vs ½) and R2 (bit vs ½) use exact binomial tests and Clopper–Pearson intervals. They need i.i.d. rows within the sample (a partition, or a class stratum of a partition).
- **Two-sample comparisons.** These are D (Y = 1 vs Y = 0 within a partition), A/B/C (partition vs partition) and E.
  - **G1 bounds** use separate Clopper–Pearson intervals per sample and need only i.i.d. rows within each sample.
  - **G2 exact tests** (Fisher conditional tests) additionally need the two samples to be independent. For D, the classes within one partition are independent given i.i.d. (X, Y) rows. For A/B/C, the partitions come from separate generator calls.
- **Dependence among tests vs among rows.** Bonferroni over hypotheses, cells and families holds under **arbitrary dependence between tests**: overlapping pairs sharing a bit, the train partition reused across comparisons, multinomial cells. It does **not** relax the requirement that **rows** be i.i.d. Dependent rows would make the intervals too narrow and the tests anti-conservative.
- **D1.** A duplicate census can detect exact repeats, but it **does not establish i.i.d. sampling**: absence of duplicates is compatible with many dependent processes.
- **D1 gate (frozen policy).** Run the instance-bound D1 census (`audit/dataset/experiments/d1_gohr_canonical_census.py`) against the exact persisted D0 instance. If D1 does not return the preregistered PASS outcome, stop before D3 production, preserve all generated data and evidence, and investigate. Do not regenerate the dataset or choose another instance merely to obtain a PASS. A D1 PASS does not prove i.i.d. sampling, does not establish independence, and does not validate the other statistical assumptions of §5. The D3 runner enforces this gate itself: it refuses both preflight and production (`--d1-dir`) unless the D1 evidence is complete, hash-consistent, bound to the same instance, manifest and build attestation, and reports PASS; the D1 certificate's SHA-256 enters the D3 provenance chain.
- **D2.** D2 has not been run on this instance. The earlier Gohr D2 rerun (another instance) recorded radius-8 near-duplicate counts of 111, 12 and 7 among 20,480 queried rows, against roughly 57, 5.7 and 5.7 expected at full retrieval for independent *uniform* 64-bit rows. Pooled Gohr rows are not uniform (Y = 1 rows carry the differential), so this comparison does not by itself indicate row dependence, and D2's own null could not resolve it; it is recorded as an unresolved observation, and the i.i.d. condition remains an assumption.
- **No D2 gate (approver decision, 2026-10-23).** No instance-bound D2 gate is part of this protocol. The available Gohr D2 runner regenerates its data and is not instance-bound; the existing Gohr D2 sensitivity calibration covers engine schema 6.4, not the later 6.5 rerun; and D2 cannot certify i.i.d. sampling in any case. The earlier D2 observations concern another, unpersisted dataset. D1 PASS detects exact duplicates only. D3's inferential claims therefore remain conditional on i.i.d. rows within each sample and independent compared samples; independence is assumed, not proved.
- **Low order only.** Equality of all 1- and 2-bit marginals does not imply equality of the 64-bit joint law, and many sub-threshold coordinate effects can combine in a full-width classifier.

## 6. The thresholds

**Source.** m = 0.01 is `DESIGN.delta_materiality` of **II-4-GOHR-IMPL-V1** (`audit/implementation/preregistrations/ii4_impl_gohr_v1.py:96`).
- The frozen record `ii-4-gohr-impl-v1.frozen.json` (commit `d89590f`, 2026-10-13) pins that file by SHA-256 (`09d8a105…`) under `source_manifest.preregistration`.
- Its `preregistration_sha256` field is a different object: the plan's design hash.
- The margin bounds the absolute difference in sealed test accuracy between the reference and an independent Gohr implementation (reference 5-round accuracy 0.9291).
- The runner re-derives m by checking the file hash and parsing the literal (`materiality_margin()`), and refuses a preregistration whose tolerances differ.

**What the thresholds are and are not:**
- They are an **operational design convention**: the AutoND rule (reuse the target's own frozen accuracy margin) applied to Gohr's frozen margin.
- They are **not a mathematical conversion** from an accuracy difference to a TVD or proportion difference.
- They are **not validated practical-equivalence margins.**
- They are **not a certificate of full-distribution equivalence.**

The readings in the JSON justifications (constant classifier, single-bit classifier, pair-restricted classifier) explain what each threshold caps, for that restricted classifier only. The legacy Gohr thresholds (0.005 / 0.01 / 0.01) carry no recorded derivation and were not adopted. AutoND's m = 0.00144 derives from AutoND's own margin and does not transfer.

## 7. Prior observations (disclosed; they justify nothing)

The following were examined before this draft:
- the legacy Gohr D3 PASS (on an unpersisted instance, under the legacy method);
- the AutoND D3 INCONCLUSIVE.

Neither shows that these thresholds, this allocation or the new calibration are appropriate. The legacy PASS meant non-detection under an asymptotic test with per-partition α, and the AutoND outcome concerns a different dataset.

For the Gohr D1 and D2 results (each on its own unpersisted instance), the approver saw the outcome labels only and did not inspect the underlying records or statistics before approval. The protocol drafter inspected the complete records on 2026-10-23, during protocol development and before approval; their content is summarised in item 3 of `data_examined_before_approval`. The protocol was not drafted blind to these outcomes, and none of them validates this design.

The Gohr D2 detection-sensitivity calibration (item 3b; `audit/dataset/evidence/d2/calibration/d2_calibration_final_v1.json`) is a synthetic fault-injection calibration of the D2 engine at schema 6.4 on a fresh 100,000-row instance. It is a separate artifact from the later D2 rerun (schema 6.5, whose own calibration slot is `NOT_RUN`) and does not demonstrably calibrate that rerun's detector. The approver ran it and, before this protocol review, opened its JSON or examined its individual per-condition detection rates. The protocol drafter inspected the complete artifact during protocol development.

## 8. Calibration

The AutoND calibration `c39726e5` **cannot** be used:
- making R1 confirmatory changes the allocation and every per-test α;
- the engine verifier binds a calibration to the tool file the runner passes (`d3_gohr_method_validity_calibration.py`);
- the runner also checks the calibrated roles, allocation, sizes and class-fraction range.

A new calibration run is required before production. It is not run in this phase.

```
cd ~/Documents/deep_speck && PYTHONDONTWRITEBYTECODE=1 \
~/.venvs/autond-tf215-audit-20261015T050853Z/bin/python -B -m audit.dataset.tools.d3_gohr_method_validity_calibration \
  --out-dir audit/dataset/evidence/d3_gohr_method_validity_<id>
```

| Check | Kind | What it shows |
|---|---|---|
| S0 | test run | the bound suite `test_d3_method_validity.py` passes on this engine |
| S1 | exact computation | binomial-test size at the R1 and R2 α |
| S6 | exact computation | Clopper–Pearson non-coverage at every interval α above, over an (n, p) grid |
| S8 | exact computation | Fisher cell-test size at every cell α above |
| S7 | Monte Carlo | TVD-bound coverage, at α ≥ 0.001 only |
| S2 | Monte Carlo | complete gated path, false-detection rate ≤ the G2 budget, on a synthetic law at reduced sizes |
| S3–S5 | reported only | power, noise floor, expected-table resolution |

None of these proves row independence, and none validates the practical meaning of any threshold.

## 9. D1 on the persisted instance

**The legacy runner cannot consume a persisted instance.** `audit/dataset/experiments/d1_gohr_exact_census.py` always regenerates data (`run_d1` → `generate_gohr_dataset` → `GohrAdapter.generate_partition`) and has no instance argument.

**An instance-bound pattern exists for AutoND.** `d1_autond_exact_census.py` verifies the D0 instance, specification and source pin *before* calling the unchanged generic engine (`audit.dataset.d1_duplicate_detection.audit_duplicates` / `build_d1_certificate`), through a `D1_ENGINE` indirection. It hard-codes AutoND, so it cannot be reused unchanged.

**A separate Gohr D1 runner** (`audit/dataset/experiments/d1_gohr_canonical_census.py`), modelled on the AutoND one, does this. It:
1. load the manifest;
2. require the frozen Gohr specification;
3. verify the build attestation and generator pin, reusing `d0_gohr_attested_build`;
4. run `canonical.verify`;
5. load partitions through `canonical.load_partition` (never regenerate);
6. call the generic engine;
7. bind the certificate to the instance ID, manifest and attestation hashes;
8. write atomically, never overwriting.

Its fail-closed tests (`test_d1_gohr_canonical_census.py`) use an engine spy to prove that refusals never reach the engine, and block every generation entry point to prove the runner never regenerates data. The legacy D1 runner and evidence stay unchanged. The D1 gate of §5 decides whether D3 production may proceed.

## 10. Provenance and publication

**The absolute path in the decision block.** `verify_preregistration` resolves the preregistration path (`d3_distribution_statistics.py:2216`), and `evaluate_confirmatory_decision` records it in the decision block (line 2362). The certificate's `decision.preregistration.path` will therefore hold an absolute path containing the local home directory. The runner's own fields (`chain`, `code_identity`) use repository-relative paths and record no interpreter path.

**The D0 manifest.** The attested D0 build passes a relative repository root to `canonical.build()`, so the new manifest's `git.repo_root` is not an absolute home path.

**The sealed audit** (`d3_gohr_sealed_audit.json`) contains no path: the AutoND sealed audit was checked and holds none.

**Strategy:**
1. Sealed and production evidence is never edited in place. The originals stay the authoritative private record, with their hashes recorded in the run manifest and the git history.
2. For anonymous review, publish a **separate derived bundle**: the sealed audit unmodified (byte-identical, hash-verifiable), plus copies of the certificate and manifests with path and identity fields redacted. A redaction manifest lists every redacted field, the original file's SHA-256 and the derived file's SHA-256.
3. The original hashes, commits and approver identity commitment can be disclosed after review, so reviewers can check the derived bundle against the originals.

## 11. What requires the approver (not done, and must not be done by anyone else)

1. Done: items 3 and 3b of `data_examined_before_approval` are answered.
2. Approve the design choices: R1 confirmatory, the equal six-family split, m = 0.01 from II-4, and E exploratory.
3. Fill in `approved`, `approved_by`, `approved_utc` and the engine's exact `APPROVAL_STATEMENT`, read as in §2.
4. Commit the record with the runner, the calibration tool, the attestation tool and the tests, **before** the Gohr D0 instance is generated.
5. Then, as separate gates:
   - the attested D0 build;
   - the instance-bound Gohr D1 census, under the D1 gate of §5 (no D3 production unless D1 returns PASS);
   - the Gohr calibration run;
   - a read-only preflight (`--preflight-only`);
   - production.
