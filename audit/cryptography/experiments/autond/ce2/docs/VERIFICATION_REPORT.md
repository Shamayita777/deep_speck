# AutoND CE2: verification report (design r3, pre-production)

Date: 2026-10-31.
- **Covered:** the r3 implementation, before any production run.
- **Design:** `3f6340e8625749d1a1281ea44984ea4ee48521e00c6c774598398680b8a0635f` (r3, final; the
  unapproved r3 draft 1 `7e69ce6c...` is archived in `history/r3-draft1/`).
- **Preregistration:** `PENDING_REVIEW`. No confirmatory CE2 run has been started.

The r2 report (design `ef3d764a...`) is kept unchanged at `history/r2/docs/VERIFICATION_REPORT.md`, and
the r1 report (design `7245d152...`) at `history/r1/docs/VERIFICATION_REPORT.md`. Which checks were re-run
under r3 and which are reused from r2 is listed in `validation/r3/VALIDATION_RECORD.md`.

## 0. What the verifier establishes about the primary inference

The verifier recomputes the model scores, then recomputes every run's three coefficients (primary,
permuted target, twin) from them with its own ranks. It derives the primary sign-test decision from
those recomputed coefficients, not from stored scores or recorded values:
- its own exact two-sided sign test (`sign_two_sided`, rational arithmetic) counts positive and
  negative coefficients and removes exact zeros;
- it compares the positive and negative counts and the p-value with the certificate, and (new in r3)
  checks that the certificate's `n_effective` = positive + negative and `n_zero_ties_removed` =
  `n_valid` - `n_effective`, for the primary test and for each control test;
- with no nonzero coefficient it derives `NO_EFFECTIVE_OBSERVATIONS` itself (p = None, never 1.0),
  requires the certificate to report the same status and a null p-value
  (`primary_testability_status`, `control_*_testability_status`), and requires an INCONCLUSIVE decision
  (primary) or a non-rejected control that keeps the Holm family at two;
- it parses the preregistration with a duplicate-key-rejecting JSON parser and refuses to verify
  (`VerificationError`) if any key repeats;
- an undefined coefficient is `None` in the verifier, never 0.0: an EXCLUDED run must have one
  (`exclusion_justified`), and a VALID run with one fails (`rho_defined_on_recomputed_scores`);
- the decision, including Holm and serial gatekeeping, is recomputed and must equal the certificate.

So the verifier checks the test of p_+ = Pr(rho_i > 0 | rho_i != 0, run i is valid) end to end from
recomputed data. It does not, and cannot, check that the hypothesis was the right one to ask.

## 1a. What changed from r2 to r3

| r2 | r3 | Tests |
|---|---|---|
| The null was written as full-distribution symmetry of the run-level population Spearman coefficient; the estimand was that coefficient | The null and alternative are p_+ = 1/2 and p_+ != 1/2; p_+ is the primary estimand; rho_i is the per-run statistic and its distribution is descriptive. The sign-test code is unchanged | `tests/test_r3_specification.py` (hypothesis consistency; no obsolete wording in the design or docs) |
| Zero and undefined coefficients were distinguished in code but not stated side by side | The design states both policies: a defined zero is a VALID run and a tie; an undefined coefficient makes the run EXCLUDED | six zero-handling cases plus the full procedure, against both the analysis and the verifier |
| The certificate's zero counts were not checked | `primary_zero_ties_and_n_effective` and `control_*_zero_ties_and_n_effective` | tamper case `certificate_zero_ties` |
| Power assumed no zeros, without saying so | Assumption stated; exact power by n_effective and a zero-run scenario added; zero counts in the pipeline-faithful check | `test_zero_tie_sensitivity_*`, `test_pipeline_faithful_check_counts_exact_zeros` |
| `clopper_pearson` overflowed for about 1,030 < n <= 2,000 (not reached by any r2 result, which used n = 50,000) | log-space terms | `test_clopper_pearson_does_not_overflow_for_moderate_n` |
| two `PROPOSED_PENDING_APPROVAL` markers | both `FROZEN`, values unchanged | `test_markers_frozen_and_preregistration_matches_design` |
| a sign test with n_effective = 0 returned p = 1.0 (producer and verifier) | status `NO_EFFECTIVE_OBSERVATIONS`, p null, never rejects; decisions unchanged | `test_all_zero_coefficients_are_not_testable_*`, `test_untestable_control_*`, `test_producer_and_verifier_agree_with_zeros_and_untestable_tests` (300 random cases); tamper case `certificate_not_testable` |
| `json.loads` silently kept the last of duplicate keys | strict parse in `design.load_preregistration` and in the verifier | `test_preregistration_parses_strictly_*`, `test_verifier_refuses_a_preregistration_with_duplicate_keys` |
| power intervals were described as Clopper-Pearson | the docstring and documents name both branches; every row records `ci95_method` | `test_interval_methods_are_described_for_both_branches`, `test_power_evidence_has_control_zero_scenarios_and_records_interval_methods` |
| zero differences in the control tests not in the power analysis | fixed-count D_perm, D_twin and joint scenarios | `test_control_zero_difference_scenarios` |

## 1. What changed from r1 to r2, and why

| r1 defect | r2 behaviour | Tests |
|---|---|---|
| `data.npz` written, then the record advanced. A crash between the two left NOT_STARTED with a file in the way, and `save_npz` refused to overwrite. | Every artifact is committed by intent, staged write, hash, validate, atomic link and then transition. On resume, uncommitted files are quarantined (retained, hashed, journalled) and the step runs again. Committed damage makes the run FAILED. | `core/tests/test_evidence_lifecycle.py` (all 7 fault points); `tests/test_pipeline.py::test_runner_recovers_from_interruption_at_every_boundary` (all 7, through the real runner) |
| The record was written before the journal, so a crash could leave a record transition with no event. The verifier checked only the hash chain. | The journal is authoritative. The record is a cache materialised from it, rebuilt when stale or absent (and that is journalled), and InconsistentEvidence otherwise. The verifier replays the journal against its own state machine and requires `record.json` to equal the replay. | the evidence tests listed in section 4 |
| The verifier compared recomputed scores within 1e-4, then computed rho from the *stored* scores. | The verifier computes every rho and the decision from *its own* recomputed scores. It reports bitwise equality, maximum difference, rank-vector and tie-structure equality, and fails if any decision-relevant sign or the decision differs. | `test_forged_scores_with_consistent_hashes_are_caught_by_recomputation`; `test_small_rank_perturbation_is_reported_but_decision_invariant` |
| `model_input_sha256`, `published_weights_sha256` and `twin_weights_sha256` were recorded but never re-derived. | Each is re-derived independently: own bit encoding and content hash, own HDF5 strict comparison, own weight hash. | `run_XX:model_input_sha256` and `*_weights_sha256` checks; `test_weight_hash_ignores_variable_names_but_not_values` |
| The preflight hash was checked, not its content. | Gates C and D are re-executed by the verifier: transfer matrix against literal enumeration at n = 4 and 6, and n = 8 with `--preflight-word8`. The recorded counts are compared. The 16-bit rows are re-checked with the transfer matrix. | `preflight_gate_CD_word{4,6,8}_independently_recomputed`; tamper case `preflight` |
| The legacy framework test encodes a positive hypothesis and maps undefined to 0.0. | `core.theory_consistency` with an explicit `TWO_SIDED`, recorded in the manifest, measurements and certificate. Undefined gives `statistic = None` with a status. | `test_measurement_uses_one_frozen_dataset_and_is_two_sided`, `test_undefined_twin_is_reported_undefined_not_zero`, `test_undefined_is_distinguished_from_zero` |
| `within_run_p_value` returned 0.0 for abs(rho) >= 1 and could underflow to 0.0. | `PValue` records with EXACT, APPROXIMATE, UNDERFLOW (log10 kept), DEGENERATE and UNDEFINED. | `test_within_run_p_value_underflow_and_degenerate_are_not_zero`; `core/tests/test_core_semantics.py` |
| The upstream root stayed on `sys.path`, and its modules (`main`, `optimizer`, ...) stayed in `sys.modules`. | Isolated import: colliding entries are set aside and restored, and the root and the modules are removed afterwards. Identity is stable through a cache. | `test_upstream_import_is_isolated_and_identity_stable`, `test_upstream_import_does_not_clobber_or_use_a_foreign_module` |
| Power reported point estimates under one model. | Monte Carlo SE and 95% intervals, a wider dependence grid, and a pipeline-faithful check. Stated as a sensitivity analysis. | `test_joint_power_*`, `test_clopper_pearson_small_n_is_exact`, `test_pipeline_faithful_check_*` |

## 2. Scope of each mathematical gate (what it proves and what it does not)

| Gate | Kind | Coverage (r2 run, `validation/r2/validate_math.json`) | Code paths compared | Independently re-executed by the verifier |
|---|---|---|---|---|
| A, B | definitions | n/a | closed form (A) and carry-pair automaton (B); B shares no code with A (test) | n/a |
| C | exhaustive identity check | n = 4, 6, 8: all 4,096, 262,144 and 16,777,216 triples; 1,372, 67,228 and 3,294,172 possible (= 4·7^(n-1)); 0 mismatches | A vs B | yes. A third derivation (transfer matrix) is checked against literal enumeration at n = 4 and 6 (and 8 with the flag), and the recorded counts are compared |
| D (n <= 8) | exhaustive literal enumeration | every (alpha, beta) with the full gamma histogram at n = 4, 6, 8; 0 mismatches | literal vs A vs B | yes, as for C |
| D (n = 16) | selected literal enumeration | 8 transitions of one corpus trajectory, each over all 2^32 operand pairs | literal vs A vs B | partly. The recorded rows are re-checked by the transfer matrix; the 2^32 enumeration is not re-run |
| E | per-sample consistency | every round of every sample (production: every run; preflight: 4,096 pairs) | A vs B, every factor > 0 | yes, per run. The verifier recomputes every factor with the transfer matrix and the target from them |
| F | per-sample reconstruction | final state = upstream `encrypt`; initial difference; addition output difference = next dx; dy linear layer; model input = upstream `hstack` | upstream `enc_one_round` / `expand_key` vs upstream `encrypt` | yes, per run. Own SPECK round function and key schedule, plus upstream `encrypt` on the stored keys and plaintexts |
| Encoding | source-level construction check | 2,048 rows from one PCG64 byte stream substituted for `os.urandom`; the 1,027 genuine rows are identical to upstream `make_train_data` | CE2 generator vs upstream function | recorded only |

**What the gates establish.**
- The three xdp+ derivations (closed form, automaton, transfer matrix) and literal enumeration agree on
  every input for n <= 8.
- They agree on the 16-bit transitions checked literally.
- Every production factor is computed identically by two derivations, and by a third in the verifier.
- The reconstructed trajectories reproduce the upstream cipher bit for bit.

**What they do not establish.**
- Exhaustive correctness at n = 16. This rests on the closed form's proof (Lipmaa-Moriai 2001) and on
  agreement with two other derivations, not on enumeration.
- That the Markov proxy approximates any fixed-key probability. It does not, and claims nothing of the
  kind.
- Anything about what the model learned.

**Shared code.**
- The production E/F path and the verifier both call upstream `encrypt` on the same inputs. The
  verifier also uses its own round function.
- Model scores are recomputed with the same upstream DBitNet builder and the same TensorFlow. This
  shows that inference is reproducible. It is not an independent implementation of the network.

## 3. Numerical reproducibility policy

The confirmatory tests use only sign(rho_primary,i) and sign(|rho_primary,i| - |rho_c,i|). The
verifier:
1. recomputes the scores;
2. reports per run and model whether they are bitwise equal, the maximum absolute difference, whether
   the average-rank vectors are equal, and whether the tie structures are equal;
3. computes rho and the decision from the recomputed scores.

It reports `EXACT`, `DECISION_INVARIANT_NUMERICAL_DIFFERENCE` (pass, reported) or
`DECISION_RELEVANT_DIFFERENCE` (fail). No elementwise tolerance can turn a rank change into a pass
silently: a rank change is always reported, and it fails whenever it reaches a decision-relevant sign.

**Platform.** Production runs on a Kaggle GPU. Verification on a different device can produce last-bit
differences, which this policy reports rather than hides. Where possible, verify on the same hardware
class and record both environments.

## 4. Evidence-integrity tests (verifier and store)

**Journal and record, in `core/tests/test_evidence_lifecycle.py`:**
- record updated with no journal event;
- journal updated with the record absent (rebuilt and journalled);
- duplicate, missing or reordered event;
- modified event payload;
- modified record state;
- wrong previous hash;
- wrong sequence number;
- wrong experiment id;
- a transition starting from the wrong state;
- leaving a terminal state;
- an illegal transition;
- a torn journal tail (set aside, not deleted);
- committed-artifact damage.

**Verifier tamper matrix, in `tests/test_pipeline.py::test_verifier_rejects_tampering`.** Each case
must fail with the named check.

| Tampered | Rejected by |
|---|---|
| data bytes | `journal_replay_state_machine_and_commits` |
| a record field | `record_caches_equal_journal_replay` |
| a journal line | `journal_chain_sequence_and_identity` |
| the journal re-chained with a changed state | `record_caches_equal_journal_replay` |
| stored scores | `journal_replay_state_machine_and_commits` |
| the run manifest | `run_manifest_hash_matches_initialised_event` |
| the twin identity | `untrained_twin_weights_hash` |
| the certificate decision | `decision_recomputed_from_recomputed_scores` |
| the certificate's zero-tie count and n_effective (r3) | `primary_zero_ties_and_n_effective` |
| the certificate's primary status set to NO_EFFECTIVE_OBSERVATIONS with p null (r3) | `primary_testability_status` |
| a preregistration value | `preregistration_design_hash` |
| preflight content | `preflight_hash_matches_manifest` |
| an unlisted file in a run directory | `every_run_file_committed_cached_or_quarantined` |
| measurement content | `journal_replay_state_machine_and_commits` |

**Forgery.** A forgery whose hashes, journal chain, record caches, measurement and certificate are all
made consistent passes every hash and replay check. It is rejected only by score recomputation, which
gives `DECISION_RELEVANT_DIFFERENCE`.

The approval check is in `test_production_initialisation_refused_while_pending`. The production-only
approval checks in the verifier run only on production directories, and none exists.

## 5. Results of the r2 checks (2026-10-31, AutoND environment, see section 7; reused, not re-run under r3)

| Check | Result |
|---|---|
| `--check-model` (`validation/r2/check_model.json`) | strict load passes: 23 groups, 68 arrays bit-identical; architecture as designed; recorded runtime deviations are pandas 2.2.3 and SciPy 1.17.1 |
| `--validate-math` (`validation/r2/validate_math.json`, 291 s) | C and D exhaustive at n = 4/6/8 with 0 mismatches; 8 sixteen-bit literal transitions agree; E, F and the encoding check pass |
| `--power` (`validation/r2/power.json`, 33 s) | see `POWER_ANALYSIS.md`; superseded by `validation/r3/power_final.json` (52 s, final r3; draft-1 output `validation/r3/power.json` kept), whose seeded tables reproduce r2 exactly |
| exploratory smoke run (3 x 4,096; `validation/r2/exploratory_smoke_summary.json`) | INCONCLUSIVE, as designed. Verifier: 77/77 checks; with `--preflight-word8`, 78/78 including the n = 8 re-enumeration; numerical status EXACT; inference from recomputed scores. Behavioural binding at n = 20,000: accuracy 0.5079, Wilson lower bound 0.5021 |

The r2 smoke verification (77/77, 78/78) predates the r3 zero-count and testability checks; no smoke run was made
under r3. The test suite's own small smoke pipeline exercises them (`test_pipeline.py`).

The full 10^6 behavioural binding has not been run. The production preflight in the pinned
environment has not been run.

## 6. Tests

See the repair report for the full counts and commands. The suites relevant to CE2 are:
- `experiments/autond/ce2/tests/test_theory_statistics.py` (no TensorFlow);
- `experiments/autond/ce2/tests/test_r3_specification.py` (no TensorFlow);
- `experiments/autond/ce2/tests/test_pipeline.py` (TensorFlow);
- `core/tests/`.

## 7. Environment of these checks

The local pinned venv has Python 3.11.14, TF 2.15.1, Keras 2.15.0, NumPy 1.26.4 and h5py 3.10.0,
with three deviations: pandas 2.2.3, SciPy 1.17.1 and pytest 9.1.1. It ran on CPU (macOS arm64).

This is not the production environment. Production enforces the pinned versions and runs on Kaggle.
