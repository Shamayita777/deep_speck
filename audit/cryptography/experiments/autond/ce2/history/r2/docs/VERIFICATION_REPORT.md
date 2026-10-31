# AutoND CE2: verification report (design r2, pre-production)

Date: 2026-10-31.
- **Covered:** the r2 implementation, before any production run.
- **Design:** `ef3d764a8350e01a68fdcacf2fb38a15fd5622bfc8665f20e0e71df3f2ba4ba5`.
- **Preregistration:** `PENDING_REVIEW`. No confirmatory CE2 run has been started.

The r1 report (design `7245d152...`) is kept unchanged at `history/r1/docs/VERIFICATION_REPORT.md`.

## 1. What changed from r1, and why

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
| a preregistration value | `preregistration_design_hash` |
| preflight content | `preflight_hash_matches_manifest` |
| an unlisted file in a run directory | `every_run_file_committed_cached_or_quarantined` |
| measurement content | `journal_replay_state_machine_and_commits` |

**Forgery.** A forgery whose hashes, journal chain, record caches, measurement and certificate are all
made consistent passes every hash and replay check. It is rejected only by score recomputation, which
gives `DECISION_RELEVANT_DIFFERENCE`.

The approval check is in `test_production_initialisation_refused_while_pending`. The production-only
approval checks in the verifier run only on production directories, and none exists.

## 5. Results of the r2 checks (2026-10-31, AutoND environment, see section 7)

| Check | Result |
|---|---|
| `--check-model` (`validation/r2/check_model.json`) | strict load passes: 23 groups, 68 arrays bit-identical; architecture as designed; recorded runtime deviations are pandas 2.2.3 and SciPy 1.17.1 |
| `--validate-math` (`validation/r2/validate_math.json`, 291 s) | C and D exhaustive at n = 4/6/8 with 0 mismatches; 8 sixteen-bit literal transitions agree; E, F and the encoding check pass |
| `--power` (`validation/r2/power.json`, 33 s) | see `POWER_ANALYSIS.md` |
| exploratory smoke run (3 x 4,096; `validation/r2/exploratory_smoke_summary.json`) | INCONCLUSIVE, as designed. Verifier: 77/77 checks; with `--preflight-word8`, 78/78 including the n = 8 re-enumeration; numerical status EXACT; inference from recomputed scores. Behavioural binding at n = 20,000: accuracy 0.5079, Wilson lower bound 0.5021 |

The full 10^6 behavioural binding has not been run. The production preflight in the pinned
environment has not been run.

## 6. Tests

See the repair report for the full counts and commands. The suites relevant to CE2 are:
- `experiments/autond/ce2/tests/test_theory_statistics.py` (no TensorFlow);
- `experiments/autond/ce2/tests/test_pipeline.py` (TensorFlow);
- `core/tests/`.

## 7. Environment of these checks

The local pinned venv has Python 3.11.14, TF 2.15.1, Keras 2.15.0, NumPy 1.26.4 and h5py 3.10.0,
with three deviations: pandas 2.2.3, SciPy 1.17.1 and pytest 9.1.1. It ran on CPU (macOS arm64).

This is not the production environment. Production enforces the pinned versions and runs on Kaggle.
