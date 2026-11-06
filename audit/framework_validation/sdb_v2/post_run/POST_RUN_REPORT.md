# SDB-V2: post-run corrections and archival record

**Scope.** This report corrects the reporting of the registered SDB-V2 run. Nothing registered was rerun or modified.

**Preserved unchanged:**

| item | identifier |
|---|---|
| anchor commit | `39c25819240a0784c0bf37c686aecc19da2643de` |
| registration | `95fd1ce54a2d70c4…` |
| manifest | `09253359d51a6fb2…` |
| result | `results/20261106T171400Z/result.json` `db6f6502a2879cdc…` |
| independent verification | `dde53a09c3c765cd…` |
| H2 bundle manifest | `dd2245f3…` |

All evidence is unchanged.

## 1. B1 coverage, reconciled

B1 and B1-S are different measurements. They are counted separately.

| measurement | cases | n |
|---|---|---|
| **B1**, fresh-data inference with an existing model (released generator, `> 0.5`) | H1, S2, K3 | 3 |
| **B1-S**, rescoring of stored predictions (H2's 10 archived arrays against `test_y.npy`, `>= 0.5`); not model re-execution | H2 | 1 |
| B1 NOT_ASSESSED: the bundle reports an accuracy, but no existing model or predictions exist | S1, S3, S4, S5, S6, K2, K4, K5, K6, K7, K8 | 11 |
| B1 NOT_APPLICABLE: no reported accuracy (decision-rule and evidence-record controls) | P1, U1–U7 | 8 |

**Totals:**
- 3 + 1 + 11 + 8 = 23.
- **B1 denominators:**
  - B1 executed on **3 of the 14** cases where B1 applies (3 executed + 11 not assessed);
  - that is **3 of 23** cases overall;
  - H2 is not in either B1 denominator.
- **B1-S** covers **1 case**. Its 10 arrays all reproduce the certificate's correct counts and class counts exactly.

**B1 values** (n = 200,000 each; descriptive; no decision threshold):

| case | accuracy | Δ from 0.9291 | 95% Wilson interval |
|---|---|---|---|
| K3 | 0.9290 | −0.0001 | [0.9279, 0.9302] |
| S2 | 0.5001 | −0.4290 | [0.4979, 0.5023] |
| H1 | 0.4159 | −0.5132 | [0.4137, 0.4181] |

## 2. H1 = 0.4159 versus K3 = 0.9290: investigation

**Method.**
- Diagnostic script `post_run/h1_diagnostics.py`. Raw values are in `h1_diagnostics.json` (`1ff981bb…`) and `h1_diagnostics_extra.json` (`d3a8161d…`).
- n = 100,000 per check, from fresh `os.urandom` draws, so the values differ from B1's draw at the third decimal.
- K3 is the control throughout.
- **No cause is assumed.**

| check | H1 (`de630afc…`) | K3 (`256eb4a5…`) |
|---|---|---|
| Keras-loaded weights vs raw HDF5 datasets | all tensors identical (0 unmatched) | all identical |
| architecture | 11 Conv1D, 5 Add; 70,177 weights | 21 Conv1D, 10 Add; 102,497 |
| HDF5 attributes | saved by Keras 3.13.2; loss mse; Adam lr 1e-4 | same |
| accuracy at rounds 3 / 4 / 5 / 6 / 7 | 0.745 / 0.536 / 0.419 / 0.515 / 0.612 | 0.474 / 0.476 / 0.928 / 0.497 / 0.493 |
| round 5: accuracy on label 1 / label 0 | 0.165 / 0.672 | 0.904 / 0.952 |
| round 5: inverted-label accuracy | 0.581 | 0.072 |
| round 5: AUC | 0.346 | 0.975 |
| round 5: mean prediction; fraction > 0.5 | 0.433; 0.247 | 0.498; 0.476 |
| best of 24 word orders × bit reversal (48 encodings) | 0.542 (order 1032) | 0.930 (identity is effectively best) |
| transposed 16×4 bit layout | 0.500 | 0.503 |

**Companion files beside H1:**
- `hist5r_depth10 (10).p`, read with an unpickler that refuses every class: 200 epochs, `val_acc` 0.9183 → max 0.9291 → last 0.9290; `val_loss` last 0.0543.
- `h5r_depth10 (10).npy`: a `val_loss` series. Gohr's `train_nets.py` saves `val_acc` and then overwrites the same file with `val_loss` (l.80–81).

**What the checks establish:**
- **Model loading:** not a loading defect. The loaded weights equal the stored ones, and the same loading path gives 0.929 for K3.
- **Generator and encoding:** the generator is the one the CE pipeline uses (`make_train_data` is identical in `speck.py` and `audit/cryptography/gohr/speck.py`). None of the 50 tested encodings recovers accuracy above 0.55.
- **Round count:** no round from 3 to 7 gives the published-level accuracy.
- **Labels:** inverting them gives 0.581, not about 0.93.
- **The checkpoint and its history disagree:** the training history stored beside the checkpoint records a run reaching 0.929. The checkpoint itself scores 0.419 on fresh 5-round data, with AUC below 0.5. This file is the CE pipeline's depth-5 checkpoint, the one CE2–CE4 loaded (implementation certificate V5, `downstream`).

**Not established:**
- why the checkpoint behaves this way;
- whether the companion history belongs to this checkpoint;
- whether the round-7 value (0.612) relates to the recorded CE1 baseline score (0.6108). The two are numerically close, and nothing more is claimed.

**Effect on H1.** None on the registered result. H1 is scored on the architecture probe and the binding refusal, and both matched. The B1 value is reported descriptively and is not interpreted as a cause.

## 3. Corrected reading of "7/7 detected"

The registered C-level output was reached for all 7 defect inputs. Their character differs, so they are reported separately:

| case | kind | what the match shows |
|---|---|---|
| H1 | **historical** defect, on the real historical checkpoint | the architecture probe reads depth 5 (11 Conv1D, 5 Add) against the published depth 10, and the binding gate refuses it |
| S1 | synthetic detector case (II) | the M1 vector, key-schedule and round-trace checks all fail on the mutated cipher (α = 8); the fixture's roll-up and precedence follow |
| S2 | synthetic detector case (binding), using an existing file | the hash gate refuses the substituted `signal_destroyed.h5` |
| S3 | synthetic detector case (DI) | D1 FAIL on 200 copied rows; the fixture's roll-up and precedence follow |
| S6 | synthetic detector case (EV) | the frozen EV chain reports an effect beyond ε; the fixture's roll-up and precedence follow |
| S5 | **detector-only conformance** | the progression rule gives best_round 5 and stop_round 6, and the criterion is not met. No DI or claim outcome is registered (gap G3) |
| S4 | **reduced-scale algorithm conformance** | D2-N rank-p and gates give EXCESS on a 20,000-row fixture with 59 null draws. It is not D2-N evidence and not confirmatory DI evidence |

**These are conformance results on designed inputs, not a detection rate.**
- The 0/7 false alarms are 7 clean component controls.
- The 7/7 abstention matches are 7 constructed controls: 3 run real components, 1 reads a real certificate, and 3 test decision rules only.

## 4. Corrected statement about historical evidence

The earlier statement that no case contains real historical evidence was wrong.

**Historical evidence:**
- **H1** runs real components on a real historical checkpoint.
- **H2** replays real archived A-D4 evidence: the certificate, 330 epoch records and 10 prediction arrays.

**Real artifacts used as inputs to constructed scenarios:**
- S2 and K3 use existing checkpoints;
- U2 reads a real A3 certificate field, and P1 a real E6 certificate field;
- K7 and U4 are runs of the frozen PAR1 scheduler with its test adapter.

**The accurate statement:**
- No case produces new dimension-level or claim-level evidence about Gohr or AutoND.
- H1 reproduces a known historical finding on the real artifact.
- H2 reproduces recorded historical values from raw evidence.
- The other 21 cases are constructed.

## 5. Limitations, reworded

1. **Historical replay vs constructed cases.**
   - H1 and H2 use historical evidence, and H2 is scored only as replay MATCH/MISMATCH.
   - The remaining 21 cases are constructed with knowledge of the detectors.
   - None of the counts estimates a population detection or false-alarm rate.
2. **What the independent verifier recomputes from inputs, with its own code:**
   - architecture from the HDF5 files;
   - the hash binding gate;
   - SPECK test vector, key schedule and round traces, using its own SPECK implementation and the α value parsed from each cipher file (the file is not executed);
   - D1 duplicates and overlaps;
   - progression and the co-primary criterion from the epoch records (S5, K8, H2);
   - EV paired t, CI and TOST;
   - D4 binomial band and baseline tests;
   - claim precedence;
   - B1-S counts from the arrays;
   - checklist items C1, C3, C5, C6, C7 and C8;
   - every aggregate.
3. **What the verifier takes from the run's recorded raw outputs:**
   - **D2-N:** the per-scan hit counts. Query sampling is not replicated and the 59 null scans are not regenerated. It does check the observed dataset's exact neighbour count for consistency.
   - **K3:** the behavioural-binding accuracies.
   - **K7 and U4:** the PAR1 verifier report. Only the analysis-hash binding is rechecked.
   - **B1:** the per-example correctness vectors. The models are not re-run, and the test inputs are stored only as hashes, because the released generator draws from `os.urandom`.
4. **Checks not re-run or not implemented:**
   - checklist item C2 (loadability) is not re-run by the verifier;
   - the boundary cases planned at the D4 and EV confidence-interval margins were not implemented;
   - the verifier's precedence covers COMPLETE reproduction and levels L0–L3, which spans every registered case.
5. **Undecided rules.**
   - X1–X4 are left UNSPECIFIED. X4 was identified during implementation and is disclosed in the registration.
   - Conflict and invalid-evidence handling have no rule: U8 is excluded, and U4 and S5 are detector-level only.
6. **Baseline limits.**
   - B1 is descriptive and covers 3 of 14 applicable cases.
   - B1-naïve is illustrative only.
   - The checklist's C3 item is uninformative on constructed bundles, and C7 is trivially satisfied wherever the constructed bundle publishes its own hashes.

## 6. Artifacts

**Local, uncommitted** (under `audit/framework_validation/sdb_v2/post_run/`):

| file | SHA-256 |
|---|---|
| `POST_RUN_REPORT.md` | this file |
| `h1_diagnostics.py`, `h1_diagnostics.json`, `h1_diagnostics_extra.json` | `1ff981bb…`, `d3a8161d…` (JSON files) |
| `restore_h2_predictions.py` | `1185f4b5…` (archive copy) |
| `clean_checkout_verification.json` | `dde53a09…`, byte-identical to the original verification |
| `archives/sdb_v2_h2_predictions_supplement.tar.gz` | `5292d04e…`, the 11 `.npy` files at their registered paths |
| `archives/sdb_v2_results_20261106T171400Z.tar.gz` | `0ef30859…`, result.json, independent_verification.json and 30 raw records |
| `archives/SHA256SUMS`, `archives/CONTENTS.json` | per-file hashes |

**Clean-checkout procedure** (performed 2026-11-06, verifier VERIFIED, 0 problems):

```
git clone --branch audit-framework https://github.com/Shamayita777/deep_speck.git ds && cd ds
git checkout --detach 39c25819240a0784c0bf37c686aecc19da2643de
python3 -I <archives>/restore_h2_predictions.py --archive <archives>/sdb_v2_h2_predictions_supplement.tar.gz \
    --archive-sha256 5292d04eb0f63f66a8ba348f6388787f07c3c3bb565ceab88edd39e1a207c18c --repo-root .
tar -xzf <archives>/sdb_v2_results_20261106T171400Z.tar.gz
<python 3.13 with numpy, scipy, h5py> -I audit/framework_validation/sdb_v2/verify_sdb_v2.py \
    --result audit/framework_validation/sdb_v2/results/20261106T171400Z/result.json --repo-root .
```

**Publication of the 11 arrays.** A commit adding them at their exact registered paths is prepared but **not pushed** (see the report to the operator). Pushing it, or depositing the supplementary archive elsewhere, awaits review.
