# AutoND CE2: theory consistency of the published AutoND SPECK32/64 round-8 DBitNet

**Status (2026-10-31):**
- design r2 `ef3d764a...`; preregistration `PENDING_REVIEW`;
- two design items remain `PROPOSED_PENDING_APPROVAL`: `runs.status` and `statistics.power.status`;
- no production run and no certificate;
- exploratory smoke runs only (disclosed in `POWER_ANALYSIS.md` section 6).

**Question.** Is the published model's score statistically associated with a Markov-model
single-characteristic probability proxy computed from each genuine pair's realised eight-round XOR
trajectory? The study is observational. The hypothesis is TWO-SIDED: no direction is predicted.

## Files (one place)

| | |
|---|---|
| `design.py` | the design (single source); r2 supersedes r1 with no scientific change |
| `preregistration/preregistration.json` | deterministic serialisation of the design plus the approval block |
| `theory.py` | xdp+ by Lipmaa-Moriai closed form, carry-pair automaton and literal enumeration |
| `adapter.py` | genuine pairs, trajectories, frozen run data, theory-consistency capability |
| `analysis.py` | Spearman, exact sign tests, Holm, decision rule |
| `runner.py` | preflight, behavioural binding, journalled run state machine, finalisation |
| `verify.py` | independent verifier (imports nothing from the producer) |
| `power.py` | scenario-based sensitivity analysis and pipeline-faithful check |
| `validation/r2/` | r2 check outputs: model binding, gates at production sizes, power, exploratory smoke summary |
| `history/r1/` | r1 design, preregistration, implementation and documents, byte-identical (`SHA256SUMS`) |
| `docs/` | this README, `RUNBOOK.md`, `POWER_ANALYSIS.md`, `VERIFICATION_REPORT.md` |
| `tests/` | `test_theory_statistics.py` (no TensorFlow), `test_pipeline.py` |

The CLI is `audit/cryptography/kaggle/autond_ce2_kaggle.py`. The target layer
(`audit/cryptography/targets/autond/`) holds the hash-verified, import-isolated upstream loader and the
strict checkpoint binding. The pinned upstream itself stays at `autond_ce1/AutoND-upstream/`, the only
copy.

## Design in brief

- **Data.** Each run draws 10^6 genuine pairs with the upstream `make_train_data` construction: a fresh
  64-bit master key per pair, shared by both members, and P1 = P0 XOR 0x400000. The source is
  `os.urandom`, so runs are not seed-replayable. Every raw array is retained and hashed.
- **Target.** T = -sum_r k_r, where xdp+ = 2^-k_r on the realised round transitions. Every factor
  is checked by two independent derivations (gate E).
- **Score.** The sigmoid output of the published checkpoint (SHA-256 `a7be788d...`), strictly bound:
  23 weight-bearing layers, 68 arrays bit-identical, 298,113 total parameters = 296,065 trainable
  + 2,048 non-trainable.
- **Controls, on the same frozen run dataset.**
  - **Permuted target:** a seeded permutation.
  - **Untrained twin:** one seeded, untrained DBitNet. A control result is conditional on this one
    twin, not a claim about untrained DBitNets in general.
- **Inference.** The unit is the run (20 runs, all required valid).
  1. An exact two-sided sign test on rho_primary.
  2. Only if step 1 rejects, a one-sided sign test on D_c = |rho_primary| - |rho_c| for each control,
     with Holm.
  - SUPPORTED needs both controls separated. A significant primary with an unresolved control is
    ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED. Anything else is INCONCLUSIVE or PRECHECK_FAILED, never
    evidence of zero association.
- **Undefined coefficients** (constant input, non-finite values) exclude the run with a recorded
  reason. They are never scored as 0. Within-run p-values are descriptive only, with explicit
  UNDERFLOW and DEGENERATE states.

## Behavioural binding (model binding, not learning evidence)

The binding set is a fresh balanced set: n/2 genuine pairs and n/2 uniform random 64-bit inputs (the
upstream negative class), shuffled by a fixed seed. It requires:
- a one-sided 95% Wilson lower bound on accuracy above 0.5;
- exact label balance;
- all scores finite;
- generation gates passed.

The untrained twin's accuracy on the same set is recorded as a diagnostic, not as a gate.

The design uses n = 10^6. **This has not been run.** The only behavioural results so far are smoke
runs at n = 20,000:
- r1: accuracy 0.519, lower bound 0.513;
- r2: accuracy 0.5079, lower bound 0.5021 (twin 0.4975).

These do not give the assurance of the planned 10^6 set. The r2 margin at 20,000 is thin. The binding
shows that the checkpoint and input pipeline behave as a distinguisher. It is not evidence about what the
model learned.

## Interpretation limits

- The study is association only.
- The target is a proxy: not the fixed-key characteristic probability, and not the differential
  probability of the cipher.
- The results are conditional on one published checkpoint and one twin.
- Only genuine pairs are evaluated.
- The results do not amend or explain AutoND CE1.
