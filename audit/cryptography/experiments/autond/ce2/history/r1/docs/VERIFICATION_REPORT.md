# AutoND CE2 — verification report

Date: 2026-10-30. This report covers the pre-production state only. No confirmatory CE2 run has been
started, and the preregistration is `PENDING_REVIEW`.

## Baseline and isolation

- **Repository at baseline:** branch `audit-framework`, HEAD `fb539bc83d605cf1200fb2eed3640bc2dcbed9d2`.
  The working tree was not clean beforehand; those modifications and untracked files are unchanged.
- **Comparison method:** all 1,769 files under `audit/` were hashed before implementation and again
  after.
- **Result:** no pre-existing file was modified, renamed or removed. The groups checked were:
  - AutoND CE1 code, upstream copy and evidence (35 files, including the CE1 certificate
    `d2ca9a95…ec5e`);
  - Gohr `ce234/` (19);
  - the shared framework (`adapters/base.py`, `test/base.py`, `test/ce2/theory_consistency.py`,
    `results.py`, `statistics.py`);
  - `experiments/ce2/design.py` and `experiments/ce2/gohr_theory_consistency.py`;
  - root-level files.
- **Tracked changes:** `git diff` of tracked files is byte-identical to the baseline. `git status`
  differs only by the new files listed below. Nothing is staged or committed.
- **One incident, corrected:**
  - **What happened:** an exploratory model-loading script was run without `-B`. Python wrote one
    bytecode cache, `autond_ce1/AutoND-upstream/__pycache__/dbitnet.cpython-311.pyc`, created at
    2026-10-30 19:32 and not present at baseline.
  - **Correction:** that file and its empty directory were removed, so the CE1 directory matches the
    baseline again.
  - **Prevention:** upstream imports in the adapter and the verifier now set
    `sys.dont_write_bytecode` while importing. A test confirms that a fresh process with bytecode
    writing enabled leaves the upstream tree unchanged.
- **CE1:** nothing ran against CE1: no training, resume, initialisation, smoke run or finalisation.

## Stage 1 — read-only model binding (`validation/model_binding_check.json`)

The check made no `fit` call, saved no weights, retained no data and wrote nothing to CE1.

| Item | Observed |
|---|---|
| checkpoint SHA-256 | `a7be788dbbb760a7be9053748e6e0f37dedc5768db3a74a5c3ebd294dca4cc26` (verified before loading) |
| HDF5 | 34 layers, 23 with weights, 68 weight arrays; saved by Keras 2.7.0 |
| strict load | every array of every weight-bearing layer bit-identical to its HDF5 dataset, in order; no unused dataset |
| architecture | 34 layers, 23 weight-bearing; input (None, 64, 1); output (None, 1) |
| parameters | 298,113 total = 296,065 trainable + 2,048 non-trainable (BatchNormalization moving statistics) |
| inference | finite; `predict` and `call(training=False)` agree to 6.0 × 10⁻⁸ |
| upstream | all 8 files in `upstream_sha256.json` verified; every imported module resolves inside the pinned root |

The handoff figure of 298,113 is the **total** parameter count, not the trainable count. The design
records all three numbers as observed.

## Stage 2 — mathematical gates (`validation/mathematical_verification.json`)

| Gate | Coverage | Checked | Mismatches |
|---|---|---|---|
| C, n = 4 | exhaustive (α, β, γ), closed form vs automaton | 4,096 (1,372 possible) | 0 |
| C, n = 6 | exhaustive | 262,144 (67,228 possible) | 0 |
| C, n = 8 | exhaustive | 16,777,216 (3,294,172 possible) | 0 |
| D, n = 4 / 6 / 8 | exhaustive literal operand enumeration for every (α, β), full γ histogram, vs both | 4,096 / 262,144 / 16,777,216 | 0 |
| D, n = 16 | 8 distinct transitions of the first corpus trajectory, each by literal enumeration of all 2³² operand pairs | 8 | 0 |
| E | 4,096-pair corpus, every round: closed form = automaton, all factors > 0 | 32,768 factors | 0 |
| F | final state = upstream `encrypt`; initial difference; addition output difference = next Δx; Δyᵣ = ROL₂(Δyᵣ₋₁) ⊕ Δxᵣ; model input = upstream `hstack` | 4,096 pairs | 0 |

The counts of possible transitions equal 4·7ⁿ⁻¹, the known number of possible XOR differentials of
addition, which is an independent consistency check.

The 16-bit literal check covers selected transitions only. Every distinct transition of the corpus is
covered by closed form against automaton (gate E).

**Encoding against upstream.** With the same byte stream substituted for `os.urandom`, the CE2
genuine-pair inputs are identical to the `Y = 1` rows of upstream `main.make_train_data` (1,027 rows
of 2,048).

The upstream generator was confirmed to use:
- an independent 64-bit master key per sample, shared by both members of the pair;
- an independent 32-bit plaintext per sample, with P₁ = P₀ ⊕ Δ;
- labels with 1 = genuine and 0 = 64 uniform random bits;
- a model input of [C0 x, C0 y, C1 x, C1 y], most significant bit first.

## Stage 3/4 — dataset, power, computation

- **Data:** CE2 data come from `os.urandom` with the upstream construction. They are not
  seed-replayable. The raw evidence of every run is retained: keys, both plaintexts, both
  ciphertexts, per-round differences, factors, target, scores and permutation.
- **Power and computation:** see `POWER_ANALYSIS.md`. The recommendation is 20 runs × 10⁶ pairs with
  all 20 required valid, giving at least 90% joint power for |ρ| ≥ 0.0020–0.0028.
- **Disclosure:** coefficients from two exploratory pipeline checks were observed and are disclosed
  there. They were not used in the recommendation.

## Stage 6 — tests and pipeline

- **Tests:** `tests/test_autond_ce2.py` has **42 passed** (pinned TF 2.15.1, Keras 2.15.0,
  NumPy 1.26.4, Python 3.11.14). The 252 warnings are `DeprecationWarning`s raised inside upstream
  AutoND code imported by `main`. The suite covers:
  - exact arithmetic and the three independent xdp⁺ paths;
  - trajectory and encoding;
  - definedness and the undefined-is-not-zero rule;
  - exact tests, Holm, the order-statistic interval and every decision path;
  - preregistration and approval gating;
  - framework calls on one frozen dataset;
  - adapter scope;
  - an end-to-end pipeline with independent verification and tamper detection;
  - resume without regeneration and terminal-record immutability;
  - output-directory rules;
  - CE1, Gohr and verifier import isolation;
  - no bytecode written into the upstream copy.
- **CPU smoke** (3 runs × 4,096 pairs, `SMOKE_NOT_CONFIRMATORY`, scratch directory outside the
  repository):
  - preflight passed;
  - the behavioural check reached accuracy 0.519 with Wilson lower bound 0.513 on 20,000 samples;
  - all runs were VALID with 3 framework test runs and 15 adapter calls each, every call seeing a
    single dataset hash;
  - the decision was `INCONCLUSIVE`, as expected for 3 runs;
  - the independent verifier passed all 59 checks, including recomputed scores.
- **Runtime deviations of the local machine:** SciPy 1.17.1 (pinned 1.11.4), pandas 2.2.3 (2.2.2)
  and pytest 9.1.1 (8.3.3). The production runtime gate enforces the pinned versions, so production
  must run in the pinned environment of `RUNBOOK.md`.

## Not yet done

- No CE2 production preflight in the pinned environment, no production run, no certificate.
- The design is frozen only after review: `runs.status` and `statistics.power.status` remain
  `PROPOSED_PENDING_APPROVAL`.
- Verifier limits:
  - The secondary descriptive coefficients and the run-level bootstrap sensitivity interval are not
    recomputed by the verifier.
  - Model scores are compared with an absolute tolerance of 1 × 10⁻⁴, because GPU and CPU
    arithmetic may differ.

## New files (SHA-256 at the time of this report)

| File | SHA-256 |
|---|---|
| `adapters/autond_ce2.py` | `42a6e4713b4a4be7e4dbd2f2be10b10302e0744035e649f658e9d83ce159d153` |
| `frozen_design_autond_ce2.py` | `67139e70d9a574fa0bda8b0894a2bf5bc7bdfbe7a315430592e444fe57693be6` |
| `experiments/ce2/autond_theory.py` | `1b13c89fd6387da6d507663d283e1f2b566793aeefa5c1f41698a2d50ad3c45b` |
| `experiments/ce2/autond_ce2.py` | `685abc4faeb032df6c7de101978fdab5c8c3fb128d202b45f735cc79c3d5171f` |
| `experiments/ce2/autond_runner.py` | `4ede422109cb6335bbeaec415d303a9ed1afbe741b0066caa5ef292be1a3092c` |
| `experiments/ce2/autond_verify.py` | `3ea96ae5911a90f1266b55498135f95117a19fd4c8dd94d1d07a26b84e2710ce` |
| `kaggle/autond_ce2_kaggle.py` | `556d83104f247d46d5fcc8ca7b0a53255e424f1f0e51336f0ae12f45fda38a6b` |
| `tests/test_autond_ce2.py` | `32d77275f29027bb53e5be5ed4e5268543550c49ab0c4890c746e028460f0516` |
| `autond_ce2/preregistration.json` (design `7245d152…06b1`) | `20f2e8457852ff084c9d20504e46e8a78f80ad4377944e2729432a0dfff6fc1b` |
| `autond_ce2/README.md` | `1e9fe80ace51d4fe7d0bf1d2ef8773f59762a24b26ecd7aab7df8d37825e2b8f` |
| `autond_ce2/POWER_ANALYSIS.md` | `e3acbfc33435814d4305fe8b1d3d70623eb6d97dfd13ad88e27dcbcb1bddfa7f` |
| `autond_ce2/RUNBOOK.md` | `1ed142a5aece065257de6166cef9b5ce9ef496519cf98f3bedbc541f3890c3b0` |
| `autond_ce2/validation/model_binding_check.json` | `3b9b011df45b41c9385c0c9f755cbc8d5211e9d5567f0ef3825e5c9cfc8e18a3` |
| `autond_ce2/validation/mathematical_verification.json` | `3109dfcda73d4c6f0eee9c5d2c2ec0f3adf496e693b1fd4bfddb53f83ac8d93e` |
| `autond_ce2/validation/power_analysis.json` | `86cc3199577e6dd33e8851efa2a0da0650d3b26080ad306034ee1ca8fd7ea07d` |
| `autond_ce2/validation/benchmark_cpu.json` | `c9716e8743340954e526cd3ac7c4bd89485482412883420cb20b10328800f98d` |
