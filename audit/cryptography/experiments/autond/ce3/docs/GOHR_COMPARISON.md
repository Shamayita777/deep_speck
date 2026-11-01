# What AutoND CE3 takes from Gohr CE3, and what it does not

This comparison comes from a read-only inspection of the live tree. No Gohr file was changed. AutoND CE3 is a new experiment, not a reproduction of Gohr CE3.

## Four different things are called "Gohr CE3"

**1. Legacy route.**
- `adapters/gohr.py:62-63, 373-391`: `provide_control_model()` loads `evidence/ce1/signal_destroyed.h5`, not an untrained twin.
- `adapters/gohr.py:134`: the representation is `layers[-2]`.
- `adapters/gohr.py:187-219`: per-replicate quantiles with `np.digitize`.
- `test/ce3/representation_interpretation.py:109`: the test uses that control.
- `test/ce3/representation_interpretation.py:138-145`: it thresholds at p < 0.025.
- `probe/evaluation.py:58-78`: `LogisticRegression(max_iter=1000)` with no scaling.
- `probe/evaluation.py:160-176`: `RepeatedStratifiedKFold`, stratified on the true labels.
- `probe/significance.py:48`: SciPy Wilcoxon.
- `statistics.py:68` says Wilcoxon has "no distributional assumption"; in fact its null is symmetry about 0.

**2. Frozen design.**
- `frozen_design.py:137` states H0 as "mean <= 0" but tests it with a signed-rank test.
- Fixed sequence at alpha = 0.05 (`frozen_design.py:149-150`).

**3. CE234 production.** `ce234/production.py`:
- `211-216`: the twin is rebuilt from a seed in every session and its weights are not hashed.
- `387-390`: the representation is `layers[-2]`.
- `518-520`: per-replicate quintiles of a tied target, giving unequal classes (for example 19110/16240/21582/19071/23997).
- `539-547`: the calibration balance is random, not exact.
- `548-565`: only summaries are retained.
- `ce234/verify.py:147-175` checks consistency of the stored summaries and recomputes nothing.
- `ce234/FREEZE_MANIFEST.json:77` records that the probe did not converge.

**4. Certificates.**
- The historical `evidence/ce3/ce3_certificate.json` has control score exactly 0.20000000000000004: five-class chance from an all-zero representation. Its selectivity of 0.0376 is trained minus chance.
- `evidence_current/ce3/certificate.json` uses an untrained twin (about 0.230) and gives a selectivity of about 0.338.

**A3.** `adversarial/battery.py:64-113` builds a network whose penultimate layer encodes the CE3 target while its output ignores it. CE3 still reports SUPPORTED there, with selectivity about 0.503, so CE3 measures decodability, not use.

## Kept

- The replicate as the unit, with folds nested inside it.
- A calibration gate first, then a fixed sequence.
- The same examples and folds for every arm.
- An untrained twin as the representation control.
- The raw-input arm as descriptive only.
- The rule that decodability is not use.

## Not kept, and what replaces it

| Gohr | AutoND CE3 |
|---|---|
| signal-destroyed (all-zero) control; trained minus chance | the hash-pinned CE2 twin, required to be non-degenerate |
| p < 0.025 thresholds | exact sign tests in a fixed sequence at 0.05 |
| unscaled probe, non-converged | fold-local scaler; a convergence gate with an independent gradient check |
| folds stratified on true labels | label-independent `KFold` on indices |
| per-replicate quantiles of a tied target | whole-value edges frozen once from the model-free P1 |
| mean hypothesis tested with Wilcoxon | the estimand q = P(D > 0) over all valid replicates, which the exact binomial test actually tests; zeros count as non-positive |
| calibration = BA above twin | C_task (recover at least half of the model's own advantage) and C_twin |
| twin rebuilt from a seed | one twin file, pinned by file and weights hash |
| summaries only; verifier checks arithmetic | full raw evidence; verifier recomputes from raw pairs to the decision |
