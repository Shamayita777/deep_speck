# Discrepancies Discovered During EV Implementation

These are repository-derived facts surfaced while building EV. They are
Implementation Integrity findings, not EV factors - EV records them for
visibility but does not attempt to resolve them.

## 1. Historical accuracy reporting convention (resolved by direct inspection)

`archive/notebooks/gohr_reproduction_tf2.ipynb`, cell 19's captured output,
shows the reported `0.929099977016449` figure is printed by the script's
own `print("Best validation accuracy: ", np.max(h.history['val_acc']))`
line - i.e. it is the **maximum validation accuracy over 200 epochs of
training history**, not a reloaded best-checkpoint evaluation and not
necessarily the literal final epoch's value. For this specific run the
terminal epochs' `val_acc` values are numerically close to the history
maximum, so the distinction has negligible impact on this particular
number - but the reporting convention itself should be documented
wherever this figure is cited.

## 2. `GohrModel` default depth mismatch (CE-dimension)

`audit.cryptography.gohr.model.GohrModel` defaults to `depth=5`. The
documented/actual reproduction baseline uses `depth=10`
(`archive/train_5_rounds.py`: `train_speck_distinguisher(200, num_rounds=5,
depth=10)`). Any call site that constructs `GohrModel` without an explicit
`depth=10` override silently diverges from the documented baseline.

## 3. `train_speck_distinguisher`'s own bare defaults

`archive/train_nets.py::train_speck_distinguisher`'s function signature
default is `num_rounds=7, depth=1` - neither of which matches the
5-round/depth-10 headline configuration. The actual baseline run required
explicit overrides (`archive/train_5_rounds.py`), not the function's own
defaults.

## 4. Round-count mismatch between existing evidence dimensions

D1-D5 evidence (this project's Dataset Integrity work) is centered on 5
rounds. CE1-CE4 evidence (`cryptography.zip`) reflects 7 rounds
(`baseline_score=0.6108` in the CE1 certificate is consistent with 7-round
Speck32/64 performance). Per the frozen scope, EV uses 5 rounds
exclusively; CE1-CE4 must be rerun at 5 rounds to belong to the same
primary evidence chain (this is a project-level action item, not
something EV resolves).

## 5. Software version drift

This EV implementation was built and smoke-tested against TensorFlow
2.21.0 / Keras 3.15.1 / Python 3.12.3 (this sandbox's installed
versions), while the documented reproduction baseline used TensorFlow
2.20.0 / Keras 3.13.2 / Python 3.12.13
(`archive/notebooks/gohr_reproduction_tf2.ipynb`, cells 8-10). This
mismatch is recorded automatically by every certificate
(`framework.provenance.software_provenance` +
`gohr.baseline.compare_software_environment`) and does not block
smoke-level validation, but production runs should ideally be executed
in an environment matching the documented versions, or the mismatch
should be explicitly accepted and reported alongside any production
result.
