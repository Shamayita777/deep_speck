# ERRATUM 1 — `balanced_accuracy_at_calibrated_threshold`

Applies to: `a4_certificate.json` (run `a4_20261006`), and to any earlier
adversarial certificate produced by `battery.py` before patch set 6.

## What was wrong

In `battery.py::assess_distinguisher()` the true-negative rate was computed as
the positive-prediction rate among the negative class, which is the **false
positive rate**. The field
`distinguisher_assessment.balanced_accuracy_at_calibrated_threshold` therefore
reported `(TPR + FPR) / 2` rather than `(TPR + TNR) / 2`.

## Corrected values

Recomputed from the persisted, hash-verified sealed set
(`datasets/sealed.npz`, content SHA-256 `13c97bf2…`, n = 1,000,000):

| quantity | value |
|---|---|
| confusion at the calibrated threshold | tp 454,578 · fn 45,756 · tn 45,556 · fp 454,110 |
| TPR | 0.908549 |
| TNR | 0.091173 |
| FPR | 0.908827 |
| **balanced accuracy, corrected** | **0.499861** |
| balanced accuracy, as reported | 0.908688 |

## Why no verdict changes

`balanced_accuracy_at_calibrated_threshold` is registered in
`preregistration_v2.DISTINGUISHER_CRITERION["secondary_descriptive"]`. It is a
descriptive quantity and enters no decision rule. Construction validity is
decided by ROC AUC and its bootstrap confidence interval under the
conservative 95% CI equivalence criterion:

    auc 0.499914   ci95 [0.498781, 0.501061]   region [0.49, 0.51]
    equivalence_satisfied = true   construction_status = VALID

Those are unaffected. Every other reported figure reproduces exactly from the
persisted artifacts: `sealed_accuracy_at_0.5` 0.500510 and
`sealed_accuracy_at_calibrated_threshold` 0.500134, both to all recorded
digits. The corrected balanced accuracy of 0.499861 is consistent with an AUC
of 0.4999, whereas the reported 0.908688 was not — the inconsistency was in
the reporting, not in the experiment.

## Disposition

The certificate is **retained unmodified** and this erratum is filed beside
it. The defect is fixed in `battery.py` from patch set 6 onward. A4 was
re-run under the corrected code against the same hash-bound sealed set; the
re-run's AUC, confidence interval and accuracies are bit-identical, which is
itself the reproducibility check on this erratum.

Recompute command:

    python erratum_recompute.py audit/adversarial/evidence/a4_20261006
