# EV Calibration Plan and ε Decision Report

Status: **EV production remains BLOCKED.** This document records the
calibration tier that now exists in code, the pilot design, the
unresolved human decisions, and why they are not resolved here.

---

## 1. The calibration tier

A third run mode, `calibration`, sits beside `smoke` and `production`.

| | smoke | **calibration** | production |
|---|---|---|---|
| protocol | 3 rounds, depth 1, 1 epoch, 256 samples | **identical to production** | frozen baseline |
| purpose | exercise the pipeline | **estimate nuisance variance** | confirmatory evidence |
| `non_evidentiary` | true | **true** | false |
| `calibration` flag | false | **true** | false |
| may enter Holm family | no | **no** | yes |

Calibration deliberately uses the **identical frozen protocol**
(5 rounds, depth 10, 200 epochs, batch 5000, Adam, cyclic LR,
`reg_param=1e-5`, 10M/1M/1M). A reduced-fidelity pilot would estimate
the variance of a *different* experiment and could not support a power
analysis for this one.

That fidelity is precisely why calibration is dangerous, and why the
guards are in code rather than convention: **a calibration certificate is
numerically indistinguishable from a production certificate**. Enforced:

- `framework.certificate.require_confirmatory_certificate` — gates every
  confirmatory consumer; missing status fields are treated as *not*
  confirmatory, so absence of a claim is never read as a claim.
- `apply_primary_family_correction` calls that gate on both certificates,
  so a pilot can never enter the Holm family.
- `run_ev.py` mode-locks each output directory via a `RUN_MODE` marker;
  calibration and production can never share a directory, so a resumed
  run cannot mix pilot and confirmatory replicates into one denominator.
- `analyze_ev_calibration.py` **refuses production certificates** as
  input and emits variance only — no mean, no CI, no p-value — so an
  observed pilot effect cannot leak into planning.

---

## 2. Pilot design

### What each pilot arm estimates

| Arm | Varies | Estimates |
|---|---|---|
| **A. EV-BASELINE** | fresh dataset + fresh model seed | σ: total single-arm variability |
| **B. EV-NOISE** | nothing (same dataset, same seed, same config) | σ_exec: pure execution nondeterminism |
| **C. H-EV-SHUFFLE paired** | shuffle policy only, within pair | **σ_Δ: SD of the paired difference** |

### Gap 1 — the documented pilot cannot power the primary family

`docs/statistical_plan.md` (line 85) proposes estimating the margin and
power from "EV-BASELINE/EV-NOISE pilot replicate-to-replicate spread".
**Neither arm estimates σ_Δ**, and σ_Δ is what drives power for the two
primary hypotheses, both of which are *paired*:

- Arm A measures variability **across** datasets and seeds. H-EV-SHUFFLE
  pairs **share** both (`same_dataset=True`, `same_model_initialization=True`),
  so σ_Δ excludes exactly the components A measures. Using σ from A would
  badly **overestimate** the required n.
- Arm B holds *everything* fixed including the shuffle policy, so it
  measures execution noise only and **underestimates** σ_Δ, which also
  contains the shuffle factor's own run-to-run variability.

A paired calibration arm (C) is therefore required. Using a pilot of the
same comparison to estimate nuisance variance is standard internal-pilot
practice and does not contaminate confirmatory inference, **provided**
pilot pairs are excluded from the confirmatory analysis (enforced by the
mode lock) and are never used to choose ε or the assumed effect
(enforced by the analyser omitting effects).

### Gap 2 — pilot size is not determined by the methodology

The repository mentions "10–15 pilot pairs". That figure is not derived
anywhere. Precision of a variance estimate from n replicates:

| n | rel. SE(s) | 95% CI for σ | CI width |
|---|---|---|---|---|
| 5 | 35.4% | [0.60s, 2.87s] | 4.80× |
| 10 | 23.6% | [0.69s, 1.83s] | 2.65× |
| 15 | 18.9% | [0.73s, 1.58s] | 2.15× |
| 20 | 16.2% | [0.76s, 1.46s] | 1.92× |
| 30 | 13.1% | [0.80s, 1.34s] | 1.69× |
| 40 | 11.3% | [0.82s, 1.28s] | 1.57× |

Sample size for a 95% upper bound within +X% of s: **+50% → n=18;
+40% → n=25; +30% → n=37; +25% → n=49; +20% → n=70.**

**Consequence:** required n scales with σ², so a pilot at n=10–15 leaves
σ uncertain by a factor of 2.2–2.7 and the resulting production design
uncertain by roughly **5–7×**. At n=10 the true σ could plausibly be
1.83× the estimate, meaning the "correct" production size could be
~3.3× larger than the pilot suggests.

**DECISION: 10 matched pairs per paired primary** (H-EV-SHUFFLE and
H-EV-REPRESENTATION). This is an **operational predeclared pilot size,
NOT a claimed universal statistical minimum and not a literature-derived
rule**; the table above records exactly what precision it buys (σ known
to within roughly a factor of 2.6, hence required-n uncertain by ~5-7×).
That residual uncertainty is handled by sizing production on the **95%
upper confidence bound** for σ rather than the point estimate.

EV-BASELINE and EV-NOISE pilot sizes remain undeclared: they do not
estimate σ_Δ and therefore do not size the production design.

---

## 3. ε — **FROZEN at 0.01** (decision taken; report retained below for provenance)

**DECISION: ε = 0.01 absolute confirmatory accuracy (Option 1).** The
audit-wide smallest effect treated as practically meaningful for this
metric, matching the margin frozen for II-4 so one predeclared scale
applies across the audit. It is a methodological judgment, **not** a
literature-derived universal threshold, and it was **not** selected from
or validated against pilot noise. The option analysis that preceded the
decision is retained below.

### Option 1 — adopt ε = 0.01, consistent with the frozen II-4 δ
*Evidence:* δ = 0.01 was already frozen by human decision for II-4
materiality, on the **same metric** (accuracy on a sealed 10⁶ test set)
and the same task.
*Consequence:* cross-pillar coherence — one margin of practical
significance for the whole audit; a difference too small to matter for
implementation conformance is also too small to matter for a procedural
factor. Reviewers see one number, justified once.
*Weakness:* II-4's δ was itself a decision, not a derivation. Adopting it
inherits that, and the two questions are not identical (II-4 compares
architectures; EV compares procedural factors).

### Option 2 — anchor to the CE evidential classification
*Evidence:* the strongest anchor in principle — "an accuracy change large
enough to change which evidence level the CE conclusion sits at".
*Consequence:* genuinely consequence-based, exactly what the methodology
asks for.
*Weakness:* **not yet computable.** It requires the depth-10 CE reruns to
exist. CE1's 0.05 relative-difference threshold and CE4's 0.00072 are
thresholds on *different quantities* and are not transferable.

### Option 3 — literature tolerance (Picard 2021)
*Evidence:* ~0.1% SD and ~0.5% max–min from seed alone on
ResNet-50/ImageNet.
*Consequence:* **rejected as a source for ε.** This is a *measurement-noise*
argument, not practical significance — precisely the confusion the task
warns against. It can legitimately place a **lower bound** (ε should
exceed pure seed noise) but cannot set the value, and it is from a
different task with no established transfer to Speck32/64.

### Option 4 — cryptanalytic-consequence anchor
*Evidence:* distinguishing advantage is `2p − 1`. At p = 0.9293,
advantage = 0.8586; ε = 0.01 shifts it by 0.02 (~2.3% relative), and
since distinguishing data complexity scales ~1/advantage², by ~4.7%.
*Consequence:* the only route that ties ε to a cryptanalytic quantity.
*Weakness:* the natural endpoint is key-recovery data complexity, which
is **explicitly out of the primary case-study scope**. Using it would
either expand scope or rest on an untested bridge.

### Decision taken
**Option 1 chosen: ε = 0.01**, with its weakness recorded (II-4's δ was
itself a judgment, not a derivation). Deferring is scientifically
cleaner and blocks EV production for longer. Options 3 and 4 are not
recommended as primary sources on the evidence available.

Until ε is frozen, `run_ev.py` refuses production for both paired
experiments, and a non-significant result could only ever be
INCONCLUSIVE — a foreseeable design defect, not a finding.

---

## 4. Power analysis

`scripts/plan_ev_power.py` uses the **existing** `framework/power.py`
simulation (no second implementation), simulating the actual procedures:
paired t-test for difference detection and TOST for equivalence.

Multiplicity: the frozen plan applies Holm across the two-member primary
family to difference-detection p-values only (TOST has its own IU
control). For a family of two, the worst case for the first-ordered
hypothesis is α/2, so the planner simulates and reports power at **both**
α and α/2 and selects on the conservative α/2 value.

The planner refuses to run without (a) a hash-bound calibration variance
artifact containing a **σ_Δ** estimate and (b) a supplied ε; and
`framework/power.py` rejects `target_effect_source="observed_production"`.

---

## 5. Unresolved decisions blocking EV production

1. **Pilot precision target** → pilot sizes for arms A, B, C.
2. **Whether to run paired arm C** (required for a defensible power
   analysis; see Gap 1).
3. **ε** — Option 1 now, or defer to Option 2 after the CE reruns.
4. **Assumed target effect** for the power simulation — must be a
   scientifically motivated minimum effect of interest, not an observed one.
5. **Compute budget**, which bounds the feasible design.

Items 1–2 must be settled before the pilot runs; 3–5 before production.


## 6. Calibration compute

The two primary calibration configurations comprise **40 trainings in
total: 20 per hypothesis** (10 matched pairs x 2 arms), each at the full
frozen protocol (5 rounds, depth 10, 200 epochs, 10M/1M/1M).

**No runtime claim is made.** Earlier text asserted ~1.1 GPU-hours per
training; that figure was never measured in this project and has been
removed. A runtime estimate may only be stated once an explicitly
labelled observed benchmark artifact exists. Compute is in any case never
an input to the power analysis and cannot alter the required K.

## 7. Dataset independence wording

Calibration and production datasets are **independently generated; no
explicit overlap guarantee** is provided. Earlier wording claimed blocks
were "non-overlapping by construction", which the generator does not
enforce: `make_train_data` draws from `os.urandom` independently per
call, making collisions astronomically unlikely but not excluded by
construction.
