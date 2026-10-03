# CE2–CE4 reconstruction, forensic findings and historical comparison

## 1. Why the historical CE2–CE4 evidence cannot be carried forward

Reproduced on fresh data against the historical certificate values. Historical
artifacts were read only; none was modified.

| CE | Historical value | Reproduction | Configuration identified |
|---|---|---|---|
| CE2 | rho −0.1776 … −0.1889 (mean −0.1834) | −0.1817 … −0.1899 (mean −0.1863) | depth-10 checkpoint `256eb4a5…` on **7-round** data |
| CE1 | baseline accuracy 0.6108 | 0.6110 at 7 rounds (0.4176 at 5) | **7-round**, depth 5 |
| CE3 | real 0.23761, selectivity 0.03761 | 0.24068 / 0.04068 at 5 rounds (0.3282 at 7) | depth-5 model, 5-round data |
| CE4 | 0.12619 / 0.05390 / gap 0.07230 | 0.1256 / 0.0536 / 0.0721 | depth-5 model, 5-round data |

Timeline from certificate timestamps plus `test/ce1/signal_destruction.py`,
which saves the trained baseline to `adapter._baseline_model_path`:

* **Aug 6, 04:10–04:17** CE2 ran while `evidence/ce1/best5depth10 (10).h5` was
  still the depth-10 checkpoint, on 7-round data.
* **Aug 6, 18:12–22:51** CE1 trained (16,701 s) at 7 rounds, depth 5, and
  **overwrote that same file**.
* **Aug 17** CE3 and CE4 ran at 5 rounds but loaded the overwritten depth-5,
  7-round-trained model.

So the three CEs used three different model/round combinations and are not
mutually comparable. The historical certificates remain valid records of what
was observed; they are not evidence about the frozen configuration.

## 2. The historical CE2 sign is a configuration artifact, not a contradiction

Decomposing the target into its per-round Lipmaa–Moriai factors, with the
frozen depth-10 model (n = 100,000):

| Rounds | mean output | rho(full trail) | rho(final-round factor only) | rho(rounds 1..n−1 only) |
|---|---|---|---|---|
| 5 (frozen) | 0.8935 | **+0.806** | +0.448 | +0.757 |
| 7 (historical) | 0.0837 | −0.186 | −0.132 | −0.159 |

At 7 rounds the model is outside the problem it was trained on and labels
almost everything "random", which is where the historical negative correlation
comes from. At the frozen 5-round configuration the association is strongly
positive, and it is **not** reducible to the final-round factor, which is the
only part of the target that is a deterministic function of the ciphertext
pair the model sees. Both decompositions are preregistered as secondary
endpoints so this cannot be invoked post hoc.

*Smoke-scale only — not evidence.* These numbers come from forensic runs and
from smoke runs, both explicitly non-evidentiary.

## 3. Defects found and corrected

1. **Degenerate CE3 control.** `signal_destroyed.h5` has an all-zero 64-dim
   penultimate representation, so its probe scores exactly 1/n_classes.
   Historical CE3 measured "real minus chance". The corrected design refuses a
   degenerate control and adds a raw-input control.
2. **CE4's "ceiling" was not a ceiling.** The historical threshold 0.000719
   implies a ceiling of 0.00719, yet the observed gap was 0.0723 — ten times
   larger. No magnitude threshold is used now.
3. **Pseudo-replication.** CE2 inference rested on n = 100,000 samples of one
   fixed model. The corrected unit is the independent evaluation run, and the
   primary test (exact sign test) cannot be driven by the sample size.
4. **Filename trust.** `best5depth10 (10).h5` is depth-5. Model binding is now
   by hash, architecture read from the graph, and a behavioural round check:
   measured 0.9293 at 5 rounds versus Gohr's reported 0.9291, and ≈0.50 at 6
   and 7 rounds.
5. **p-underflow bias.** A Wilcoxon p that underflows (p < 1e-300) was read as
   "missing" and would have produced INCONCLUSIVE for an extremely significant
   result. Fixed and regression-tested.

## 4. Claim ceiling (maximum defensible wording)

| CE | Maximum defensible wording if SUPPORTED |
|---|---|
| CE2 | "this frozen model's output is monotonically **associated** with an analytical single-trail lower bound, beyond permuted-target and untrained-architecture controls" |
| CE3 | "the quintile-discretised single-trail quantity is **decodable** from this model's penultimate representation beyond an untrained-architecture control" — decodability is not use |
| CE4 | "this model's output is **more sensitive** to perturbation of difference-bearing ciphertext structure than to a magnitude-matched XOR-preserving perturbation" — not causal necessity of the analytical quantity |

CE3's target is CE2's construct coarsened, so CE3 is not independent
confirmation of CE2. CE4 never manipulates the analytical quantity. A
conjunctive claim across CE2–CE4 is an intersection-union test and must not be
stated as "cryptographic learning proven".
