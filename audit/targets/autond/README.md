# AutoND Implementation Integrity (case study 2) — revision 3

AutoND-specific adapters, configuration and tests for the CipherMind v3
Implementation Integrity pillar. **Everything here is a target adapter.**
Stage semantics, tiering, validation, the outcome decision and the
evidence tag all come from the generic framework under `audit/common/`
and `audit/implementation/`.

**No file outside `audit/targets/autond/` is added, modified or replaced.**
Verified by diffing the whole tree against a pristine framework copy and
enforced by `tests/test_packaging.py`.

## Revision 3 corrects ten defects found in review of revision 2

| # | Defect in revision 2 | Correction |
|---|---|---|
| 1 | Launcher built `eval_nets.py --model …`, which the pinned parser **does not define**, and omitted required `--model_type`, `--cipher`, `--input_difference`, `--round_number` | New `upstream_interface.py` extracts the parser's own `add_argument` definitions with `ast` (no import, so no TensorFlow), records them bound to `eval_nets.py`'s digest, and validates every command against them. A flag the target does not define now raises at build time. |
| 2 | Nothing could produce a conforming result bundle — the only route was writing `RESULT.json` by hand | New `evaluation_assemble.py` derives the record from the artifacts: loads the NPZs, verifies the checkpoint copies against the plan, cross-checks journal against NPZ, and writes `RESULT.json` + `SHA256SUMS.txt`. Tested end to end against `validate_bundle`. |
| 3 | Freeze gate checked only the declarative `PLAN` hash, so **rewriting `decide()` left the gate open**; the sidecar needn't be committed; generated `git add` used a bare basename | The sidecar now binds the plan **file's bytes** and the live file must match; the sidecar must be tracked and identical to its committed copy; all generated paths are repository-relative. |
| 4 | `--strict` could **never pass** on valid R1 evidence (manifest-bound files have no separate digest), and `ok`/exit hid a wrong upstream commit and FAILED stages | Integrity counts as established when VERIFIED **or** bound by a harness-validated manifest. Three separate gates: `stage_record_problems`, `blocking_problems` (mismatch, wrong commit, FAILED stage — regardless of `--strict`), `strict_problems`. |
| 5 | II-3 reported "8 AGREES" **without reading the pinned source**, including with no checkout at all | `verify_sources()` checks `dbitnet.py`, `eval_nets.py`, `train_nets.py` and HEAD against the pinned commit. `effective_summary()` downgrades every AGREES to `NOT_CHECKABLE_HERE` when they do not verify. |
| 6 | Validator accepted one checkpoint, the string "not clean", unordered timestamps, escaping paths, and metrics that disagreed with the NPZ | Requires both checkpoint IDs, defaults the digests and pinned commit from the plan, compares every metric against the raw NPZ, parses and orders timestamps (and rejects a run predating its own freeze), enforces an exact attestation grammar, and refuses escaping paths. |
| 7 | `--paper-pdf` printed a digest and persisted nothing | Returns a persistable record with digest, file identity, size, PDF header and `revision_asserted: false`; `write_record()` binds the verified digest into the saved comparison. |
| 8 | A second run would **overwrite** the previous run's checkpoints, logs and results | `run_dir_is_usable()` refuses a non-empty directory; a refused launch touches nothing. |
| 9 | Archive wrote `SHA256SUMS.txt` and `run_evidence/` at the **repository root** | Everything lives under `audit/targets/autond/`. `tests/test_packaging.py` enforces it. |
| 10 | An untracked `.DS_Store` in the checkout **blocked every run** | `checkout_state()` separates modified tracked files (blocking) from OS/interpreter debris (recorded, non-blocking) from other untracked files (blocking). Nothing is ever deleted. |

```
audit/targets/autond/
  upstream_interface.py                 the target's real CLI, read from source  NEW
  evaluation_assemble.py                result-bundle assembly                   NEW
  evaluation_gate.py                    gated launcher                           rev3
  evaluation_results.py                 schema, validator, exact Student-t       rev3
  implementation_stages.py              II-1 and II-3 stage records              rev3
  paper_source_architecture.py          II-3 paper-to-source comparison          rev3
  run_autond_implementation_audit.py    entry point -> generic certificate       rev2
  preregistrations/autond_ii_v2.py      successor plan (NOT FROZEN)              rev3
  preregistrations/__init__.py
  tests/portable.py                     portable fixtures + frozen-repo fixture  rev3
  tests/test_r3_corrections.py          70 tests for the ten items above         NEW
  tests/test_packaging.py               the packaging contract                   NEW
  tests/test_corrections.py                                                      rev3
  tests/test_implementation_stages.py                                            rev2
  tests/test_run_autond_implementation_audit.py                                  rev2
  README.md                             this file
  SHA256SUMS.txt                        the manifest, inside the target
  run_evidence/                         logs and exact commands
```

**Not included, deliberately:** `audit/targets/__init__.py`,
`audit/targets/autond/__init__.py`, `audit/targets/autond/tests/__init__.py`.
Your Phase 1 copies are correct and are left alone; this package reads
`PINNED_UPSTREAM_COMMIT` from yours rather than redefining it.

## How the read-only upstream requirement is kept (item 10)

Your checkout had an untracked `.DS_Store`. Revision 2 ran
`git status --porcelain` and blocked on any output, so that one file
would have stopped every run — and the obvious "fix" of deleting it
would have modified the pinned target, which is forbidden.

`checkout_state()` classifies instead of cleaning:

| Category | Examples | Effect |
|---|---|---|
| `tracked_dirty` | ` M dbitnet.py`, staged or deleted tracked files | **Blocks.** The target is no longer the pinned revision, so every digest the plan records may describe a changed file. |
| `untracked_benign` | `.DS_Store`, `__pycache__/`, `*.pyc`, `.idea/`, `*.swp` | **Does not block.** Recorded in the attestation and **left exactly where it is**. A `.DS_Store` beside the checkpoints does not make them different files. |
| `untracked_other` | `sitecustomize.py`, `ciphers/evil.py`, a stray `.h5` | **Blocks.** An untracked file in the checkout *can* change what the target computes: a stray module shadows an import, and the cipher list is built by globbing `ciphers/*.py`. |

Nothing is deleted, moved or modified — a test walks the gate's AST and
fails if it calls `unlink`, `rmtree`, `remove`, `rmdir`, `move` or
`rename`. The attestation records what was found, in an exact grammar the
validator checks:

```
tracked-clean;untracked-benign=0
tracked-clean;untracked-benign=2:.DS_Store,__pycache__/
```

Revision 2 accepted any string containing "clean", which accepted
`"not clean"`. On this build container the pinned checkout currently
carries an untracked `__pycache__/`, and the gate records
`tracked-clean;untracked-benign=1:__pycache__/` and proceeds.

## Installation (macOS)

NumPy 1.26.x is **mandatory**: NEP 50 integer promotion breaks
`ciphers/speck3264.py` at import under NumPy 2.

```bash
cd ~/Documents/deep_speck
python3 -m venv .venv-autond
source .venv-autond/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install 'numpy==1.26.4' 'tensorflow==2.15.1'
```

Every path in the archive is under `audit/targets/autond/`, so extracting
at the repository root cannot create or replace anything else. Check that
for yourself before extracting:

```bash
# 1. list what the archive would write - expect ONLY audit/targets/autond/...
unzip -l autond_ii_implementation_r3.zip

# 2. confirm nothing outside the AutoND target, and no Phase 1 initializer
unzip -Z1 autond_ii_implementation_r3.zip \
  | grep -v '^audit/targets/autond/' || echo "nothing outside the target: OK"
unzip -Z1 autond_ii_implementation_r3.zip \
  | grep -E 'audit/targets/(__init__|autond/(__init__|tests/__init__))\.py$' \
  && echo "UNEXPECTED: a Phase 1 initializer is present" \
  || echo "no Phase 1 initializer shipped: OK"

# 3. extract, then verify the manifest from the repository root
unzip -o autond_ii_implementation_r3.zip -d .
shasum -a 256 -c audit/targets/autond/SHA256SUMS.txt
```

The manifest lives at `audit/targets/autond/SHA256SUMS.txt` and its paths
are repository-root-relative, which is why `shasum -c` is run from the
repository root. The logs live at `audit/targets/autond/run_evidence/`.
Revision 2 put both at the archive root; if you extracted revision 2,
delete its root-level `SHA256SUMS.txt` and `run_evidence/`.

## Running

## Running

```bash
cd ~/Documents/deep_speck
source .venv-autond/bin/activate
export AUTOND_ROOT=/path/to/pinned/AutoND      # also read as AUTOND_UPSTREAM

# 1. This package's tests
python3 -m unittest discover -s audit/targets/autond/tests -t . -p 'test_*.py'

# 2. Your Phase 1 census + load-verification tests (need AUTOND_UPSTREAM)
AUTOND_UPSTREAM=$AUTOND_ROOT python3 -m unittest \
    audit.targets.autond.tests.test_architecture_census \
    audit.targets.autond.tests.test_architecture_census_real \
    audit.targets.autond.tests.test_census_load_verify

# 3. Serialise the architecture comparison so II-3 can bind a real file
python3 -m audit.targets.autond.implementation_stages \
    --repo-root . --home ~ --autond-root "$AUTOND_ROOT" \
    --write-architecture-record \
      audit/targets/autond/evidence/architecture/paper_source_architecture_v2.json

# 4. Stage records (runs no experiment). --strict fails closed.
python3 -m audit.targets.autond.implementation_stages \
    --repo-root . --home ~ --autond-root "$AUTOND_ROOT" --strict \
    --output audit/targets/autond/evidence/stage_record.json

# 5. Plan status: recomputes every recorded digest against the checkout
python3 -m audit.targets.autond.preregistrations.autond_ii_v2 \
    --autond-root "$AUTOND_ROOT" --repo-root .

# 6. Certificate, dry run then emit
python3 -m audit.targets.autond.run_autond_implementation_audit \
    --repo-root . --home ~ --autond-root "$AUTOND_ROOT" --dry-run
python3 -m audit.targets.autond.run_autond_implementation_audit \
    --repo-root . --home ~ --autond-root "$AUTOND_ROOT" \
    --output-dir audit/targets/autond/evidence

# 7. The target's real interface, read from its source
python3 -m audit.targets.autond.upstream_interface --autond-root "$AUTOND_ROOT"

# 8. The evaluation gate. This REFUSES while the plan is unfrozen, and
#    refuses a run directory that already has anything in it.
python3 -m audit.targets.autond.evaluation_gate \
    --autond-root "$AUTOND_ROOT" --repo-root . \
    --output-dir ~/Documents/autond_eval_runs/$(date -u +%Y%m%dT%H%M%SZ)

# 9. AFTER a separately authorised run, assemble and then validate.
#    Neither step runs an evaluation.
python3 -m audit.targets.autond.evaluation_assemble <run-dir> --repo-root .
python3 -m audit.targets.autond.evaluation_results <run-dir> \
    --plan-hash $(python3 -c "from audit.targets.autond.preregistrations \
        import autond_ii_v2 as p; print(p.plan_hash())")
```

`evaluation_results` **fails** without `--plan-hash`: a structurally valid
bundle is not a plan-bound one, and revision 2's CLI reported the first
as though it were the second. Pass `--structure-only` to assert structure
alone and have it say, in the output, that the result is not confirmatory
for any preregistration.

`--autond-root` must be the pinned checkout at commit
`8d66e0a43a9e6fbbb3624d68296c1b77aa2da779`. It is opened **read-only**;
nothing here writes to it, and the gate refuses an output directory
inside it.

Step 4 with `--strict` **writes nothing** when validation fails, so a
failed record cannot be left on disk for something else to read. Step 6
refuses to overwrite an existing manifest or certificate.

## Test results — what was run, where, and on what

Every figure is from a logged run in this build environment. Logs are in
`run_evidence/`.

| Suite | Modules | Count | Result | Interpreter | Log |
|---|---|---|---|---|---|
| **This package** | `test_corrections` 91, `test_implementation_stages` 41, `test_run_autond_implementation_audit` 26 | **158** | **OK** | py3.11.17 / TF 2.15.1 / NumPy 1.26.4 | `autond_suite_tf215.log` |
| **This package**, no `AUTOND_ROOT` | same three | **158** | **OK**, 1 skipped | py3.11.17 | `autond_suite_portable.log` |
| **Phase 1** census + load-verify (yours) | `test_architecture_census` 31, `test_architecture_census_real` 9, `test_census_load_verify` 21 | **61** | **OK** | py3.11.17 / TF 2.15.1 | `phase1_census_tf215.log` |
| Full discovery over `tests/` | all six above | **219** | **OK** | py3.11.17 / TF 2.15.1 | `autond_discovery_tf215.log` |
| **Search harness** (yours) | `test_search_runner` 29, `test_no_training_import` 11 | **40** | **OK** | py3.11.17 / TF 2.15.1 | `search_tf215.log` |
| **Gohr framework regression** (yours) | `audit/tests/` | **426** | **OK** | py3.13 | `gohr_regression_v2.log` |

**Test provenance, stated plainly.** 158 + 61 + 40 + 426 = **685 distinct
tests executed here**, logs included. Of these I **wrote the 158** in this
package; the 61 Phase 1, 40 search-harness and 426 Gohr tests are
**yours**, executed unmodified to show this package breaks nothing. The
219 figure is not additional work — it is the 158 plus the 61 discovered
together, and the no-`AUTOND_ROOT` row is the same 158 under a different
environment. Nothing in this table is a count of tests I did not run.

Two environment facts worth knowing:

* The Phase 1 census tests **fail rather than skip** when
  `AUTOND_UPSTREAM` is unset, by design — a skipped audit test is
  indistinguishable from a passing one in a summary line. Set the
  variable or they will report 5 errors.
* One test in this package skips without `AUTOND_ROOT`: the one that
  requires recorded digests to actually match. Every negative test —
  wrong digest, invented commit, torn log — runs unconditionally, since
  those are the tests that catch a gate failing open.

## Current status

| Stage | Tier | Status | Why |
|---|---|---|---|
| II-1 Baseline reconstruction | CORE | `PARTIAL` | 3 components; R1 and evaluation evidence not on the audit machine |
| II-2 Independent reimplementation | CORE | `NOT_PERFORMED` | No independent Algorithm 1 (R3) implementation exists |
| II-3 Controlled verification | CORE | `PARTIAL` | 3 components; census evidence not on the audit machine |
| II-4 Comparative evaluation | CORE | `NOT_PERFORMED` | Nothing to compare against — depends on II-2 |
| II-5 Statistical assessment | CORE | `NOT_PERFORMED` | One execution per cipher, no replicates, no variance |

**Pillar decision `INCONCLUSIVE`. Evidence tag `EXPLORATORY`.**

Both are derived by your generic code. The tag is `EXPLORATORY` because
no frozen pre-execution preregistration exists; the runner passes
`preregistration_reference=None` rather than manufacturing confirmatory
status.

Statuses depend on which machine you run on. On this build container,
with the staged-training bundle installed, `II1-STAGED-TRAINING` reaches
`CONFORMANCE_VERIFIED` with 12 bound artifacts. With deliberately wrong
census files in place, II-3 correctly reports `FAILED` with the observed
and expected digests printed.

### II-3 paper-to-source comparison (revision 2)

**16 claims: 8 AGREES, 0 DIFFERS, 8 UNDERSPECIFIED.**
Record digest `8cba1c2086fc6fb06017c88c42dec05b1bdcb061d1ab57c11d7c7306ec2c8b3f`.

Compared against **ePrint 2022/1467, last of 4 revisions, 2024-01-24**
(IACR ToSC 2024). Revision 1's 3/0/9/1 summary is superseded; the four
changes are recorded in `CORRECTIONS` inside the module so the earlier
reading is not silently replaced.

Now agreeing, with locators: dilation schedule (§6.2 rule + **Table 3**
values `[31,15,7,3]`), filter progression (§6.2, `32 + i*16`), block
count, the `d=1` companion layer, and — newly checkable — **total
parameter count against Table 4's 298.11k**, which the census's 298,113
matches at the table's printed precision. That last one is the strongest
single check in the comparison, because one number depends on every
architectural choice at once.

**Three corrections to the brief I was given.** I did not encode these as
instructed, because the authoritative revision does not support them:

1. **Dilation rates are in Table 3, not Table 4.** Table 3 is "Settings
   for Gohr's neural network and DBitNet"; Table 4 is "FLOPs, parameters
   and runtime per epoch". Three independent extractions agree.
2. **Head sizes `256, 256, 64` are not in the paper.** The *source* has
   exactly those (`dbitnet.py:26-27`, `:61-70`) — your values are right
   about the code. But "prediction head" appears once in the paper, in
   §3.2, about **Gohr's** network; the only dense size it states is
   Gohr's 64 neurons in §4.2; and `256` appears only as SPECK's key size,
   Gohr's filters/neurons, Table 3 input sizes, and HIGHT's `C·256`.
   ARCH-09 is therefore `UNDERSPECIFIED`, not `AGREES`. **Figure 4c
   ("The actual design of DBitNet") carries no machine-extractable
   text** and is the one place the paper might state them — if your
   rendered copy shows them, flip the row to `AGREES` with a figure
   locator. Given that a prior review here retracted paper-derived head
   widths after reading a stale revision, I would not assert this in
   either direction without reading that figure.
3. **§6.2's filter statement is confirmed** — that one was right, and
   ARCH-06 moved from `UNDERSPECIFIED` to `AGREES` because of it.

One finding worth your attention: **ARCH-16**. The paper says neuronal
width shrinks "by a factor of two", but the source realises
`64 → 33 → 18 → 11 → 8`, not `64 → 32 → 16 → 8 → 4`, because dilations
are precomputed from the *original* input size while `valid` padding
subtracts `d` from the *current* length. It is recorded
`UNDERSPECIFIED` rather than `DIFFERS` because a qualitative phrase
attached to a schematic figure cannot be falsified quantitatively — but
the realised sequence is in the row so a reviewer who reads it as
quantitative can score it `DIFFERS`.

**The paper PDF's own sha256 is `UNVERIFIED`.** This build environment
cannot reach `eprint.iacr.org`, so the comparison was made against
extracted text, never a locally hashed file. Bind it yourself:

```bash
python3 -m audit.targets.autond.paper_source_architecture \
    --paper-pdf ~/Downloads/2022-1467.pdf
```

### Preserved discrepancies

* **SIMON accuracy**: paper 0.5179 vs README 0.5182. Preserved
  unreconciled. The equivalence margin (0.00179) **exceeds** the 3e-4
  gap, so this experiment *cannot* identify which is correct — which is
  exactly why `H-AUTOND-SIMON-REPRO` predicts `INCONCLUSIVE`.
* **R1 reproduction**: recovered `0x400000` **and** `0x408000`; the
  README documents one. `highest_non_random_round = 5` against README
  prose 7 and its own log 6. Recorded as a partial match, as observed.
* **Polished vs unpolished**: the staged-training bundle evaluated the
  **unpolished** round-8 checkpoint (`815062f1…`) against 0.5096. The
  plan's target is the **polished** one (`a7be788d…`) at 0.5144. Two
  different artifacts; the component summary now names which, and the
  plan carries an explicit note against conflating them.

## The statistical decision rule (replaces revision 1's)

Revision 1 concluded equivalence when "the 95% interval of the five-set
mean contains the published value". That is unsound, and not marginally:
five observations at the SD this project has observed (6.707e-4) give an
interval about **1.7e-3 wide**, so any published value within 8.5e-4 of
the mean is "contained" — including **both** disagreeing SIMON values at
once. The rule would have returned `SUPPORTED` for either, and could
never have identified which. Holm was also named but applied to interval
containment, which has no p-value to correct.

**Replacement.** Per comparison, with `d = mean − p_ref`, `SE = s/√5`,
`df = 4`:

* **TOST**: `t_lower = (d+Δ)/SE` upper-tailed, `t_upper = (d−Δ)/SE`
  lower-tailed, `p_TOST = max(p_lower, p_upper)`. Equivalence when the
  Holm-adjusted `p_TOST < α`.
* **Difference test**: two-sided `t = d/SE`, so "shown to differ" and
  "too imprecise to tell" are distinct outcomes instead of both
  collapsing into `NOT_SUPPORTED`.
* **Holm**, stated exactly: sort the family's p-values ascending,
  compare `p_(i)` against `α/(m−i+1)` for `m = 3`, stop at the first
  failure and reject nothing from that rank onward. Applied
  **independently** to the TOST and difference families; a p-value is
  never carried between them.

**Margin**: `Δ = 0.10 × (p_published − 0.5)` — a fraction of the
*distinguishing advantage*, not of the accuracy, because the advantage
is the scientific quantity. SPECK 0.00144, SIMON 0.00179 / 0.00182. Four
justifications are recorded in `EQUIVALENCE_MARGIN`, including the one
that cuts against the result: the margin **exceeds** the SIMON
paper/README gap, so the experiment cannot separate them, declared in
advance rather than discovered afterwards.

**Power** is computed by `power_at()`, not asserted: at the most
conservative Holm step (α/3), approx **0.895** for SPECK and **0.995**
for SIMON, using a planning SD of 6.7e-4 drawn from this project's own
five-set evaluation and used for **design only** — never as an analysis
prior.

Student-t is computed exactly (regularised incomplete beta, Lentz
continued fraction) rather than tabulated, because Holm pushes α to
values no small table carries. Verified against published critical
values to 4 decimals for df 1–30.

**This plan is unfinished and must not be frozen or executed.** Items 6
and 7 of your review are addressed in code, but the plan is a scientific
document and the margin, the power target and the hypothesis predictions
are yours to accept or change before it is frozen.

## Freezing (when you are ready — not yet)

Plan hash `fd8de5dd0e2779090e646b4d32b06a4b121e0ef2505772529655998f22e7f48d`.
No sidecar is shipped; `is_frozen()` returns False and the gate refuses.

Revision 1's procedure was **impossible**: it asked for
`git rev-parse HEAD` *after* committing the sidecar, but committing the
sidecar changes HEAD, so the value it had to contain could never exist.
The fix is two commits — the plan file alone first, whose commit is
knowable because the sidecar does not exist yet; then the sidecar. The
commit is verified by reading the plan back out of it:

```
git show <plan_file_commit>:<plan_file_path>
```

and checking it hashes to the recorded `plan_file_sha256`. Run
`python3 -m audit.targets.autond.preregistrations.autond_ii_v2` for the
exact steps.

`is_frozen()` rejects: an incomplete sidecar, a stale plan hash, a
timestamp that is not `YYYY-MM-DDTHH:MM:SSZ`, a commit that is not 40 hex
characters, a commit that does not exist, a commit that does not contain
the plan file, and a commit whose plan file hashes differently. All eight
are tested, and so is the **positive** path — a test builds a real git
repository, commits the plan and confirms the gate opens, because a gate
that cannot open is a wall, not a gate.

That positive test earned its place immediately: it found a real bug in
my own code. `verify_freeze_commit` was hashing `git show` output that
had been decoded to text and `.strip()`ped, which can never equal a
file's byte digest — so the gate would have stayed closed even with a
correct sidecar. Fixed by reading the blob as raw bytes.

## Expected artifacts

21 artifacts across three roots (`repo`, `home`, `upstream`). On this
build container, with the staged-training bundle installed: **9
VERIFIED** (2 shipped checkpoints + 7 bundle files), 1 `RECORDED_ONLY`
(the bundle's own manifest), the rest `MISSING` because they live on your
Mac.

The census, reconciliation and load-verification artifacts expect your
timestamped filenames (`census_v2_20261015T043855Z.json`,
`load_verify_r2_20261015T051547Z.json`). Untimestamped
container-generated versions exist here but are **different files**;
binding them would mean marking evidence VERIFIED that was never
inspected, so they are left `MISSING`.

## Framework version note (unresolved, for your decision)

Five framework files differ between your two trees — the one bound by the
Gohr II-4 preregistration manifest and the one in your `common.zip`
installs:

| File | Nature of difference |
|---|---|
| `audit/common/outcomes.py` | **docstring only**, in `AssessmentTier` |
| `audit/__init__.py` | differs from the frozen manifest |
| `audit/implementation/preflight.py` | differs from the frozen manifest |
| `audit/tests/test_freeze_manifest_coverage.py` | differs from the frozen manifest |
| `audit/implementation/adapters/gohr_declared_architecture.json` | differs from the frozen manifest |

I verified all five are **byte-identical to your pristine framework
copy**, so none was touched by this work; the divergence is between your
framework and the hashes the Gohr preregistration recorded, and it
predates this package. `audit/implementation/stages.py` is byte-identical
everywhere (`4a57419cd218…`), so all five II stages are CORE regardless.

The `outcomes.py` one has a consequence worth noting: the 4508-byte copy
carries a **stale** docstring asserting II-2 is ENHANCED and "not a
prerequisite for CORE Implementation Integrity". That text already caused
one incorrect claim in this project. This package was built and tested
against the 4469-byte copy. Reconciling them is your call; nothing here
modifies either.

## Outstanding empirical work

* **II-2**: implement published Algorithm 1 from the paper text alone
  (R3), without reading `optimizer.evo()`, and run it on SPECK32/64 and
  SIMON32/64.
* **II-4**: run the released repository and that independent
  implementation under identical protocols; compare recovered
  differences and `highest_non_random_round`.
* **II-5**: replicate the search with independent seeds for a variance
  estimate; evaluate the shipped checkpoints once the plan is frozen.
* **ARCH-09**: read Figure 4c in your copy of the paper and either flip
  the row to `AGREES` with a figure locator or leave it unresolved.
* **Paper PDF digest**: bind it with `--paper-pdf`.

## Superseded certificate (r3e)

A certificate written before the r3e fix may record
`II-1_BASELINE_RECONSTRUCTION  FAILED` with the finding

```
R1 evidence failed validation ... .DS_Store is present but not named
in the manifest.
```

That verdict is wrong — see `CHANGELOG_R3E.md`. It is nonetheless a real
record of a real run and **must not be deleted**. Move it aside, record
why, and write the corrected certificate:

```bash
cd ~/Documents/deep_speck
EV=audit/targets/autond/evidence
mkdir -p "$EV/superseded"

mv "$EV/autond-implementation-certificate-V1.json" \
   "$EV/superseded/autond-implementation-certificate-V1.SUPERSEDED-r3d.json"
mv "$EV/run_manifest_ii-autond-assembly-001.json" \
   "$EV/superseded/run_manifest_ii-autond-assembly-001.SUPERSEDED-r3d.json"

cat > "$EV/superseded/WHY_SUPERSEDED.md" <<'NOTE'
These two artifacts were produced under revision r3d, before the r3e fix
to search/verify_outputs.py.

They record II-1_BASELINE_RECONSTRUCTION = FAILED. That verdict is an
artifact of a defect in the audit code, not of the evidence: a .DS_Store
created by Finder in the R1 search output directory was counted as an
unmanifested file, which failed bundle validation, which suppressed the
scientific reading of a valid bundle.

The R1 evidence was intact: every digest named in MANIFEST.json matched,
then and now. The finding that was suppressed - the released search
returning 2 epsilon-close differences where the README documents one -
is unchanged and appears in the superseding certificate.

Retained rather than deleted because a record of a run that produced a
wrong verdict is itself evidence about the audit instrument.
NOTE
```

Then write the corrected certificate:

```bash
python -m audit.targets.autond.run_autond_implementation_audit \
  --autond-root "$AUTOND_ROOT" --repo-root . --dry-run
python -m audit.targets.autond.run_autond_implementation_audit \
  --autond-root "$AUTOND_ROOT" --repo-root .
```

`II-1` should read `PARTIAL`, with `II1-SEARCH-R1  PARTIAL`,
`II1-STAGED-TRAINING  CONFORMANCE_VERIFIED` and
`II1-SHIPPED-CHECKPOINT-EVAL  COMPLETED`.
