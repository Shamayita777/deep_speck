# Changelog — r3d → r3e

**One defect, fixed: a `.DS_Store` dropped into the R1 search evidence
directory by Finder made a valid bundle read as corrupt, which flipped
`II1-SEARCH-R1` from `PARTIAL` to `FAILED` and discarded a real
scientific finding.**

No statistical, preregistration, gate or decision logic is touched. The
plan is untouched, so **the plan hash is unchanged and the operator's
existing freeze remains valid**:

```
f20cd9b12909d21dffd22aed83a6f303e8f445412490708f1f01c79a3c264bb7
```

## What happened

`search/verify_outputs.py` rule 2 required every file present in the
output directory to be named in `MANIFEST.json`. That is the right rule
for evidence. It is the wrong rule for operating-system debris.

Browsing the evidence folder in Finder creates `.DS_Store`. It was not
present when the manifest was written, it is not evidence, and it cannot
affect a search result — but rule 2 counted it as an unmanifested file,
failed the bundle, and `validate_r1` then correctly refused to read any
scientific result from a bundle that failed validation. The visible
effect was:

```
II1-SEARCH-R1  FAILED
  R1 evidence failed validation, so no scientific result was read from
  it: .DS_Store is present but not named in the manifest.
```

and with a `FAILED` component, `II-1_BASELINE_RECONSTRUCTION` became
`FAILED` rather than `PARTIAL`.

**The operator's evidence was intact throughout.** Every digest in the
manifest still matched. The finding that was suppressed — the released
search returning 2 epsilon-close differences where the README documents
one — is unchanged and is restored by this fix.

**This was an inconsistency in my own code.** `evaluation_gate` already
had exactly this policy, in `BENIGN_UNTRACKED`: OS/interpreter files are
recorded, do not block, and are never deleted. The operator saw it work
in the preflight they had just run —
`untracked-benign=2:.DS_Store,__pycache__/`, *"LEFT IN PLACE"*. I never
applied the same policy to the R1 bundle validator, so two validators in
one adapter disagreed about what counts as evidence.

## The fix

`search/verify_outputs.py`:

- `BENIGN_NAMES`, `BENIGN_DIRS`, `BENIGN_SUFFIXES` and `_is_benign()`,
  matching `evaluation_gate`'s policy so the two cannot drift apart.
- Rule 2 now excuses debris, **records** it in
  `summary["unmanifested_benign"]` with a note, and leaves it on disk.
  Nothing is ever deleted.
- Matching is on path **parts**, not substrings, so a file called
  `my.DS_Store.json` or `__pycache__notes.txt` is still content and
  still fails.
- A file the manifest **does** name is hash-checked as before, whatever
  it is called. The exemption applies only to unmanifested paths.

`implementation_stages.py`:

- `validate_r1()` carries `unmanifested_benign` up from the harness.
- The `II1-SEARCH-R1` component adds the note to its own `limitations`,
  so the record **discloses** that the directory held files the manifest
  does not name and why that did not invalidate it. Passing silently
  would have been the wrong fix.

## Tests

`search/tests/test_search_runner.py`, 4 new:

| test | pins |
|---|---|
`test_os_debris_does_not_invalidate_the_bundle` | the regression: `.DS_Store`, `run_000/.DS_Store`, `._result.json`, `__pycache__/x.pyc`, `.ipynb_checkpoints/…`, `Thumbs.db` all excused, recorded, and **still on disk afterwards** |
`test_a_real_added_file_still_fails_even_beside_debris` | the exemption is not a hole — debris excused, content not |
`test_a_name_merely_containing_a_benign_string_is_not_excused` | `my.DS_Store.json`, `__pycache__notes.txt`, `results.pyc.json` all still fail |
`test_a_manifested_file_is_still_hash_checked_even_if_benign_named` | the exemption never reaches manifested files |

`test_an_added_file_is_detected` is unchanged and still passes: `extra.txt`
is content.

## Results

```
search harness (both modules) ............... 44 OK
full target discovery, TF 2.15.1 ............ 400 OK, 9 skipped
Gohr/DeepSpeck regression ................... see run_evidence/gohr_regression_v3e.log
```

Both changed files are under `audit/targets/autond/`, so the generic
framework is untouched; `audit/common/equivalence.py` is byte-identical
to r3c.

## What the operator must do about the already-written certificate

`autond-implementation-certificate-V1.json` on disk records
`II-1 FAILED`. That verdict is wrong, but it is a real record of a real
run and **must not be deleted**. The procedure is to move it aside under
a name that says it was superseded, record why, and write the corrected
certificate. See the README section "Superseded certificate".

## Not changed

- The preregistration, its hash, the freeze, the decision rule, the
  power calculation, the margin, `n = 10`, the two-invocation design.
- The confirmatory evaluation. It does not need rerunning: the bundle is
  valid, ingested, and `II1-SHIPPED-CHECKPOINT-EVAL` was `COMPLETED`
  even in the run that reported `II-1 FAILED`.
- Every gate; none weakened. The one rule that changed became *more*
  precise, not more permissive: it now distinguishes evidence from
  debris instead of treating both as evidence.
