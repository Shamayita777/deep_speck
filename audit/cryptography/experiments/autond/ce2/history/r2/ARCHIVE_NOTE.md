# r2 archive (AutoND-CE2-frozen-design-2026-10-r2, design ef3d764a8350e01a68fdcacf2fb38a15fd5622bfc8665f20e0e71df3f2ba4ba5)

Never approved; no confirmatory production run; exploratory smoke runs only.

Every file here is the exact r2 version of a file that r3 changed (or the r2 document it replaced).

| Archived file | r3 path | Source of the r2 bytes |
|---|---|---|
| `design.py.r2` | `design.py` | copied from the working tree before the first r3 edit |
| `preregistration.json` | `preregistration/preregistration.json` | same |
| `docs/*.md` (4) | `docs/*.md` | same |
| `implementation/power.py.r2` | `power.py` | same |
| `implementation/tests_test_theory_statistics.py.r2` | `tests/test_theory_statistics.py` | same |
| `implementation/kaggle_autond_ce2_kaggle.py.r2` | `../../../kaggle/autond_ce2_kaggle.py` | same (not changed by r3; kept for completeness) |
| `implementation/verify.py.r2` | `verify.py` | copy-on-write clone of the r2 working tree taken 2026-10-31 before r3 (scratch `clone4`); its design.py, power.py, test_theory_statistics.py, analysis.py, docs and preregistration are byte-identical to the files above, and the clone differs from the first r3 verify.py only by the two r3 checks |
| `implementation/tests_test_pipeline.py.r2` | `tests/test_pipeline.py` | same clone; differs from the first r3 version only by the r3 tamper case |
| `implementation/analysis.py.r2` | `analysis.py` | same clone; equal to the working tree until the final r3 review |
| `related/REPAIR_REPORT_2026-10-31.md.r2` | `../../../REPAIR_REPORT_2026-10-31.md` | the working-tree file minus the r3 addendum; r3 only appended to it (one `>>` append), so this prefix is the pre-r3 file |

Files added by r3 have no r2 version: `tests/test_r3_specification.py`, `validation/r3/*`, `history/r2/*`,
`history/r3-draft1/*`.

`SHA256SUMS` covers every file here except itself and this note.
