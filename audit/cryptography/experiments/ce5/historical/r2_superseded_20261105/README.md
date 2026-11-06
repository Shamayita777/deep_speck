# G-CE5 r2: superseded, kept as historical validation

**Superseded on 2026-11-05.** The operator's reading of the primary source (ePrint 2021/287, Supplementary Algorithms 1 and 2) reports three things that r2 did not do:
- the published distinguisher enumerates **2^12** candidate partial subkeys;
- it normalises by 2^12 and uses a **strict** threshold;
- it builds the table from (δl, δv).

r2 instead averaged over all 2^16 last-round subkeys, computed exactly over 2^13 classes, and indexed (δl, δr).

The r2 proof that the 2^13-class average equals the 2^16 average still holds. **It says nothing about equivalence to the published 2^12-candidate algorithm.**

## What these files validated, and what they did not

They validated:
- the r2 implementation against its own specification: 58 tests, all passing on 2026-11-05;
- the r2 tables (`tables/`, built 2026-11-05T10:20:16Z), which passed the independent table check for the r2 definition (p = 0.198 and 0.360; p ≈ 0 at the wrong round counts);
- the r2 scoring cost (`validation/`: 0.109 ms/pair, and 0.160 ms per real pair for a full replicate).

They are **not** validation of the corrected algorithm, and must not be cited as such. None of these files was produced from confirmatory data. No gate, calibration or production run ever used them.

## Hashes

| file | SHA-256 |
|---|---|
| `tables/addt_r4_counts.npz` | `a9b35972a82707081bc24ccec9747ae58accab20b45152c5746014dc43c5f058` |
| `tables/ddt_r5_counts.npz` | `d8010a5d9e2bc6ec2098bba6775e4effd10ee7aafe046047bd1d11882bf69e2f` |
| `tables/tables_manifest.json` | `837c7e6cb088877c1c6e4873cf7d9c7681e9d67da3184b5b2c020f43b670f4d5` |
| `validation/benchmark.json` | `41ba27376403b7c59959603fa7106cdbfd6ff669964ae88b1b071023be15b15c` |
| `validation/benchmark_scoring_only_20261105T102145Z.json` | `07b50fa1b803b14264a233e2044f054cc63c242aa7fea86c1c3b8bfb4e970b98` |
| `akr_r2.py` (copy) | `6c564da19bf5956a95c3ff9581c43888194fdbfe7f2d1a9c29bf4d186fb50ec5` |
| `reference_r2.py` (copy) | `443b38a63000d2ec4a2a0a752d733466db87ea13e52a80d586a84e1fc606b359` |
| `design_r2.py` (copy) | `1593071983471cb99e4d276eb1c1b7d903da1db6048cff1933cffa5f4aa814d6` |
| `preregistration_r2.json` (copy, draft, never approved) | `21f70ea94f71e336f669b241349fedeae2a538e19f69559598196aa1c5809673` |
