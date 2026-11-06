# G-CE5 cryptographic correctness (r3)

This document covers what the code computes and how each step is checked.

- **Sources:** where each step comes from in P1 is in SOURCE_VERIFICATION.md §3.
- **Reproduction:** whether Table 7 is reproduced is decided by the baseline gate, which has not been run.
- **History:** the superseded r2 implementation is archived in `historical/r2_superseded_20261105/`.

## 1. Conventions

**Cipher code.** SPECK32/64 comes from Gohr's code, `audit.cryptography.gohr.speck`. This is the code that generated the network's training data and the CE2 data.
- State: (l, r), two 16-bit words.
- One encryption round:
  - l ← ((l ⋙ 7) + r) ⊕ k
  - r ← (r ⋘ 2) ⊕ l
- One decryption round:
  - r ← (l ⊕ r) ⋙ 2
  - l ← ((l ⊕ k) − r) ⋘ 7
- The decrypted right word does not depend on the key (P1 p. 17); this is tested.

**Key.** Four 16-bit words, (l2, l1, l0, k0). Gohr's `expand_key` equals the specification's key schedule in `reference.py` on 50 random keys.

**Known answers.**
- Both implementations reproduce the published test vector: key `1918 1110 0908 0100`, plaintext `6574 694c` → `a868 42f2` (22 rounds).
- They agree at 5 rounds on 50 random inputs.
- Decryption inverts encryption in both.

**Network input.**
- Encoding: `convert_to_binary([c0l, c0r, c1l, c1r])`, MSB first within each word.
- Inverse: `words_from_binary`, independently re-derived by `verify._bits`.
- The round-trip is exact on production data (§7).

**Pairs.**
- Every pair has its own random key.
- Real pair: P′ = P ⊕ (0x0040, 0x0000).
- Random pair: the second plaintext is independent, under the same key (P1 p. 7).
- Real pairs for the primary endpoint come from the frozen CE2 generator, unmodified, so N, T and D are computed on the same pairs.

## 2. Algorithm 1 table

- **Differences:** δl = l ⊕ l′ and δv = (l ⊕ r) ⊕ (l′ ⊕ r′). A fixed-value test distinguishes δv from δr.
- **Mask:** m = 0xff8f on δl and 0xff8f on δv; HW(m) = 26.
- **Cell:** `pext(δl, 0xff8f) ≪ 13 | pext(δv, 0xff8f)`.

**Lossless representation.** Algorithm 1 L11 writes only indices of the form x ∧ m, so the 2^26 cells hold its entire 2^32-entry table. An exhaustive test over all 2^16 word values shows two things:
- the per-word compaction depends only on the masked bits;
- it maps the 2^13 masked values bijectively onto [0, 2^13).

The joint index is therefore a bijection onto [0, 2^26) with no aliasing.

**Counts.**
- Stored as integer uint32 counts. With n = 10^7 < 2^32, no cell can overflow, and `build_table` refuses n ≥ 2^32.
- Normalised by n only after summation, which equals Algorithm 1's sum of 1/n terms exactly.
- `build_table` generates one batch per loop iteration and advances the counter by that batch's size. Both registered tables hold exactly 10^7 counts.

**Independent table check** (`reference.table_check`):
- A fresh sample is built by the scalar transcription of Algorithm 1, using its own SPECK and Python's `random.Random` stream.
- It is compared with the recorded table by a Pearson chi-square test over the table's 200 most frequent cells with expected count ≥ 5, plus one pooled remainder cell; df = number of cells used.
- Assumptions: the sample cells are independent multinomial draws; the expected counts are taken from the recorded table, so table estimation noise inflates the statistic by a factor of about 1 + n_sample/n_table = 1.01.

What the result does and does not establish:
- A non-significant result shows only that the table is consistent with Algorithm 1's distribution on its most frequent cells. It is not proof of identity.
- Rejection at the wrong round count, the wrong input difference, or with a (δl, δr) key shows that the check can detect those errors.

## 3. Algorithm 2: the 2^12 candidates

**The code** runs the literal loop `key = 0 … 4095` (`akr.CANDIDATES`) and divides the accumulated count by 4096 · n. `full_average()`, the average over all 2^16 subkeys, exists only for tests.

**Candidates versus decryptions.** P1 p. 18 counts "2^13 one-round SPECK-32/64 decryptions, and 2^13 table lookups". That is two decryptions (C and C′) for each of the 2^12 candidates; the candidate count itself is 2^12.

**Claim.** For m = 0xff8f/ff8f, the masked (δl, δv) after one-round decryption depends only on subkey bits 0–11.

**Derivation from the SPECK equations.** One-round decryption of (l, r) under k gives:
- r′ = (l ⊕ r) ⋙ 2, which contains no key;
- l′ = s ⋘ 7, with s = ((l ⊕ k) − r′) mod 2^16.

The steps:
1. For two ciphertexts, δr′ is key-free and δv′ = δl′ ⊕ δr′. So the masked δv′ bits require the same bits of δl′ as the masked δl′ bits do.
2. Bit j of δl′ is bit (j − 7) mod 16 of s0 ⊕ s1.
3. The masked positions {0, 1, 2, 3, 7, …, 15} map to bits {9, 10, 11, 12, 0, …, 8} of s0 ⊕ s1. These are bits 0–12.
4. Write s = x − y with x = l ⊕ k. Then bit b of s is x_b ⊕ y_b ⊕ β_b, where the borrow β_b depends only on bits 0 … b−1 of x and y.
5. In s0 ⊕ s1, the term k_b enters as part of x_b in both s0 and s1, so it cancels. Key bits enter bit b only through the borrows, that is, only through k_0 … k_{b−1}.
6. Bits 0–12 of s0 ⊕ s1 therefore depend on k_0 … k_11 alone.

**Consequence.**
- All 16 subkeys sharing bits 0–11 give the same lookup.
- The 2^12 literal candidates (bits 12–15 zero) represent each class exactly once.
- The integer sum over 2^16 subkeys is 16 times the sum over the 2^12 candidates, so the two averages are the same real number and are rounded identically.
- This agrees with P1 p. 18: "the 2^12 keys affecting the rightmost 13 bits of δl_{nr−1}".

**Tests.**
- `relevant_key_bits()` derives 12 from the mask. It is also checked for three other masks (15, 4 and 10 bits).
- **Exhaustive check over every subkey.** For 800 ciphertext pairs (400 real, 400 random) and all 65,536 subkeys, the cell at k equals the cell at k ∧ 0x0FFF. Flipping bit 11 changes the cell for some pairs, so 12 is the exact number of relevant bits, not an upper bound.
- The published loop equals the 2^16 average bit for bit, for the published mask, on both classes. The benchmark confirms this on production tables.
- **Test that fails if 2^13 or 2^16 candidates were used.** For the mask (0xffff, 0x0000), 15 key bits matter. There the 2^12 result differs from both the 2^13 and the 2^16 results, and equals the literal scalar transcription. `akr_scores` therefore runs exactly 4096 candidates.

## 4. Score and decision (Algorithm 2 L9–11)

- **Score:** score / 2^12, the mean over the 4096 candidates of count[(δl ‖ δv) ∧ m] / n. It is continuous and kept before thresholding; the correlations use only this value.
- **Decision:** real iff score > 2^−26, strictly. The boundary is tested in both implementations.
- **Lookup:** taken at the masked cell, an interpretation of line 9 (SOURCE_VERIFICATION.md §5.1).
- **Correct candidate:** with the true last-round subkey alone, the score equals the table entry of the true 4-round cell (8 pairs).
- **Agreement with the reference:** the vectorised score equals the literal scalar Algorithm 2 to within 1e-15 on toy tables, and exactly (difference 0.0) on the registered tables. Decisions agree.
- **Edge cases:**
  - an all-zero table gives a score of 0 and the decision random;
  - an all-2^32 − 1 table checks the uint64 accumulation;
  - a table with every cell equal to n gives exactly 1.0.

## 5. Secondary comparator (LR)

`difference_scores` reads the 5-round Algorithm-1 table at the masked (δl ‖ δv) of the ciphertexts themselves, with no key guessing. It uses the same differences and cell function as the aKR table. It is a secondary comparator and is not presented as a published distinguisher.

## 6. Registered tables

**Built 2026-11-05.** Record: `tables/tables_manifest.json`. Fresh-seed re-verification: `validation/r3_final_20261105/table_validation.json`.

| table | SHA-256 | rounds | counts | build check (seed 20261105) | re-verification (seed 20261106) |
|---|---|---|---|---|---|
| aKR, `addt_r4_counts.npz` | `09ff9388…f54be6` | 4 | 10^7 in 2^26 uint32 cells; 697,494 non-zero | p = 0.210 (df 200) | p = 0.194; 3 and 5 rounds p ≈ 0; swapped difference p ≈ 0 |
| LR, `addt_r5_counts.npz` | `2ba703db…f91e48` | 5 | 10^7; 3,114,972 non-zero | p = 0.995 | p = 0.641; 4 and 6 rounds p ≈ 0; swapped difference p ≈ 0 |
| (δl, δr)-keyed control, 10^7 pairs, not kept | n/a | 4 | n/a | n/a | rejected: χ² = 53,429, df 200, p ≈ 0 |

**Why the wrong-round statistics are identical.** At the wrong round counts the χ² values coincide (for example, 53,354.7 at both 3 and 5 rounds). The fresh sample hits none of the table's top cells, so the statistic reduces to Σe + (Σe)²/(n − Σe) in both cases.

## 6a. Fail-closed table validation

- **`tables`** runs the independent check before writing each table. On FAIL it writes no table, keeps a `tables_manifest_FAILED_<UTC>.json` record, and stops.
- **`tablecheck`** reports one overall decision: every registered table PASS at its registered n_pairs = 10^7, the (δl, δr) control rejected, and the compaction bijective. It exits non-zero on FAIL.
- **Registered record.** `baseline_D.table_validation` records the repository-relative path and the SHA-256 of the validation used: `validation/r3_final_20261105_tablecheck_v2/table_validation.json`, `28808b8a…`. Absolute paths are refused.
- **Who requires it.** Calibration and production refuse unless that file exists, matches its hash, reports PASS and names the registered tables.
- **Table loading.** `registered_tables` checks hash, dtype, size, rounds, mask, the sum of counts, and n_pairs against the registered table size.

## 7. Model binding

The frozen `verify_reference_model(behavioural=True)` was run on 2026-11-05:
- SHA-256 `256eb4a5…6737`, equal to the registered value;
- 21 Conv1D layers, 10 residual merges, L2 = 1e-5;
- input (None, 64), output (None, 1);
- behavioural accuracy: 0.93008 at 5 rounds, 0.49808 at 6 and 0.4919 at 7, so the frozen round-binding verdict is PASS;
- the word encoding round-trips exactly on generated data.

**Provenance.** The checkpoint is the output of the authors' re-execution of Gohr's `train_nets.py` (commit 2b54425). It is not Gohr's released weights. The frozen plan records `reference_accuracy_reported_by_gohr = 0.9291` (tagged SOURCE-GOHR). P1 Table 1 (p. 9) quotes Gohr's N5 accuracy as 0.929 ± 5.13×10^−4. Gohr 2019 was not read here, so the extra digit is not independently confirmed. The field is frozen and is not changed.

## 8. Memory

| item | value |
|---|---|
| each table | 256 MiB |
| scoring chunk (256 pairs × 4096 candidates) | 2 MiB per uint16 temporary, 4 MiB per uint32 |

Measured peak footprints:

| stage | peak footprint |
|---|---|
| table build | 640 MB |
| table re-verification, including a 10^7-pair control table | 1.29 GB |
| benchmark, including a 50,000-pair replicate with the network | 850–885 MB |
| variability stage | 360 MB |

A 10^6-pair replicate adds about 0.25 GB of arrays.

**Free memory.** Available memory (free, inactive and speculative pages) was 1.6–2.6 GB at different times on 2026-11-05. Calibration and production refuse below 2 GiB.

## 9. Verifier independence

`verify.py` imports only `design.py` and `reference.py`, never `akr.py`, `analysis.py`, `runner.py` or the gohr package. From raw files it recomputes:
- the preregistration hash, the approval status and hash, the timestamp digest and the Bitcoin attestation;
- table hashes, sums and the independent table check;
- gate scores (a subsample) with the literal scalar Algorithm 2; the strict decisions, confusion matrix, CP intervals, both gate components and the decision;
- D for random pairs of both classes in every replicate; a class-label sanity check; T = ∏F;
- every correlation, Fisher z, Δ, Δ_matched, Δ_LR, the primary and secondary intervals and outcomes, the qualifier, incremental value, Holm adjustment and the final decision;
- optionally, N from the checkpoint.

`verify_variability` independently recomputes a variability record, including literal scalar re-scoring of the stored pairs.

**Limitation.** This is implementation independence, not interpretive independence. Both implementations read Algorithm 2 line 9 as a masked lookup.
