# G-CE5 source verification (r3)

**Primary source.** P1 is A. Benamira, D. Gérault, T. Peyrin, Q. Q. Tan, "A Deeper Look at Machine Learning-Based Cryptanalysis", ePrint 2021/287.
- Copy: `~/Downloads/benamaira.pdf`, 35 pages, SHA-256 `a0f48abeefb48358ac629d92bc653dd9d141a8d83c1116be9491bb3b22d03cae`.
- PDF metadata: created 2021-03-23 (MiKTeX pdfTeX).
- Page numbers below are PDF pages. On every cited page (7, 9, 17, 18, 19, 27, 32) the printed folio equals the PDF page number; this was checked against the folio printed at the foot of each extracted page.
- The text was extracted with pypdf. Superscripts are lost in extraction (for example, "2 12" is 2^12 and "10 7" is 10^7) and are restored below.

## 1. Quotations

| # | location | text (verbatim, superscripts restored) |
|---|---|---|
| Q1 | p. 7 | "Gohr computed the full DDT for the input difference Δin, using the Markov assumption. Then, to classify a ciphertext pair (C, C′), the probability p of the output difference C⊕C′ is read from the DDT and compared to the uniform probability … Real if DDT(Δin → Δout) > 1/(2^32 − 1)" |
| Q2 | p. 9, Table 1 ("taken from Table 2 of [11]", i.e. Gohr 2019) | "5 D5 0.911 0.877 0.947 N5 0.929 ± 5.13×10^−4 0.904 ± 8.33×10^−4 0.954 ± 5.91×10^−4" |
| Q3 | p. 17, Sect. 4.5 | "The training set of Gohr's neural network consists of 10^7 ciphertext pairs. Thus, we restrict our distinguisher to only use 10^7 ciphertext pairs as well." |
| Q4 | p. 17 | "If we know the difference at round i, the i − 1 round difference for the right part is known as well, since r_{i−1} = (l_i ⊕ r_i) ≫ 2" |
| Q5 | p. 18 | "we created a distinguisher that uses an approximated DDT (aDDT); that is, a truncated DDT that is experimentally constructed based on n ciphertext pairs. In this distinguisher, we use n = 10^7 … The idea of the distinguisher is to decrypt the last round, nr, using all possible subkey bits that are relevant to the bits we are interested in. Then, we compute the average of the probabilities of all partial decryptions for a given pair, read from aDDT (nr − 1), to get a score. If the score is greater than that of the random distribution, the distinguisher will return 1 (Real) and 0 (Random) otherwise." |
| Q6 | p. 18 | "The mask value we have chosen is 0xff8f/ff8f rather than the expected 0xc183/c183 as we believe the truncated differential they are detecting is at nr − 2 rounds." |
| Q7 | p. 18 | "The pseudocodes for creating the aDDT and the average key rank distinguisher can be found in Algorithm 1 and Algorithm 2 given in Supplementary Materials. We applied the distinguisher for 5, 6 and 7 rounds of SPECK-32/64 and the results are given in Table 7. It shows that our distinguisher closely matches the accuracies of Gohr's neural distinguishers." |
| Q8 | p. 18 | "we gave a set of 10^5 5-round ciphertext pairs (approx. 50,000 from real and random distribution each) to both our average key rank distinguisher and N5 … about 97.6% of the ciphertext pairs tested have the same classification in both distinguishers." |
| Q9 | p. 18 | "In the partial decryption, we enumerate the 2^12 keys affecting the rightmost 13 bits of δl_{nr−1} covered by our mask. Therefore, the complexity of our distinguisher is 2^13 one-round SPECK-32/64 decryptions, and 2^13 table lookups." |
| Q10 | p. 19, Table 7 ("Accuracy of the average key rank distinguisher with a mask value of 0xff8f/ff8f") | "5 92.98% 90.76% 95.22%" (rows 6 and 7: 78.79% / 72.53% / 85.07% and 60.28% / 55.31% / 65.24%) |
| Q11 | p. 19, Table 8 | "AKR Dist 1: 46.6% 1.48%; 0: 0.953% 51.0%" (closeness to N5) |
| Q12 | p. 27, Table 10 | "D5 91.1% 87.7% 94.7%; N5 92.9% ± 0.05 90.4% ± 0.08 95.4% ± 0.06; M-ODT (Ours) 92.3% ± 0.08 95.5% ± 0.09 89.1% ± 0.2" |

## 2. Algorithms 1 and 2 (p. 32, Supplementary Materials A), verbatim

```
Algorithm 1 Function used to build the approximated DDT from a given number of random samples and a bitmask.
 1: function BuildApproxDDT(nr, np, m)   ⊿ where nr - Number of rounds, np - No. of pairs, m - bitmask
 2:   aDDT ← ZEROS(2^32)
 3:   for i = 1 to np do
 4:     p ← rand(0, 2^32 − 1)
 5:     p′ ← p ⊕ 0x0040/0000
 6:     key ← rand(0, 2^64 − 1)
 7:     l_nr, r_nr ← ENC_key(p, nr);
 8:     l′_nr, r′_nr ← ENC_key(p′, nr)
 9:     δl ← l_nr ⊕ l′_nr;
10:     δv ← l_nr ⊕ r_nr ⊕ l′_nr ⊕ r′_nr
11:     aDDT[(δl||δv) ∧ m] = aDDT[(δl||δv) ∧ m] + 1/n
12:   end for
13:   return aDDT
14: end function

Algorithm 2 Pseudocode for the average key rank differential distinguisher.
 1: function AKRDD(nr, C, C′, m)   ⊿ where nr - Number of rounds, C, C′ - two ciphertexts, m - bitmask
 2:   aDDT ← BuildApproxDDT(nr − 1, 10^7, m)
 3:   score ← 0
 4:   for key = 0 to 2^12 − 1 do
 5:     l_{nr−1}, r_{nr−1} ← DEC_key(C, 1);   ⊿ Decrypt the last round with the subkey key
 6:     l′_{nr−1}, r′_{nr−1} ← DEC_key(C′, 1)
 7:     δl ← l_{nr−1} ⊕ l′_{nr−1};
 8:     δv ← l_{nr−1} ⊕ r_{nr−1} ⊕ l′_{nr−1} ⊕ r′_{nr−1}
 9:     score += aDDT[δl||δv]
10:   end for
11:   return (score / 2^12 > 2^−HW(m))   ⊿ HW(x) refers to the Hamming weight of x
12: end function
```

## 3. Source-to-code mapping

| source | `akr.py` | `reference.py` | status |
|---|---|---|---|
| A1 L2 table over 2^32 | `MaskedIndex`: 2^26 cells, because only `x ∧ m` indices are ever written (L11); lossless | `cell()` | VERIFIED (representation documented) |
| A1 L4 p ← rand 32-bit | `_pairs`: two uint16 words (l, r) from `os.urandom` | `getrandbits(16)` ×2 | VERIFIED |
| A1 L5 p′ = p ⊕ 0x0040/0000 | `p1l = p0l ^ 0x0040`, `p1r = p0r` | same | VERIFIED; "0x0040/0000" is (left, right) as in Q1/Q2's (0x0040, 0x0000) |
| A1 L6 key 64-bit | 4 × uint16, `expand_key` (Gohr) | `key_schedule` (specification) | VERIFIED; schedules agree (test) |
| A1 L7–8 ENC_key(·, nr) | `S.encrypt` with nr round keys | `encrypt` | VERIFIED |
| A1 L9 δl | `differences()` | `deltas()` | VERIFIED |
| A1 L10 δv = l⊕r⊕l′⊕r′ | `differences()` | `deltas()` | VERIFIED; r2 used δr here, which was **wrong** (test `test_table_is_indexed_by_dl_dv_and_not_dl_dr`) |
| A1 L11 aDDT[(δl‖δv) ∧ m] += 1/n | integer counts; ÷ n at lookup (exact) | integer counts | VERIFIED; m = 0xff8f on δl (left half), 0xff8f on δv (right half) |
| A2 L2 table at nr − 1 = 4, n = 10^7 | `table_rounds = 4`, `table_pairs = 10^7` | n/a | VERIFIED |
| A2 L4 key = 0 … 2^12 − 1 | `CANDIDATES = arange(4096)`, literal | `range(4096)`, literal | VERIFIED |
| A2 L5–6 DEC_key(C, 1) | `dec_one_round` written inline with Gohr's operations | `round_dec` | VERIFIED |
| A2 L7–8 δl, δv | inline, δv = δl ⊕ δr′ | `deltas()` | VERIFIED |
| A2 L9 score += aDDT[δl‖δv] | lookup at the masked cell | lookup at the masked cell | **INTERPRETATION**: see §5.1 |
| A2 L11 score / 2^12 | ÷ 4096 (the candidate count) | ÷ 4096 | VERIFIED |
| A2 L11 > 2^−HW(m) | strict `>`, HW = 26 | strict `>` | VERIFIED; r2 used `≥`, which was **wrong** (test) |
| Q5 "decrypt the last round" | one round | one round | VERIFIED |
| smoothing | none | none | VERIFIED: A1 L2 zero-initialises, and no other step changes the table |
| real/random data for evaluation | Gohr's construction (Q1, p. 7) | n/a | VERIFIED as the paper's setting; no evaluation-set size is given for Table 7 |

## 4. Statement types

| type | items |
|---|---|
| direct statement of P1 | cipher and rounds; input difference; 10^7 pairs; mask 0xff8f/ff8f; table at nr − 1; δl and δv definitions; table indexed by (δl‖δv) ∧ m; one-round decryption; candidate loop 0 … 2^12 − 1; division by 2^12; strict threshold 2^−HW(m); Table 7 values; Table 8 closeness; D5 values as quoted from Gohr |
| derived mathematical fact | the 2^26-cell representation is lossless; only subkey bits 0–11 affect the masked cell, so the literal 2^12 loop equals the 2^16 average for this mask (CRYPTO_CORRECTNESS.md §3); HW(m) = 26; "2^13 decryptions" = 2 per candidate × 2^12 candidates |
| interpretation | Algorithm 2 line 9 is read as a lookup at the masked cell (§5.1); the random class follows Gohr's construction described on p. 7, since P1 gives no separate pseudocode for evaluation data |

## 5. Interpretation and open points

1. **A2 L9 has no "∧ m". This is an interpretation, not a source instruction.**
   - A1 writes only masked indices, so a literal unmasked lookup would be non-zero only when every unmasked bit of (δl‖δv) happened to be 0.
   - Q5 says the score is "the average of the probabilities of all partial decryptions … read from aDDT".
   - The lookup is therefore taken at `(δl‖δv) ∧ m`. This is the only reading under which the 2^−HW(m) random baseline makes sense: an average of masked-cell probabilities is about 2^−26 for random data.
   - P1 does not repeat the mask in Algorithm 2; the masked lookup is the implementation's reading.
   - The gate tests this reading empirically against Table 7; the gate has not been run.
2. **Why 2^12 and not 2^13.**
   - Q9 states that 2^12 keys affect the masked bits, and that the "2^13" in the complexity counts decryptions (two per key: C and C′).
   - The derivation (CRYPTO_CORRECTNESS.md §3) agrees: only subkey bits 0–11 can change the masked (δl, δv) after decryption.
   - The literal 2^12 loop is therefore exactly the average over all 2^16 subkeys. The r2 record stated that 13 bits matter; that bound holds for single decrypted values but is not tight for XOR differences.
   - **The code runs the literal published loop.** The 2^16 equality is a test, not a substitution.
3. **Evaluation-set size for Table 7 is not stated.** Table 8 uses 10^5 pairs. The gate tolerance therefore cannot be derived from the published sampling error (BASELINE_GATE.md).
4. **The D5 value 0.911 is taken from P1** (Table 1 p. 9; Table 10 p. 27), which quotes Gohr 2019, Table 2. Gohr's paper itself was not read.
5. **Mask bit order.** "0xff8f/ff8f" applies to δl (the left 16 bits of δl‖δv) and to δv (the right 16 bits). The mask is the same on both halves, so a left/right swap could not change which bits are kept. Within a half, bit i has weight 2^i.

## 6. Separate published accuracies (5 rounds), not conflated

| result | value | location | role in CE5 |
|---|---|---|---|
| aKR distinguisher | 92.98% / TPR 90.76% / TNR 95.22% | P1 Table 7, p. 19 | gate target |
| plain DDT, D5 | 0.911 / 0.877 / 0.947 | P1 Table 1, p. 9 (from Gohr 2019, Table 2); P1 Table 10, p. 27 | separation reference only |
| Gohr N5 | 0.929 ± 5.13×10^−4 | P1 Table 1, p. 9 | context |
| M-ODT + LightGBM | 92.3% ± 0.08 | P1 Table 10, p. 27 | not run |
| aKR vs N5 closeness | 97.6% same classification (10^5 pairs) | P1 Sect. 4.5, p. 18; Table 8, p. 19 | descriptive comparison only; CE5 does not reuse it |

## 7. Earlier sources, now superseded

The r1/r2 notes relied on co-author slides (Caramba seminar, 2021-06-25) and search extracts. P1 agrees with the slides on the mask, the threshold form, n = 10^7 and the 0.911 baseline. **P1 differs** from the r2 reading on three points: the table is keyed on (δl, δv), the loop runs over 2^12 candidates, and the threshold is strict. The slides' "2^16" loop is marked "(Approximation)" on the slide itself.
