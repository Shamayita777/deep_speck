# G-CE5 resource measurements (r3, non-confirmatory)

**Machine.** Apple M1, 8 cores, 8 GB, macOS 13.5. Python 3.11.14, numpy 2.4.6, scipy 1.17.1, Keras 3.15.0, TensorFlow 2.21.0 (`~/.venvs/gohr-tf221-validation-20261031`).

**State on 2026-11-05:**
- free disk: 42–44 GB;
- available memory (free + inactive + speculative pages from `vm_stat`): 1.6–2.6 GB, varying with other applications.

**Method:** wall clock (`perf_counter`), process CPU time, `ru_maxrss` and `/usr/bin/time -l` (peak memory footprint).

## Units

- **Per pair:** one ciphertext pair scored by Algorithm 2 (4096 candidates).
- **Per real pair:** one real pair together with its random-class counterpart. Both are generated, both get network inference, both get aKR scoring, and the real pair also gets LR scoring. The replicate and production projections use this unit.

## Measurements

Records: `validation/benchmark.json` (11:16:06Z) and `validation/r3_final_20261105/benchmark.json`.

| measurement | 11:16:06Z | r3_final | note |
|---|---|---|---|
| Algorithm 1 table, 10^7 pairs, 4 rounds | 1.3 s | n/a | from `tables/tables_manifest.json` |
| Algorithm 1 table, 10^7 pairs, 5 rounds | 1.8 s | n/a | same |
| whole `tables` stage, including 6 independent 10^5-pair checks | 13.0 s, 640 MB | n/a | |
| Algorithm 2, literal 2^12 loop, per pair (2,048 pairs) | 0.081 ms | 0.040 ms | sensitive to concurrent load; not used for projections |
| 2^16-subkey average, per pair (128 pairs; test only) | 0.73 ms | 0.72 ms | |
| 2^12 loop vs 2^16 average | equal bit for bit | equal bit for bit | |
| literal scalar Algorithm 2 (pure Python), per pair | 11.2 ms | 11.6 ms | maximum difference from the vectorised score 0.0 |
| full replicate, per real pair (50,000 real + 50,000 random pairs) | 0.109 ms | 0.110 ms | used for projections |
| raw output, per real pair (compressed npz) | 36.3 bytes | 36.3 bytes | |
| peak footprint, benchmark | 885 MB | 850 MB | |
| `tablecheck` stage, including a 10^7-pair control table | n/a | 1.29 GB | `validation/r3_final_20261105/table_validation.json` |
| `variability` stage (5 tables, 2×10^5 scored pairs each) | n/a | 51 s, 360 MB | both runs |
| frozen model binding check | n/a | 8.5 s, 341 MB | |

## Projections at the registered size (10^6 real pairs per replicate, rate 0.110 ms per real pair)

| item | value |
|---|---|
| one replicate (2 × 10^6 pairs scored and inferred) | about 110 s |
| R = 20 | 0.61 h |
| R = 60 | 1.84 h |
| raw storage at R = 60 | 2.2 GB |
| gate (10^5 + 10^5 pairs, aKR scoring only) | under 20 s |
| calibration (10 replicates) | about 18 min |
| verifier, per replicate | about 0.7 s of scalar re-scoring (32 pairs per class at 11.6 ms), plus about 2 s of correlations and incremental models |
| verifier, fixed cost | about 15 s for the two table checks |
| verifier, optional `--model` | adds network inference for 256 pairs per class per replicate |

A 10^6-pair replicate adds about 0.25 GB of arrays to the benchmark peak. Calibration and production refuse below 2 GiB of available memory, and this guard is retained. No step copies a table, and gathers run on chunks of at most 4 MiB.

**Benchmark replicates.** They were used for timing only and discarded; no statistic was computed from them.
