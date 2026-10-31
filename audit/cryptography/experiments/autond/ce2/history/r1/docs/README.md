# AutoND CE2 — theory consistency

**Experiment** `AUTOND-CE2-THEORY-CONSISTENCY`. **Design** `AutoND-CE2-frozen-design-2026-10-r1`
(`frozen_design_autond_ce2.py`; serialised in `preregistration.json`).

**Status:** the design is complete and pending review. Production refuses to start until the design is
frozen and the preregistration is approved; see `RUNBOOK.md`.

## Question

Is the output of the published AutoND SPECK32/64 eight-round DBitNet statistically associated with a
Markov-model single-characteristic probability proxy? The proxy is calculated from the realized
eight-round XOR-difference trajectory of genuine difference-bearing plaintext pairs (ΔP = 0x400000,
words x = 0x0040, y = 0x0000 in upstream order).

- **Model:** the published checkpoint
  `model_weights/speck3264_single-key_0x400000_round8_polished.h5` (SHA-256 `a7be788d…cc26`) from
  Crypto-TII/AutoND at `8d66e0a4`. It is loaded into upstream `dbitnet.make_model(64)` and verified
  strictly.
- **Score:** the sigmoid output of `model.predict`.
- **Target:** for each pair, T = Σᵣ log₂ xdp⁺(ROR₇(Δxᵣ₋₁), Δyᵣ₋₁ → Δxᵣ) over the 8 rounds, from the
  states reconstructed with upstream `expand_key` / `enc_one_round`. Each factor is the
  Lipmaa–Moriai probability over uniform 16-bit operands. Each factor is computed exactly by two
  independent methods and must be positive.
- **Unit:** one evaluation run.
- **Primary endpoint:** Spearman's ρ per run. The primary test is an exact two-sided sign test over
  runs.
- **Controls:** the same scores against a permuted target, and an untrained DBitNet twin. Each is
  compared with the primary using exact one-sided paired sign tests on |ρ_primary| − |ρ_control|,
  with Holm correction, tested only after the primary rejects.
- **Decisions:** `SUPPORTED`, `ASSOCIATION_OBSERVED_CONTROL_UNRESOLVED`, `INCONCLUSIVE` and
  `PRECHECK_FAILED`.

## Scope and limitations

- The study is observational and supports no causal claim.
- The target is a Markov-model proxy. It is not the exact fixed-key characteristic probability:
  SPECK round keys come from one master key through the key schedule and are not independent
  (Beyne and Rijmen, CRYPTO 2022). It is also not the differential probability of the full cipher,
  and not an unconditional bound.
- The study is conditional on one published checkpoint.
- The population is genuine pairs only, which differs from the balanced population the distinguisher
  separates.
- An association does not establish learned differential cryptanalysis or an internal
  representation.
- Per-round quantities are descriptive latent trajectory quantities. The last-round factor is not
  claimed to be determined by the ciphertext pair.
- The CE2 behavioural check is an independent model-binding validation. It is not a matched
  replication of the published accuracy (about 0.5144).
- CE2 is independent of AutoND CE1. It uses no CE1 model or evidence, and it does not amend or
  explain CE1.

## Relationship to the shared framework

AutoND CE2 implements the shared `CryptographicAdapter` CE2 interface. Each condition of each run is
measured by the unchanged framework `TheoryConsistencyTest`, which makes three calls per run. Each
call receives the same immutable run dataset, and its hash is checked on every adapter call.

The framework test treats higher scores as better and maps undefined correlations to 0.0. Two
design choices keep these properties out of the CE2 result:
- The inference and decision are AutoND-specific and two-sided.
- Undefinedness is checked before measurement and raised as a non-`ValueError`, so it can never
  become a numeric zero.

Production, provenance and independent verification are a separate AutoND pipeline. They are not
the Gohr `ce234/` production path, and no Gohr, CE1 or framework file is modified.

## Files

| Path | Role |
|---|---|
| `adapters/autond_ce2.py` | framework adapter, pinned upstream, model binding, data and trajectory |
| `frozen_design_autond_ce2.py` | frozen design, hash, preregistration checks |
| `experiments/ce2/autond_theory.py` | Lipmaa–Moriai closed form, carry-pair automaton, literal enumeration, gates C/D |
| `experiments/ce2/autond_ce2.py` | Spearman, exact tests, Holm, order-statistic CI, decision, power analysis |
| `experiments/ce2/autond_runner.py` | preflight, initialisation, run state machine, finalisation, certificate |
| `experiments/ce2/autond_verify.py` | independent verifier (imports none of the above) |
| `kaggle/autond_ce2_kaggle.py` | command-line entry point |
| `tests/test_autond_ce2.py` | tests (no training) |
| `autond_ce2/` | preregistration, power analysis, verification report, runbook, `validation/` reports |

The pinned upstream is read from `audit/cryptography/autond_ce1/AutoND-upstream` (read-only), or from
`AUTOND_UPSTREAM`. Its files are checked against `audit/cryptography/upstream_sha256.json`.

## References

- A. Bellini et al., *A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input
  Differences*, IACR ePrint 2022/1467; https://github.com/Crypto-TII/AutoND.
- H. Lipmaa and S. Moriai, *Efficient Algorithms for Computing Differential Properties of Addition*,
  FSE 2001, doi:10.1007/3-540-45473-X_28; IACR ePrint 2001/001.
- T. Beyne and V. Rijmen, *Differential Cryptanalysis in the Fixed-Key Model*, CRYPTO 2022; IACR
  ePrint 2022/837.
