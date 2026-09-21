# Independence of the EV implementation from D1-D5 / CE1-CE4

## What is reused, and why reuse does not compromise independence

| Component | Source | Reused? | Rationale |
|---|---|---|---|
| Speck32/64 cipher primitives (`enc_one_round`, `expand_key`, `encrypt`, `make_train_data`, `convert_to_binary`) | `archive/speck.py` | **Yes** - ported verbatim into `gohr/speck.py`, with attribution | This is the task-defining cryptographic substrate, not a procedural choice under investigation. Its correctness is verified independently via `check_testvector()` (Gohr's own published test vector), not assumed. |
| Residual Conv1D architecture (`make_resnet`) | `archive/train_nets.py` | **Yes** - ported verbatim into `gohr/model.py`, unmodified | Architecture is a *held-constant control* in every EV core experiment; it is not itself under test in the frozen core. Reusing Gohr's exact, unmodified architecture is required for EV's baseline to mean the same thing as the documented baseline. |
| `audit.dataset.adapters.gohr.GohrAdapter` (D-dimension) | `dataset.zip` | **No** | This adapter hard-codes `shuffle=False` - one of the exact procedural choices H-EV-SHUFFLE investigates. Importing it would make EV's baseline silently inherit that choice. |
| `audit.cryptography.gohr.{model,dataset,trainer,evaluate}` (CE-dimension) | `cryptography.zip` | **No** | This adapter's defaults (`rounds=7`, `depth=5`) diverge from the frozen 5-round/depth-10 baseline, and its trainer does not expose `shuffle` as a parameter at all. Importing it would silently anchor EV to a different, undeclared configuration. |
| `audit.dataset.common.{certificate,provenance}` | `dataset.zip` | **No (schema mirrored, not imported)** | EV's `framework/certificate.py` and `framework/provenance.py` follow the same *conventions* (sha256 hashing, PASS/CONDITIONAL_PASS-style fail-closed certificates) for cross-dimension consistency, but are implemented independently so EV has no import-time dependency on the D package. |

## Residual dependency EV does not resolve

EV's conclusions remain conditional on the correctness of the shared Speck cipher/data-generation primitives themselves. Verifying that correctness beyond `check_testvector()` is Implementation Integrity's responsibility, not EV's.
