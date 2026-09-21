# CipherMind Audit Report (audit-report-v1)

Generated: 2026-09-19T16:12:11.283544+00:00
Authoritative snapshot: `audit-snapshot-V4.json` (freeze version 2)

> This report is generated mechanically from the machine-readable evidence documents. It asserts nothing that is not recorded in them, and every conclusion below cites the evidence IDs that produced it.

## 1. Claim Identification and Scope

**Claim C1**: For the 5-round Gohr/Speck32/64 neural distinguisher under differential (0x0040, 0x0000), the observed above-chance distinguishing performance is supported by cryptographically meaningful evidence after dataset, implementation and experimental alternative explanations have been evaluated.

**Scope exclusion - key_recovery**: Gohr's 11/12-round key-recovery apparatus (key_rank.py, test_key_recovery.py, key_averaging.py and released weights) exists in the repository but is NOT part of this claim chain. No CE1-CE4 experiment exercises it, and no evidence chain currently supports a key-recovery claim. It is recorded here solely so its exclusion is explicit rather than accidental.

## 2. Final Claim-Level Decision

| Field | Value |
|---|---|
| Final outcome | **INCONCLUSIVE** |
| Evidence level | **LEVEL_1_PREDICTIVE** |
| Decision policy | `ciphermind-decision-policy-v1` |
| Blocking conflicts | ['CONFLICT-CE2-CE34-V1'] |

**Rationale (from the decision record):** Unresolved conflict(s) ['CONFLICT-CE2-CE34-V1'] persist after the 3.8.8 procedure; the affected claim is INCONCLUSIVE. An unresolved conflict among cryptographic observations on the same target quantity prevents any claim that observations are consistent with cryptographic reasoning. Evidence remains at the predictive level.

## 3. Dimension Findings

| Dimension | Outcome | Rule | Rule provenance | Driving evidence |
|---|---|---|---|---|
| CRYPTOGRAPHIC_EVIDENCE | **INCONCLUSIVE** | `RULE-CONFLICT-PRECEDENCE` | SOURCE_DEFINED | `EV-CE2-RUN1-INGESTED`, `EV-CE2-RUN2-INGESTED`, `EV-CE2-RUN3-INGESTED`, `EV-CE2-RUN4-INGESTED`, `EV-CE2-RUN5-INGESTED`, `EV-CE3-INGESTED`, `EV-CE4-INGESTED` |
| DATASET_INTEGRITY | **INCONCLUSIVE** | `RULE-AGG-CONSERVATIVE-V1` | INTEGRATION_OPERATIONALIZATION | `EV-D2-INGESTED`, `EV-D3-INGESTED`, `EV-D4-INGESTED`, `EV-D5-ABSENT` |
| EXPERIMENTAL_VALIDITY | **INCONCLUSIVE** | `RULE-AGG-CONSERVATIVE-V1` | INTEGRATION_OPERATIONALIZATION | `EV-EV-ABSENT` |
| IMPLEMENTATION_INTEGRITY | **INCONCLUSIVE** | `RULE-AGG-CONSERVATIVE-V1` | INTEGRATION_OPERATIONALIZATION | `EV-II-DEPTH-001` |

### Rule provenance

- **SOURCE_DEFINED** - stated by the governing methodology.
- **INTEGRATION_OPERATIONALIZATION** - introduced by the integration layer because the methodology does not specify within-dimension aggregation. These are choices of this layer and are NOT presented as methodology findings.

## 4. Evidence Interpretation Trail

| Evidence ID | Pillar | Native status | Mappability | Normalized |
|---|---|---|---|---|
| `EV-CE1-INGESTED` | CE1 | `INCONCLUSIVE` | DIRECTLY_MAPPABLE | INCONCLUSIVE |
| `EV-CE2-RUN1-INGESTED` | CE2 | `NOT_SUPPORTED` | DIRECTLY_MAPPABLE | NOT_SUPPORTED |
| `EV-CE2-RUN2-INGESTED` | CE2 | `NOT_SUPPORTED` | DIRECTLY_MAPPABLE | NOT_SUPPORTED |
| `EV-CE2-RUN3-INGESTED` | CE2 | `NOT_SUPPORTED` | DIRECTLY_MAPPABLE | NOT_SUPPORTED |
| `EV-CE2-RUN4-INGESTED` | CE2 | `NOT_SUPPORTED` | DIRECTLY_MAPPABLE | NOT_SUPPORTED |
| `EV-CE2-RUN5-INGESTED` | CE2 | `NOT_SUPPORTED` | DIRECTLY_MAPPABLE | NOT_SUPPORTED |
| `EV-CE3-INGESTED` | CE3 | `SUPPORTED` | DIRECTLY_MAPPABLE | SUPPORTED |
| `EV-CE4-INGESTED` | CE4 | `SUPPORTED` | DIRECTLY_MAPPABLE | SUPPORTED |
| `EV-D1-INGESTED` | D1 | `PASS` | DIRECTLY_MAPPABLE | PASS |
| `EV-D2-INGESTED` | D2 | `INCONCLUSIVE` | DIRECTLY_MAPPABLE | INCONCLUSIVE |
| `EV-D3-INGESTED` | D3 | `DESCRIPTIVE_ONLY` | UNMAPPABLE | - |
| `EV-D4-INGESTED` | D4 | `EFFECT_DETECTED` | UNMAPPABLE | - |
| `EV-D5-ABSENT` | D5 | `NOT_PRODUCED` | NOT_PRODUCED | - |
| `EV-EV-ABSENT` | EV | `NOT_PRODUCED` | NOT_PRODUCED | - |
| `EV-II-DEPTH-001` | II | `INCONCLUSIVE` | DIRECTLY_MAPPABLE | INCONCLUSIVE |

Native statuses shown as `-` under *Normalized* were deliberately NOT coerced into the standardized vocabulary; see each record's rationale in `pillar_interpretation.json`.

## 5. Conflict Resolution

### `CONFLICT-CE2-CE34-V1` - status: **REQUIRES_ADDITIONAL_EXPERIMENTATION** (blocks claim: True)

**Shared subject:** The same analytical target quantity (Lipmaa-Moriai chained single-trail differential probability), verified identical across CE2/CE3/CE4 by source: CE4 reuses CE3's primary task object literally, and CE3's target and CE2's reference both read TheoryDataset.theoretical_probabilities.

| Source | Decision | Observation |
|---|---|---|
| CE2 | NOT_SUPPORTED | Model output does NOT positively rank-correlate with the analytical target; the observed correlation is negative. |
| CE3 | SUPPORTED | The analytical target IS selectively decodable from the trained model's hidden representation relative to a control representation. |
| CE4 | SUPPORTED | Perturbing the analytical structure changes the model's output more than a magnitude-matched, target-preserving control perturbation. |

**Resolution rationale:** Both the contradicting and the supporting observations are RETAINED. The conflict is NOT resolved by majority vote (two SUPPORTED against one NOT_SUPPORTED), which methodology 3.8.8 prohibits and which would here amount to discarding the only observation contradicting the desired conclusion. It is also not resolved by methodological quality alone: although CE3 has the strongest design (replicate-level unit, calibration gate, multiplicity correction), a stronger design for a DIFFERENT measurement does not overturn a weaker design's finding about its own measurement. Per 3.8.8 the conflict therefore stands as requiring additional controlled experimentation, and the affected claim cannot be certified while it stands.

**Additional experimentation required:**
- Re-execute CE2, CE3 and CE4 against a reference-conformant (depth=10) model so that all three observations describe the declared protocol rather than a divergent artifact.
- Test the measurement-target hypothesis directly: determine whether the output-agreement/representational-decodability divergence persists under a conformant model, which would distinguish a genuine dissociation from an artifact of this instance.

## 6. Evidence Ledger

- Native entries: **1** (full provenance, resolvable run manifests)
- Ingested references: **12** (historical artifacts cited by content hash)
- Total: **13**

| Evidence ID | Kind | Pillar/Dim | Recorded decision |
|---|---|---|---|
| `EV-II-DEPTH-001` | NATIVE | II-CORE-GOHR-5R | INCONCLUSIVE |
| `EV-D1-INGESTED` | INGESTED_REFERENCE | D1 | PASS |
| `EV-D2-INGESTED` | INGESTED_REFERENCE | D2 | INCONCLUSIVE |
| `EV-D3-INGESTED` | INGESTED_REFERENCE | D3 | DESCRIPTIVE_ONLY |
| `EV-D4-INGESTED` | INGESTED_REFERENCE | D4 | EFFECT_DETECTED |
| `EV-CE1-INGESTED` | INGESTED_REFERENCE | CE1 | INCONCLUSIVE |
| `EV-CE2-RUN1-INGESTED` | INGESTED_REFERENCE | CE2 | NOT_SUPPORTED |
| `EV-CE2-RUN2-INGESTED` | INGESTED_REFERENCE | CE2 | NOT_SUPPORTED |
| `EV-CE2-RUN3-INGESTED` | INGESTED_REFERENCE | CE2 | NOT_SUPPORTED |
| `EV-CE2-RUN4-INGESTED` | INGESTED_REFERENCE | CE2 | NOT_SUPPORTED |
| `EV-CE2-RUN5-INGESTED` | INGESTED_REFERENCE | CE2 | NOT_SUPPORTED |
| `EV-CE3-INGESTED` | INGESTED_REFERENCE | CE3 | SUPPORTED |
| `EV-CE4-INGESTED` | INGESTED_REFERENCE | CE4 | SUPPORTED |

## 7. Artifact Index

- Artifacts tracked: **18**
- Verification: {'VERIFIED': 18}
- Hash conflicts: none
- Evidence bundle: `CIPHERMIND-EVIDENCE-BUNDLE-V1` (14 vendored files, byte-identical to their original sources)

## 8. Reproducibility and Integrity

- **artifact_index_hash_conflicts**: []
- **ledger_artifact_hash_problems**: []
- **ledger_artifact_hash_reverification**: ALL_VERIFIED

Snapshot component bindings (each document bound to the snapshot by content hash):

- `pillar_interpretation` -> `pillar_interpretation.json` (029dabb3adc90250...)
- `evidence_ledger` -> `evidence_ledger.json` (5035345260cf8b3a...)
- `artifact_index` -> `artifact_index.json` (4a6e660ce5495b26...)
- `decision` -> `decision.json` (2bd5a1ed0289c5e0...)

## 9. Limitations

- No Experimental Validity production evidence exists yet.
- Dataset Integrity is not fully established: D2 is INCONCLUSIVE and explicitly records that it is not proof of independence; D5 has smoke evidence only.
- All CE1-CE4 evidence was produced against a depth-5 checkpoint that diverges from the declared reference protocol's depth=10 (II-FINDING-DEPTH-V1).
- No independent convergence source exists, so LEVEL_4 is unreachable on the current evidence - this reflects absent convergence evidence, not a policy cap.
- Gohr's 11/12-round key-recovery apparatus (key_rank.py, test_key_recovery.py, key_averaging.py and released weights) exists in the repository but is NOT part of this claim chain. No CE1-CE4 experiment exercises it, and no evidence chain currently supports a key-recovery claim. It is recorded here solely so its exclusion is explicit rather than accidental.

## 10. Open Blockers

- CONFLICT-CE2-CE34-V1 unresolved; blocks claim C1.
- CE1-CE4 evidence bound to a non-conformant depth-5 checkpoint.
- No Experimental Validity production evidence (epsilon, replicate count, runtime benchmark all unresolved).
- Dataset Integrity incomplete: D2 INCONCLUSIVE, D5 smoke-only.
- No independent convergence source, so LEVEL_4 is unreachable on current evidence.

## 11. Evidential Scope Statement

Per methodology Section 3.12, this audit evaluates the **evidential support** for the claim above. It does not establish mathematical security properties, does not prove the underlying cryptographic hypothesis true or false, and does not certify software correctness. An INCONCLUSIVE outcome means the available evidence is insufficient to decide - not that the claim is false.

CE mappings, per-experiment hypotheses, controls and limitations are recorded in `decision.json` (4 CE experiments mapped).
