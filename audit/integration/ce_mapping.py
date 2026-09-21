"""
Source-verified mapping of the existing CE1-CE4 tests.

LOCATION: audit/integration/. This is DESCRIPTIVE metadata about the
Cryptographic Evidence pillar consumed by the integration layer. The CE
pillar itself - its code, drivers, checkpoints and certificates - lives
in audit/cryptography/ and is not modified by this module. (Previously
housed in a separate audit/cryptographic/ package, which duplicated the
pillar's name; relocated so there is exactly one CE package.)

This module DESCRIBES the CE pillar's existing, preserved experiments;
it does not reimplement, rerun, or redesign them. Every field below was
established by direct inspection of the CE source and evidence
artifacts, not from documentation or naming.

PRIMARY SCOPE (unchanged): the 5-round Gohr/Speck32/64 neural
distinguisher under differential (0x0040, 0x0000). The separate
11/12-round key-recovery endpoint is NOT part of this claim chain and
is represented only as an explicitly out-of-scope note.

VERIFIED FACTS worth recording, because they correct assumptions that
naming alone would produce:

  * All four CE drivers instantiate GohrDataset(rounds=5,
    differential=(0x0040, 0x0000)). CE1 is NOT a 7-round experiment;
    that earlier belief was an inference from its low baseline_score
    and is refuted by source.
  * CE1 TRAINS its own baseline and signal-destroyed models via
    adapter.train(). CE2/CE3/CE4 LOAD the frozen checkpoint
    "best5depth10 (10).h5".
  * That checkpoint's realized architecture is depth-5 (11 Conv1D
    layers), not the depth-10 its filename asserts - see the
    Implementation Integrity finding II-FINDING-DEPTH-V1.
  * CE2's "theoretical reference", CE3's primary TargetSpecification,
    and CE4's primary InterventionTask are THE SAME QUANTITY. CE4's
    adapter reuses CE3's primary task object literally; CE3's target
    and CE2's reference both read
    TheoryDataset.theoretical_probabilities. Their agreement is
    therefore structural, not coincidental.
  * That quantity is the Lipmaa-Moriai closed-form modular-addition
    XOR differential probability, chained along each sample's realized
    round-by-round trajectory under a Markov/independence assumption,
    and interpreted as a SINGLE-TRAIL quantity. It is NOT the exact
    differential probability of the cipher, NOT the full trail
    probability, and NOT established to be "what the network learned".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CEMapping:
    ce_id: str
    test_name: str
    claim: str
    hypothesis: str
    cryptographic_mechanism: str
    control: str
    manipulated_variable: str
    dependent_variable: str
    data_source: str
    statistical_method: str
    statistical_unit: str
    evidence_artifact: str
    recorded_decision: str
    rounds: int
    differential: tuple[int, int]
    model_provenance: str
    limitations: list[str] = field(default_factory=list)
    open_specification_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ce_id": self.ce_id, "test_name": self.test_name, "claim": self.claim,
            "hypothesis": self.hypothesis,
            "cryptographic_mechanism": self.cryptographic_mechanism,
            "control": self.control, "manipulated_variable": self.manipulated_variable,
            "dependent_variable": self.dependent_variable, "data_source": self.data_source,
            "statistical_method": self.statistical_method,
            "statistical_unit": self.statistical_unit,
            "evidence_artifact": self.evidence_artifact,
            "recorded_decision": self.recorded_decision,
            "rounds": self.rounds, "differential": list(self.differential),
            "model_provenance": self.model_provenance,
            "limitations": list(self.limitations),
            "open_specification_gaps": list(self.open_specification_gaps),
        }


SHARED_TARGET_QUANTITY = (
    "Lipmaa-Moriai closed-form modular-addition XOR differential probability, chained along "
    "each sample's realized round-by-round trajectory under a Markov/independence assumption, "
    "interpreted as a single-trail quantity. NOT the cipher's exact differential probability, "
    "NOT the full trail probability, and NOT established to be the quantity the network learned."
)

DEPTH5_CHECKPOINT = (
    "Loads frozen checkpoint 'best5depth10 (10).h5' "
    "(sha256 de630afcf95e7d0d40c33ff70623a7dda9290c6ae7e80133323ceb80997ad2ea), whose REALIZED "
    "architecture is depth-5 despite its filename - see II-FINDING-DEPTH-V1."
)

CE1 = CEMapping(
    ce_id="CE1", test_name="Signal Destruction",
    claim="Observed distinguishing performance reflects signal in the ciphertext pair, not an artifact.",
    hypothesis="Destroying the differential signal should collapse performance toward chance.",
    cryptographic_mechanism="Signal destruction applied to the dataset generation.",
    control="Baseline model trained on the intact dataset.",
    manipulated_variable="Presence/absence of the differential signal in the training data.",
    dependent_variable="Distinguishing accuracy.",
    data_source="Freshly generated 5-round Speck32/64 datasets.",
    statistical_method="Relative performance-drop threshold (0.05); no inferential test.",
    statistical_unit="One trained model per arm (single run, not replicated).",
    evidence_artifact="evidence/ce1/ce1_certificate.json",
    recorded_decision="INCONCLUSIVE",
    rounds=5, differential=(0x0040, 0x0000),
    model_provenance=(
        "TRAINS its own baseline and signal-destroyed models via adapter.train(). The default "
        "GohrModel depth (5) is never overridden to the reference protocol's depth=10."
    ),
    limitations=[
        "baseline_score=0.6108 is far below the reference protocol's reported 0.9291. The "
        "Implementation Integrity finding identifies the depth-5/depth-10 divergence as a "
        "confirmed candidate explanation, but causal attribution is UNDEMONSTRATED.",
        "Single run per arm: no replication, therefore no uncertainty estimate and no "
        "inferential statistic. A threshold comparison on two point estimates cannot "
        "distinguish a real effect from run-to-run variation.",
    ],
    open_specification_gaps=[
        "No preregistered practical-significance justification for the 0.05 relative-difference "
        "threshold.",
    ],
)

CE2 = CEMapping(
    ce_id="CE2", test_name="Theory Consistency",
    claim="The model's outputs track the analytically derived single-trail probability.",
    hypothesis="Model output and the theoretical reference should be positively rank-correlated.",
    cryptographic_mechanism=SHARED_TARGET_QUANTITY,
    control="None (observational correlation, no intervention or control arm).",
    manipulated_variable="None - observational.",
    dependent_variable="Rank correlation between model output and the theoretical reference.",
    data_source="Theory dataset of 1e5 samples.",
    statistical_method="Rank correlation with p-value against a 0.1 threshold.",
    statistical_unit="Per-sample (n=1e5) within a single model instance.",
    evidence_artifact="evidence/ce2/ce2_certificate_{1..5}.json (5 independent runs)",
    recorded_decision="NOT_SUPPORTED",
    rounds=5, differential=(0x0040, 0x0000),
    model_provenance=DEPTH5_CHECKPOINT,
    limitations=[
        "Observational: a correlation with no control arm cannot separate the declared "
        "mechanism from any covarying quantity.",
        "Per-sample statistical unit within ONE model instance: the result characterizes that "
        "instance, not the population of models the protocol would produce.",
    ],
)

CE3 = CEMapping(
    ce_id="CE3", test_name="Representation Interpretation",
    claim="The trained model's hidden representation encodes the declared quantity beyond a control representation.",
    hypothesis="The declared target is more decodable from the trained model's representation than from the control's.",
    cryptographic_mechanism=SHARED_TARGET_QUANTITY,
    control="Adapter-supplied control model representation, plus an independent calibration target that gates the decision.",
    manipulated_variable="Which representation is probed (trained model vs control model).",
    dependent_variable="Control-normalized decodability (selectivity).",
    data_source="Per-replicate independently generated evaluation datasets.",
    statistical_method="Paired significance across replicates, Bonferroni-corrected alpha=0.025 (m=2), calibration-gated.",
    statistical_unit="Independent evaluation replicate (n=20); CV folds averaged within replicate, NOT treated as independent.",
    evidence_artifact="evidence/ce3/ce3_certificate.json",
    recorded_decision="SUPPORTED",
    rounds=5, differential=(0x0040, 0x0000),
    model_provenance=DEPTH5_CHECKPOINT,
    limitations=[
        "Decodability is not mechanism: a quantity being linearly recoverable from a "
        "representation does not establish that the model USES it to produce its output.",
        "Scoped to the depth-5 checkpoint instance, not to the declared reference protocol.",
    ],
    open_specification_gaps=[
        "The driver passes supported/inconclusive thresholds (0.20/0.05) that the evaluator's "
        "CE3 branch never reads - a code-clarity defect recorded as "
        "II-FINDING-CE3-DEADPARAMS-V1. It does not affect the recorded decision.",
    ],
)

CE4 = CEMapping(
    ce_id="CE4", test_name="Causal Intervention",
    claim="The model's output causally depends on the declared structure, not on perturbation magnitude alone.",
    hypothesis="A targeted structural perturbation should change the output more than a magnitude-matched control perturbation.",
    cryptographic_mechanism=SHARED_TARGET_QUANTITY,
    control=(
        "Magnitude-matched mirrored-pair perturbation drawn from the SAME eligible pool, which "
        "provably preserves the XOR difference at every intervened position via the identity "
        "(a^1)^(b^1) == a^b. Inert with respect to the DECLARED target only - it still alters "
        "32 raw input bits."
    ),
    manipulated_variable="Single-sided (structural) vs mirrored (control) bit perturbation.",
    dependent_variable="Absolute output-probability difference; necessity gap = targeted - control.",
    data_source="CE3's primary task dataset, reused literally.",
    statistical_method="Wilcoxon signed-rank with bootstrap CI; thresholds scaled to an empirically calibrated all-bits-flipped output-change ceiling.",
    statistical_unit="Per-sample necessity gap (n=99,748 after 0.252% exclusion) within ONE model instance.",
    evidence_artifact="evidence/ce4/ce4_certificate.json",
    recorded_decision="SUPPORTED",
    rounds=5, differential=(0x0040, 0x0000),
    model_provenance=DEPTH5_CHECKPOINT,
    limitations=[
        "Per-sample unit within a single model instance: establishes intervention-sensitive "
        "dependence FOR THAT MODEL, not a property of models the protocol produces. p-value "
        "underflowed to 0.0 at n~1e5; the effect size (d_z=0.545) is the informative quantity.",
        "Population narrowed by excluding 252/100000 samples whose realized trail could not "
        "support the intervention magnitude. Disclosed in the certificate (verified).",
        "'Causal necessity' is stronger than what is established: the evidence supports "
        "intervention-sensitive dependence on the declared structure within the tested "
        "population and model instance.",
    ],
)

ALL_CE_MAPPINGS = [CE1, CE2, CE3, CE4]

KEY_RECOVERY_SCOPE_NOTE = (
    "Gohr's 11/12-round key-recovery apparatus (key_rank.py, test_key_recovery.py, "
    "key_averaging.py and released weights) exists in the repository but is NOT part of this "
    "claim chain. No CE1-CE4 experiment exercises it, and no evidence chain currently supports "
    "a key-recovery claim. It is recorded here solely so its exclusion is explicit rather than "
    "accidental."
)
