#!/usr/bin/env python3
"""
II-3 paper-to-source architecture comparison for AutoND / DBitNet.

WHAT THIS IS
    A claim-by-claim comparison of the architecture the PUBLICATION
    describes against the architecture the PINNED SOURCE builds. It reads
    no checkpoint and runs no model; the census and load-verification
    instruments do that separately. Each row carries a paper locator and
    a source locator so a reviewer can check both sides.

VERDICT SEMANTICS
    AGREES              both sides state the property and they match.
    DIFFERS             both sides state it and they do not match.
    UNDERSPECIFIED      the paper does not pin the property down
                        precisely enough to compare. This is an
                        observation about the publication's reporting,
                        NOT a defect in the code and NOT a passed check.
    NOT_CHECKABLE_HERE  the property cannot be adjudicated with the
                        evidence this module has access to.

PAPER VERSION DISCIPLINE
    ePrint 2022/1467 has FOUR revisions. An earlier review in this
    project read a stale revision, asserted paper-derived head widths,
    and had to retract them. Every paper locator below therefore names
    the section or table in the LAST revision (2024-01-24, the version
    served at the canonical ePrint URL and published in IACR ToSC 2024),
    and a locator that could not be confirmed in that revision is
    recorded as UNDERSPECIFIED rather than assumed.

    The PDF's own sha256 is NOT recorded as verified: the build
    environment that produced this module cannot fetch eprint.iacr.org
    (egress is allow-listed), so the comparison was made against the
    document's extracted text, not against a locally hashed file. Use
    `verify_paper_pdf()` to bind the digest on a machine that holds the
    PDF.

REVISION 2 OF THIS COMPARISON
    Revision 1 reported 3 AGREES / 0 DIFFERS / 9 UNDERSPECIFIED /
    1 NOT_CHECKABLE_HERE. That summary was produced without the
    authoritative revision in hand and understated what the paper does
    specify. Three corrections drive the change, and all three are
    recorded in `CORRECTIONS` below so the earlier reading is not
    silently replaced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

COMPARISON_VERSION = "autond-paper-source-architecture-v2"

try:                                      # the pinned commit is Phase 1's
    from audit.targets.autond import PINNED_UPSTREAM_COMMIT as PINNED_COMMIT
except Exception:                                                 # noqa: BLE001
    PINNED_COMMIT = "8d66e0a43a9e6fbbb3624d68296c1b77aa2da779"


# --------------------------------------------------------------------------
# The publication being compared against
# --------------------------------------------------------------------------
PAPER = {
    "eprint_id": "2022/1467",
    "url": "https://eprint.iacr.org/2022/1467.pdf",
    "title": ("A Cipher-Agnostic Neural Training Pipeline with Automated "
              "Finding of Good Input Differences"),
    "authors": ("Emanuele Bellini, David Gerault, Anna Hambitzer, Matteo Rossi"),
    "venue": "IACR Transactions on Symmetric Cryptology (ToSC) 2024",
    "received": "2022-10-26",
    "last_revision": "2024-01-24",
    "n_revisions": 4,
    "revision_compared": ("last of 4 revisions, 2024-01-24 - the version served "
                          "at the canonical ePrint URL"),
    "pdf_sha256": None,
    "pdf_sha256_status": (
        "UNVERIFIED. The environment that built this comparison cannot reach "
        "eprint.iacr.org, so the PDF was never hashed locally. The comparison "
        "was made against the document's extracted text. Run "
        "verify_paper_pdf(path) on a machine holding the PDF to bind the digest."),
    "text_extraction_limits": (
        "Figure 4c, captioned 'The actual design of DBitNet', carries no "
        "machine-extractable internal text. Any property that the paper states "
        "ONLY inside Figure 4c is therefore recorded UNDERSPECIFIED here and may "
        "resolve against the rendered figure."),
}


#: The source files this comparison's source_locators actually cite,
#: with their digests at the pinned commit. An AGREES verdict is a claim
#: about THIS source; if the file on disk is a different file, the claim
#: has not been checked against anything.
SOURCE_FILES = {
    "dbitnet.py":
        "b19adceb6e6a9ad39c648aa41e58d23b6a0b338c9bd2be9c4b0c7799ea631736",
    "eval_nets.py":
        "838587127c08834f6f9ecd7b27fc578da3942141fdcf5d9c513d46c646d63048",
    "train_nets.py":
        "f38be25f879df0ece662cea78d8dd6a78da3b9b076a4b83bc22f2cb25e53ec8e",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _head_commit(upstream: Path) -> tuple[Optional[str], Optional[str]]:
    import subprocess
    try:
        res = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
    except Exception as exc:                                        # noqa: BLE001
        return None, f"git could not be run: {type(exc).__name__}: {exc}"
    if res.returncode != 0:
        return None, f"git rev-parse failed: {res.stderr.strip()}"
    return res.stdout.strip(), None


def verify_sources(upstream: Optional[Path]) -> dict:
    """
    Confirm the cited source files are the pinned ones, before any AGREES
    verdict is allowed to stand.

    A paper-to-source comparison is a claim about a specific source. If
    the checkout is absent, at the wrong commit, or holds a different
    dbitnet.py, then nothing was compared and the AGREES rows are not
    findings - they are assertions about a file this process never read.
    `effective_summary()` downgrades them in that case.

    Returns {"verified", "problems", "commit", "files"}.
    """
    out: dict = {"verified": False, "problems": [], "commit": None,
                 "files": {}, "upstream": str(upstream) if upstream else None}
    if upstream is None:
        out["problems"].append(
            "no upstream checkout supplied; the cited source files were not "
            "verified, so no paper-to-source agreement is claimed")
        return out
    up = Path(upstream).expanduser()
    if not up.is_dir():
        out["problems"].append(f"no checkout at {up}")
        return out

    commit, err = _head_commit(up)
    out["commit"] = commit
    if err:
        out["problems"].append(err)
    elif commit != PINNED_COMMIT:
        out["problems"].append(
            f"checkout HEAD is {commit}, not the pinned {PINNED_COMMIT}. The "
            "source compared against the paper must be the pinned revision.")

    for rel, expected in sorted(SOURCE_FILES.items()):
        f = up / rel
        rec = {"expected_sha256": expected, "observed_sha256": None,
               "present": f.is_file(), "verified": False}
        if not f.is_file():
            out["problems"].append(f"{rel} is missing from {up}")
        else:
            got = _sha256(f)
            rec["observed_sha256"] = got
            if got == expected:
                rec["verified"] = True
            else:
                out["problems"].append(
                    f"{rel}: sha256 {got} != pinned {expected}. The file cited "
                    "by this comparison's source locators is not the file that "
                    "was compared.")
        out["files"][rel] = rec

    out["verified"] = not out["problems"]
    return out


def verify_paper_pdf(path: Path) -> dict:
    """
    Hash a local copy of the paper so a reviewer can bind the exact file.

    Returns a record; it never mutates PAPER. The caller decides whether
    the digest it reports becomes the bound value.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return {"path": str(p), "present": False,
                "note": "no file at this path; nothing was hashed"}
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    st = p.stat()
    head = p.read_bytes()[:8]
    return {
        "verifier": "paper_source_architecture.verify_paper_pdf",
        "verified_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "path": str(p),
        "file_name": p.name,
        "present": True,
        "sha256": h.hexdigest(),
        "bytes": st.st_size,
        "looks_like_pdf": head.startswith(b"%PDF-"),
        "pdf_header": head.decode("latin-1", "replace"),
        "eprint_id_expected": PAPER["eprint_id"],
        "revision_asserted": False,
        "revision_note": (
            "The revision is NOT asserted from this digest or from the file "
            "name: a file called 2022-1467.pdf may be any of the four "
            "revisions, and the digest alone does not say which. Confirm the "
            "revision date on the title page, then record it yourself. This "
            "record establishes WHICH FILE was hashed, not which revision it "
            "is."),
    }


# --------------------------------------------------------------------------
# What changed since revision 1 of this comparison, and why
# --------------------------------------------------------------------------
CORRECTIONS = [
    {
        "claim_id": "ARCH-03",
        "was": "UNDERSPECIFIED",
        "now": "AGREES",
        "why": ("The dilation schedule IS specified. Section 6.2 gives the rule "
                "('dividing the input size by two and subtracting one', 'until a "
                "minimum value of 3 is reached') and Table 3 gives the realised "
                "values [31, 15, 7, 3] for SIMON32 and SPECK32. Revision 1 did "
                "not locate either."),
        "locator_correction": (
            "The dilation rates are in TABLE 3 ('Settings for Gohr's neural "
            "network and DBitNet'), NOT Table 4. Table 4 is 'FLOPs, parameters "
            "and runtime per epoch'. A review brief given to this module cited "
            "Table 4; that citation is wrong and is corrected here rather than "
            "propagated."),
    },
    {
        "claim_id": "ARCH-06",
        "was": "UNDERSPECIFIED",
        "now": "AGREES",
        "why": ("Section 6.2 states the filter progression explicitly: 'We start "
                "with 32 filters, identical to Gohr, in the first convolutional "
                "layer' and 'Whenever the neuronal width is halved, we add 16 "
                "filters, resulting in 32 + i x 16 filters in the ith dilated "
                "convolution.' Revision 1 did not locate this."),
    },
    {
        "claim_id": "ARCH-12",
        "was": "UNDERSPECIFIED",
        "now": "AGREES",
        "why": ("Table 4 reports DBitNet parameters for SIMON32 and SPECK32 as "
                "298.11k, which the census's 298,113 matches to the precision "
                "the table prints. Revision 1 treated the parameter count as "
                "unlocated."),
    },
    {
        "claim_id": "ARCH-09",
        "was": "NOT_CHECKABLE_HERE",
        "now": "UNDERSPECIFIED",
        "why": ("The SOURCE side is fully checkable and is now recorded exactly: "
                "dbitnet.py sets d1 = 256, d2 = 64 and builds Dense(256), "
                "Dense(256), Dense(64), Dense(1, sigmoid). It is the PAPER side "
                "that is absent, which makes this UNDERSPECIFIED rather than "
                "uncheckable."),
        "contested_claim": (
            "A review brief given to this module asserted that Section 6.2 "
            "specifies DBitNet prediction-head sizes 256, 256, 64. Three "
            "independent extractions of the 2024-01-24 revision do not support "
            "that: the phrase 'prediction head' occurs once, in Section 3.2, "
            "describing GOHR's network; the only dense-layer size the paper "
            "states is Gohr's 64 neurons in Section 4.2; and the number 256 "
            "occurs in the paper only as SPECK's largest key size, as Gohr's "
            "256 filters / 256 neurons in Section 4.2, as the input size of "
            "LEA/SIMON128/SPECK128 in Table 3, and in HIGHT's C.256 data "
            "requirement in Section 7. The values 256, 256, 64 are therefore "
            "NOT asserted here to be the paper's. They may appear inside "
            "Figure 4c, whose internal text is not machine-extractable; if the "
            "rendered figure shows them, this row flips to AGREES with a figure "
            "locator. Given that an earlier review in this project retracted "
            "paper-derived head widths after reading a stale revision, this row "
            "is left unresolved rather than asserted in either direction."),
    },
]


class Verdict(str, Enum):
    AGREES = "AGREES"
    DIFFERS = "DIFFERS"
    UNDERSPECIFIED = "UNDERSPECIFIED"
    NOT_CHECKABLE_HERE = "NOT_CHECKABLE_HERE"


class Basis(str, Enum):
    SOURCE_CODE = "SOURCE_CODE"
    WEIGHTS_FILE = "WEIGHTS_FILE"
    BOTH = "BOTH"
    PAPER_ONLY = "PAPER_ONLY"


@dataclass(frozen=True)
class ArchitectureClaim:
    claim_id: str
    property_name: str
    paper_says: str
    paper_locator: str
    source_does: str
    source_locator: str
    verdict: Verdict
    evidence_basis: Basis
    note: str = ""
    census_corroborates: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["evidence_basis"] = self.evidence_basis.value
        return d


#: Input size for Speck32/64 and Simon32/64: both are 32-bit block
#: ciphers, so eval_nets/train_nets call make_model(2 * plain_bits) = 64.
INPUT_SIZE = 64

#: Dilation rates the pinned source derives for INPUT_SIZE, recomputed
#: here by the same arithmetic dbitnet.get_dilation_rates uses, so the
#: expected value is not a transcribed constant.
def _source_dilation_rates(input_size: int = INPUT_SIZE) -> list[int]:
    drs, n = [], input_size
    while n >= 8:
        drs.append(int(n / 2 - 1))
        n = n // 2
    return drs


#: Realised neuronal widths: a kernel-2 'valid' convolution at dilation d
#: maps length L to L - d; the intervening d=1 convolution is 'causal'
#: and preserves length.
def _source_width_sequence(input_size: int = INPUT_SIZE) -> list[int]:
    widths, L = [input_size], input_size
    for d in _source_dilation_rates(input_size):
        L = L - d
        widths.append(L)
    return widths


#: Filters per block: 32 + i * 16.
def _source_filter_schedule(input_size: int = INPUT_SIZE) -> list[int]:
    return [32 + i * 16 for i in range(len(_source_dilation_rates(input_size)))]


DILATION_RATES = _source_dilation_rates()
WIDTH_SEQUENCE = _source_width_sequence()
FILTER_SCHEDULE = _source_filter_schedule()


CLAIMS: tuple[ArchitectureClaim, ...] = (
    ArchitectureClaim(
        claim_id="ARCH-01",
        property_name="Model family used for the reported distinguishers",
        paper_says=("DBitNet, a cipher-agnostic dilated-convolutional network, is "
                    "the architecture introduced and used for the reported "
                    "distinguishers alongside Gohr's network as a baseline."),
        paper_locator="Section 6.2 ('Rationale for DBitNet'); Tables 3, 4 and 6",
        source_does=("dbitnet.make_model builds the dilated-convolutional network; "
                     "train_nets/eval_nets select it by model_type 'dbitnet'."),
        source_locator="dbitnet.py:10 make_model; train_nets.py, eval_nets.py",
        verdict=Verdict.AGREES,
        evidence_basis=Basis.BOTH,
    ),
    ArchitectureClaim(
        claim_id="ARCH-02",
        property_name="Cipher-agnostic sizing by block width",
        paper_says=("The architecture is sized from the cipher's input size; "
                    "Table 3 lists input size 64 for SIMON32 and SPECK32."),
        paper_locator="Section 6.2 ('Neural Network Settings'); Table 3, input-size column",
        source_does=(f"make_model(input_size) derives everything from input_size; "
                     f"for a 32-bit block cipher the callers pass "
                     f"2 * plain_bits = {INPUT_SIZE}."),
        source_locator="dbitnet.py:10 make_model(input_size=64); dbitnet.py:23",
        verdict=Verdict.AGREES,
        evidence_basis=Basis.BOTH,
    ),
    ArchitectureClaim(
        claim_id="ARCH-03",
        property_name="Dilation schedule",
        paper_says=("Rule: 'for DBitNet the dilation rates are given by dividing "
                    "the input size by two and subtracting one', continuing 'until "
                    "a minimum value of 3 is reached'. Realised values for SIMON32 "
                    "and SPECK32: [31, 15, 7, 3]."),
        paper_locator=("Section 6.2 ('Neural Network Settings') for the rule; "
                       "Table 3, DBitNet dilation-rates column, SIMON32 and "
                       "SPECK32 rows, for the values"),
        source_does=(f"get_dilation_rates appends int(n/2 - 1) while n >= 8, "
                     f"halving n each step, giving {DILATION_RATES} for "
                     f"input_size={INPUT_SIZE}."),
        source_locator="dbitnet.py:1-8 get_dilation_rates; applied at dbitnet.py:23",
        verdict=Verdict.AGREES,
        evidence_basis=Basis.BOTH,
        note=("Both the rule and the realised values match. The source's loop "
              "guard (n >= 8) is what produces the paper's stated minimum of 3, "
              "since the last admitted n is 8 and 8/2 - 1 = 3. The dilation "
              "rates are in Table 3, not Table 4; see CORRECTIONS."),
        census_corroborates=("the census records dilations [31, 15, 7, 3] as "
                             "derived from the source, not read from the weights "
                             "file, which cannot carry dilation semantics"),
    ),
    ArchitectureClaim(
        claim_id="ARCH-04",
        property_name="Convolution kernel size",
        paper_says=("No kernel size is stated for DBitNet's actual layers. The "
                    "only kernel size in the DBitNet discussion is illustrative: "
                    "'If we apply such a dilated convolutional layer with dilation "
                    "d = 8 and kernel size k = 2', presented as an example of how "
                    "dilation works rather than as the configuration used."),
        paper_locator="Section 6.2, illustrative example; no normative statement located",
        source_does="Both convolutions in every block use kernel_size=2.",
        source_locator="dbitnet.py:41 (dilated conv), dbitnet.py:50 (d=1 conv)",
        verdict=Verdict.UNDERSPECIFIED,
        evidence_basis=Basis.SOURCE_CODE,
        note=("The illustrative k = 2 coincides with the source's kernel size, "
              "but an example is not a specification, so this is recorded as "
              "unlocated rather than as agreement. Reading the example as "
              "normative would be the same error that produced the retracted "
              "head-width claim."),
    ),
    ArchitectureClaim(
        claim_id="ARCH-05",
        property_name="Padding mode",
        paper_says="Not stated for DBitNet in the text.",
        paper_locator=("Section 6.2 - no padding statement located; Figure 4c "
                       "internal text not machine-extractable"),
        source_does=("The dilated convolution uses padding='valid'; the "
                     "following d=1 convolution uses padding='causal'."),
        source_locator="dbitnet.py:42 padding='valid'; dbitnet.py:50 padding='causal'",
        verdict=Verdict.UNDERSPECIFIED,
        evidence_basis=Basis.SOURCE_CODE,
        note=("The two padding modes differ between the two convolutions in a "
              "block, which is the mechanism behind the realised width sequence "
              "in ARCH-16. A weights-only HDF5 cannot carry padding, so the "
              "census cannot corroborate either side."),
    ),
    ArchitectureClaim(
        claim_id="ARCH-06",
        property_name="Filter progression across blocks",
        paper_says=("'We start with 32 filters, identical to Gohr, in the first "
                    "convolutional layer.' and 'Whenever the neuronal width is "
                    "halved, we add 16 filters, resulting in 32 + i x 16 filters "
                    "in the ith dilated convolution.'"),
        paper_locator="Section 6.2 ('Rationale for DBitNet')",
        source_does=(f"make_model defaults n_filters=32, n_add_filters=16 and "
                     f"executes n_filters += n_add_filters at the end of each "
                     f"block, giving {FILTER_SCHEDULE} across the "
                     f"{len(DILATION_RATES)} blocks."),
        source_locator=("dbitnet.py:10 defaults n_filters=32, n_add_filters=16; "
                        "dbitnet.py:56 n_filters += n_add_filters"),
        verdict=Verdict.AGREES,
        evidence_basis=Basis.BOTH,
        note=("32 + i*16 is reproduced exactly. Note the paper attaches the "
              "increment to the neuronal width halving while the source attaches "
              "it to the block iteration; for this architecture the two coincide "
              "because each block contains exactly one dilated convolution."),
        census_corroborates=("the census's per-layer filter counts across the "
                             "shipped checkpoints follow 32, 48, 64, 80"),
    ),
    ArchitectureClaim(
        claim_id="ARCH-07",
        property_name="Residual connection inside each block",
        paper_says="Not stated for DBitNet in the text.",
        paper_locator=("Section 6.2 - no residual/skip statement located for "
                       "DBitNet; Figure 4c internal text not machine-extractable. "
                       "Residual connections are described for Gohr's network in "
                       "Section 3.2"),
        source_does=("Each block captures x_skip after the dilated convolution's "
                     "batch normalisation and adds it back after the d=1 "
                     "convolution: Add()([x, x_skip])."),
        source_locator="dbitnet.py:47 x_skip = x; dbitnet.py:53 Add()([x, x_skip])",
        verdict=Verdict.UNDERSPECIFIED,
        evidence_basis=Basis.SOURCE_CODE,
        note=("The source comments the pair as 'wide-narrow blocks'. The paper "
              "describes the wide-narrow idea in prose ('The next layer is a "
              "d = 1 layer to investigate the dependencies between neighboring "
              "bits') without stating that the pair is wired residually."),
    ),
    ArchitectureClaim(
        claim_id="ARCH-08",
        property_name="Input normalisation",
        paper_says="Not stated for DBitNet in the text.",
        paper_locator="Section 6.2 - no input-normalisation statement located",
        source_does=("The input is mapped to [-1, 1] by tf.subtract(x, 0.5) "
                     "followed by tf.divide(x, 0.5), before the first "
                     "convolution."),
        source_locator="dbitnet.py:35-36",
        verdict=Verdict.UNDERSPECIFIED,
        evidence_basis=Basis.SOURCE_CODE,
        note=("These two ops are why the model cannot be built under Keras 3, "
              "which rejects tf.subtract/tf.divide applied to a KerasTensor. "
              "That is a reproducibility constraint rather than an architecture "
              "disagreement."),
    ),
    ArchitectureClaim(
        claim_id="ARCH-09",
        property_name="Prediction head widths",
        paper_says=("Nothing located. The phrase 'prediction head' appears once, "
                    "in Section 3.2, describing GOHR's network ('a densely "
                    "connected prediction head with ReLU activations'). The only "
                    "dense-layer size the paper states is Gohr's 64 neurons in "
                    "Section 4.2. No DBitNet dense-layer size was located in the "
                    "text or in any table."),
        paper_locator=("Section 3.2 and Section 4.2 describe GOHR's head only. No "
                       "DBitNet head statement located in Sections 6.1-6.3, "
                       "Tables 3-6, or the text of Figure 4's caption. Figure 4c "
                       "internal text is not machine-extractable and is the one "
                       "place the paper might state it."),
        source_does=("d1 = 256 and d2 = 64 drive Dense(256) -> Dense(256) -> "
                     "Dense(64) -> Dense(1, sigmoid), each of the first three "
                     "followed by BatchNormalization and a ReLU Activation."),
        source_locator=("dbitnet.py:26-27 d1 = 256, d2 = 64; dbitnet.py:61-70 "
                        "the four Dense layers"),
        verdict=Verdict.UNDERSPECIFIED,
        evidence_basis=Basis.SOURCE_CODE,
        note=("The source side is exact and verified: 256, 256, 64, then a single "
              "sigmoid unit. The paper side is absent, so there is nothing to "
              "agree or disagree with. Two further observations. First, the "
              "source's own comment calls these 'prediction head parameters "
              "(similar to Gohr)' and carries a TODO saying d1 'can likely be "
              "reduced to 64' - so the similarity-to-Gohr claim lives in the "
              "code comment, not in the publication. Second, a review brief given "
              "to this module asserted these sizes are specified in Section 6.2; "
              "see CORRECTIONS for why that assertion is not adopted."),
        census_corroborates=("the census's dense-layer tensor shapes are "
                             "consistent with 640 -> 256 -> 256 -> 64 -> 1"),
    ),
    ArchitectureClaim(
        claim_id="ARCH-10",
        property_name="Output interpretation",
        paper_says=("The distinguisher's output is a probability thresholded at "
                    "0.5: accuracy is computed against Zbin = (Z > 0.5), with TPR "
                    "and TNR reported alongside. Gohr's network is stated to end "
                    "in 'an output layer with a single sigmoid neuron'."),
        paper_locator=("Section 3.2 for Gohr's single sigmoid output; Section 4.1 "
                       "and Table 6 for the accuracy/TPR/TNR definition applied "
                       "to both networks"),
        source_does="Final layer is Dense(1, activation='sigmoid').",
        source_locator="dbitnet.py:70; threshold applied in eval_nets.py:33 Zbin = (Z > 0.5)",
        verdict=Verdict.AGREES,
        evidence_basis=Basis.BOTH,
        note=("Scoped deliberately: what agrees is the OUTPUT INTERPRETATION - a "
              "single probability thresholded at 0.5, which the reported accuracy "
              "and TPR/TNR presuppose. The paper does not restate the output "
              "layer for DBitNet specifically; it states it for Gohr. This row "
              "therefore does not claim the paper specifies DBitNet's final "
              "layer."),
    ),
    ArchitectureClaim(
        claim_id="ARCH-11",
        property_name="Kernel regularisation",
        paper_says="Not stated for DBitNet in the text.",
        paper_locator=("Section 6.2 - no regularisation statement located for "
                       "DBitNet"),
        source_does=("Every Dense layer carries kernel_regularizer=l2(1e-5); the "
                     "convolutions carry none."),
        source_locator="dbitnet.py:28 reg_param = 1e-5; dbitnet.py:61-70",
        verdict=Verdict.UNDERSPECIFIED,
        evidence_basis=Basis.SOURCE_CODE,
        note=("The asymmetry is worth recording: regularisation is applied to the "
              "head only, not to the convolutional trunk. A weights file cannot "
              "carry regulariser settings."),
    ),
    ArchitectureClaim(
        claim_id="ARCH-12",
        property_name="Total parameter count",
        paper_says=("Table 4 reports DBitNet parameters for SIMON32 and SPECK32 "
                    "as 298.11k."),
        paper_locator=("Table 4 ('FLOPs, parameters and runtime per epoch'), "
                       "DBitNet parameters column, SIMON32 and SPECK32 rows"),
        source_does=("The model the pinned source builds for input_size=64 "
                     "realises 298,113 parameters (296,065 trainable + 2,048 "
                     "non-trainable), as the census records for both shipped "
                     "checkpoints."),
        source_locator=("dbitnet.py make_model(64) in full; corroborated by the "
                        "architecture census over both shipped checkpoints"),
        verdict=Verdict.AGREES,
        evidence_basis=Basis.BOTH,
        note=("298,113 printed to the precision Table 4 uses is 298.11k, so the "
              "realised count matches the published count exactly at the table's "
              "stated resolution. This is the strongest quantitative agreement in "
              "this comparison, because it is a single number that depends on "
              "every architectural choice at once: a wrong kernel size, filter "
              "schedule, head width or block count would change it."),
        census_corroborates=("both shipped checkpoints census to 34 layers, 68 "
                             "tensors and 298,113 parameters"),
    ),
    ArchitectureClaim(
        claim_id="ARCH-13",
        property_name="Batch normalisation placement",
        paper_says="Not stated for DBitNet in the text.",
        paper_locator=("Section 6.2 - no batch-normalisation statement located "
                       "for DBitNet. Batch normalisation is described for Gohr's "
                       "network in Section 3.2"),
        source_does=("Batch normalisation appears after the dilated convolution, "
                     "after the residual Add, and after each of the three wide "
                     "Dense layers - six placements in total for input_size=64 "
                     "plus per-block repetition."),
        source_locator="dbitnet.py:46, 54, 62, 65, 68",
        verdict=Verdict.UNDERSPECIFIED,
        evidence_basis=Basis.SOURCE_CODE,
        note=("The census CAN see batch-normalisation layers, because they carry "
              "four tensors each (gamma, beta, moving_mean, moving_variance); the "
              "2,048 non-trainable parameters are exactly the moving statistics. "
              "So the SOURCE side is corroborated by the weights file even though "
              "the paper side is unlocated."),
        census_corroborates=("2,048 non-trainable parameters = the moving_mean "
                             "and moving_variance tensors of the batch-norm layers"),
    ),
    ArchitectureClaim(
        claim_id="ARCH-14",
        property_name="Number of dilated blocks",
        paper_says=("Implied by the dilation rule and Table 3: four dilation "
                    "rates [31, 15, 7, 3] for a 64-bit input, hence four dilated "
                    "convolutions."),
        paper_locator="Section 6.2 (rule) with Table 3 (the four realised rates)",
        source_does=(f"The block loop iterates over get_dilation_rates(64), which "
                     f"returns {len(DILATION_RATES)} rates, so the model has "
                     f"{len(DILATION_RATES)} wide-narrow blocks."),
        source_locator="dbitnet.py:38 for dilation_rate in dilation_rates",
        verdict=Verdict.AGREES,
        evidence_basis=Basis.BOTH,
        note=("Recorded as its own row because the block count is what makes the "
              "filter schedule in ARCH-06 and the parameter count in ARCH-12 "
              "determinate. Table 3's 'blocks' and 'word size' columns describe "
              "the CIPHER's word structure (2 blocks of 16 bits for SPECK32/64), "
              "not the network's block count; conflating the two would be an "
              "error."),
    ),
    ArchitectureClaim(
        claim_id="ARCH-15",
        property_name="The d=1 convolution following each dilated convolution",
        paper_says=("'The next layer is a d = 1 layer to investigate the "
                    "dependencies between neighboring bits.'"),
        paper_locator="Section 6.2 ('Rationale for DBitNet')",
        source_does=("Each block's second convolution uses dilation_rate=1, "
                     "immediately after the dilated convolution and its batch "
                     "normalisation."),
        source_locator="dbitnet.py:48-52 Conv1D(..., dilation_rate=1)",
        verdict=Verdict.AGREES,
        evidence_basis=Basis.BOTH,
        note=("The paper's stated purpose for this layer (neighbouring-bit "
              "dependencies) is consistent with dilation 1, and the source places "
              "it exactly where the paper describes."),
    ),
    ArchitectureClaim(
        claim_id="ARCH-16",
        property_name="Neuronal width progression across blocks",
        paper_says=("Qualitative only: 'the neuronal width is shrinking with each "
                    "dilated convolution by a factor of two' (described with "
                    "reference to Figure 4b). No per-layer widths are tabulated."),
        paper_locator="Section 6.2, discussion of Figure 4b",
        source_does=(f"A kernel-2 'valid' convolution at dilation d maps length L "
                     f"to L - d, and the intervening 'causal' d=1 convolution "
                     f"preserves length, so the realised sequence for "
                     f"input_size={INPUT_SIZE} is "
                     f"{' -> '.join(str(w) for w in WIDTH_SEQUENCE)}."),
        source_locator=("dbitnet.py:40-45 with padding='valid'; dbitnet.py:48-52 "
                        "with padding='causal'"),
        verdict=Verdict.UNDERSPECIFIED,
        evidence_basis=Basis.SOURCE_CODE,
        note=("This is the most interesting reporting gap in the comparison and "
              "the reason it is not scored as agreement. 'A factor of two' is not "
              f"what the source realises: {WIDTH_SEQUENCE[0]} -> "
              f"{WIDTH_SEQUENCE[1]} is not a halving, and the drift compounds "
              "because the dilation rates are precomputed from the ORIGINAL input "
              "size while 'valid' padding subtracts d from the CURRENT length. "
              "Exact halving would give 64 -> 32 -> 16 -> 8 -> 4. The claim is "
              "recorded UNDERSPECIFIED rather than DIFFERS because the paper's "
              "statement is a qualitative description attached to a schematic "
              "figure, not a quantitative specification that could be falsified; "
              "a reviewer who reads it as quantitative should read this row as "
              "DIFFERS and the realised sequence is given above so they can."),
        census_corroborates=(f"the census records the width sequence "
                             f"{' -> '.join(str(w) for w in WIDTH_SEQUENCE)} and "
                             f"flatten width {WIDTH_SEQUENCE[-1] * FILTER_SCHEDULE[-1]}"),
    ),
)


def summarise() -> dict:
    counts: dict[str, int] = {}
    for c in CLAIMS:
        counts[c.verdict.value] = counts.get(c.verdict.value, 0) + 1
    return {
        "total_claims": len(CLAIMS),
        "by_verdict": counts,
        "agrees": [c.claim_id for c in CLAIMS if c.verdict is Verdict.AGREES],
        "differs": [c.claim_id for c in CLAIMS if c.verdict is Verdict.DIFFERS],
        "underspecified": [c.claim_id for c in CLAIMS
                           if c.verdict is Verdict.UNDERSPECIFIED],
        "not_checkable_here": [c.claim_id for c in CLAIMS
                               if c.verdict is Verdict.NOT_CHECKABLE_HERE],
    }


def effective_summary(source_verification: Optional[dict] = None) -> dict:
    """
    The summary that may actually be reported.

    `summarise()` reports the DECLARED verdicts - what the comparison
    concluded when it was written against the pinned source. This
    function reports what is defensible on the machine it runs on: if the
    cited source files were not verified, every AGREES becomes
    NOT_CHECKABLE_HERE, because an agreement with a file that was never
    read is not an observation.

    Revision 2 reported 8 AGREES unconditionally, including on a machine
    with no checkout at all.
    """
    declared = summarise()
    verified = bool((source_verification or {}).get("verified"))
    if verified:
        return {**declared, "source_verified": True,
                "downgraded_from_agrees": []}

    counts = dict(declared["by_verdict"])
    moved = list(declared["agrees"])
    n = counts.pop("AGREES", 0)
    if n:
        counts["NOT_CHECKABLE_HERE"] = counts.get("NOT_CHECKABLE_HERE", 0) + n
    return {
        "total_claims": declared["total_claims"],
        "by_verdict": counts,
        "agrees": [],
        "differs": declared["differs"],
        "underspecified": declared["underspecified"],
        "not_checkable_here": sorted(declared["not_checkable_here"] + moved),
        "source_verified": False,
        "downgraded_from_agrees": moved,
        "downgrade_reason": (
            "the cited source files were not verified against the pinned "
            "checkout, so no paper-to-source agreement is claimed here: "
            + "; ".join((source_verification or {}).get("problems", [])
                        or ["no verification was attempted"])),
    }


def establishes() -> list[str]:
    s = summarise()
    return [
        f"For every property the paper DOES pin down ({', '.join(s['agrees'])}), "
        "the pinned source builds what the publication describes.",
        "No property examined was found to DIFFER between the paper and the source.",
        "The realised parameter count matches Table 4's published 298.11k at the "
        "precision the table prints (ARCH-12), which constrains every "
        "architectural choice simultaneously.",
        "The dilation rule, its realised values, the filter progression, the block "
        "count and the d=1 companion layer are all specified by the paper and all "
        "reproduced exactly by the source.",
    ]


def does_not_establish() -> list[str]:
    s = summarise()
    return [
        "That the shipped checkpoints were produced by this source. File identity "
        "and loadability are not provenance.",
        f"Agreement on the {len(s['underspecified'])} properties recorded "
        "UNDERSPECIFIED: the paper side could not be located, so those "
        "comparisons were not made rather than passed.",
        "ARCH-09 in either direction. The source head is exactly 256, 256, 64, but "
        "no paper statement of it was located; Figure 4c's internal text is not "
        "machine-extractable, and a prior review in this project retracted "
        "paper-derived head widths after reading a stale revision.",
        "ARCH-16 as agreement. The paper's 'factor of two' does not describe the "
        f"realised sequence {' -> '.join(str(w) for w in WIDTH_SEQUENCE)}.",
        "Anything about trained weight values or predictive behaviour.",
        "That the census could have established the source-only properties: "
        "padding, dilation semantics, residual wiring, input normalisation and "
        "regularisation are absent from a weights-only HDF5 by construction.",
        "That the paper PDF compared against is the one the operator holds: its "
        "sha256 was never bound, because the build environment cannot fetch it.",
    ]


def to_record(source_verification: Optional[dict] = None,
              paper_verification: Optional[dict] = None) -> dict:
    """
    The full machine-readable comparison.

    When a verification record is supplied it is embedded and the
    effective summary is included alongside the declared one, so a reader
    can see both what the comparison concluded and what was defensible
    where it ran. The PAPER block is copied and, if a paper digest was
    verified, that digest is recorded in the copy rather than in the
    module-level constant.
    """
    paper = dict(PAPER)
    if paper_verification and paper_verification.get("sha256"):
        paper["pdf_sha256"] = paper_verification["sha256"]
        paper["pdf_sha256_status"] = (
            "VERIFIED against a local file by verify_paper_pdf(); see "
            "pdf_verification for the file identity. The revision is NOT "
            "asserted from this digest or from the file name.")
        paper["pdf_verification"] = paper_verification
    return {
        "artifact": COMPARISON_VERSION,
        "pinned_commit": PINNED_COMMIT,
        "paper": paper,
        "source_verification": source_verification,
        "effective_summary": effective_summary(source_verification),
        "corrections_from_previous_revision": CORRECTIONS,
        "input_size": INPUT_SIZE,
        "derived_from_source": {
            "dilation_rates": DILATION_RATES,
            "filter_schedule": FILTER_SCHEDULE,
            "width_sequence": WIDTH_SEQUENCE,
            "flatten_width": WIDTH_SEQUENCE[-1] * FILTER_SCHEDULE[-1],
        },
        "claims": [c.to_dict() for c in CLAIMS],
        "summary": summarise(),
        "establishes": establishes(),
        "does_not_establish": does_not_establish(),
    }


def record_digest(source_verification: Optional[dict] = None,
                  paper_verification: Optional[dict] = None) -> str:
    """Content hash of the comparison, so a stage record can reference it."""
    return hashlib.sha256(
        json.dumps(to_record(source_verification, paper_verification),
                   sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_record(path: Path, source_verification: Optional[dict] = None,
                 paper_verification: Optional[dict] = None) -> dict:
    """
    Serialise the comparison to a real file so a stage component can bind
    the sha256 OF THAT FILE, which is what the framework's evidence map
    expects. Returns {path, sha256, bytes, record_digest}.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    record = to_record(source_verification, paper_verification)
    record["record_digest"] = record_digest(source_verification,
                                            paper_verification)
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    p.write_text(payload, encoding="utf-8")
    return {
        "path": str(p),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "bytes": len(payload.encode("utf-8")),
        "record_digest": record["record_digest"],
    }


def main(argv=None) -> int:
    import argparse
    import json as _json
    import sys

    ap = argparse.ArgumentParser(
        description="AutoND paper-to-source architecture comparison (reads no "
                    "checkpoint, runs no model).")
    ap.add_argument("--autond-root", type=Path, default=None,
                    help="pinned checkout; the cited source files are verified "
                         "against it before any AGREES verdict stands")
    ap.add_argument("--output", type=Path, default=None,
                    help="write the full machine-readable record to this path")
    ap.add_argument("--paper-pdf", type=Path, default=None,
                    help="hash a local copy of the paper and bind it into the "
                         "record")
    ap.add_argument("--paper-record", type=Path, default=None,
                    help="also persist the paper verification on its own")
    args = ap.parse_args(argv)

    sv = verify_sources(args.autond_root)
    pv = verify_paper_pdf(args.paper_pdf) if args.paper_pdf else None
    declared = summarise()
    eff = effective_summary(sv)

    print(f"{COMPARISON_VERSION}   pinned commit {PINNED_COMMIT}")
    print(f"paper: ePrint {PAPER['eprint_id']}, {PAPER['revision_compared']}")
    print(f"source verified: {sv['verified']}"
          + (f"  (HEAD {sv['commit']})" if sv.get("commit") else ""))
    for prob in sv["problems"]:
        print(f"  - {prob}")
    print()
    for c in CLAIMS:
        shown = (c.verdict.value if sv["verified"] or
                 c.verdict is not Verdict.AGREES else "NOT_CHECKABLE_HERE*")
        print(f"  {shown:20} {c.claim_id}  {c.property_name}")
    print(f"\n  declared : {declared['by_verdict']}")
    print(f"  effective: {eff['by_verdict']}"
          + ("" if eff["source_verified"]
             else "   (* AGREES downgraded: source unverified)"))
    print(f"  record digest: {record_digest(sv, pv)}")

    if pv is not None:
        print(f"\n  paper pdf: present={pv.get('present')} "
              f"sha256={pv.get('sha256')}")
        print(f"             looks_like_pdf={pv.get('looks_like_pdf')} "
              f"revision_asserted={pv.get('revision_asserted')}")
        if args.paper_record:
            args.paper_record.parent.mkdir(parents=True, exist_ok=True)
            args.paper_record.write_text(
                _json.dumps(pv, indent=2, sort_keys=True) + "\n")
            print(f"             record written: {args.paper_record}")

    if args.output:
        info = write_record(args.output, sv, pv)
        print(f"\n  record written: {info['path']}")
        print(f"  file sha256:    {info['sha256']}")
    return 0 if sv["verified"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
