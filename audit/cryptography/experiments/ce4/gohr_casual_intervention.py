from __future__ import annotations
import json
from audit.cryptography.adapters.gohr import GohrAdapter
from audit.cryptography.certificate import CertificateGenerator
from audit.cryptography.engine import CryptographicEngine
from audit.cryptography.evaluation import CryptographicEvaluator
from audit.cryptography.reporting import ReportGenerator
from audit.cryptography.test.ce4.causal_intervention import (
    CausalInterventionTest,
)
from audit.cryptography.gohr.dataset import GohrDataset


def main() -> None:

    adapter = GohrAdapter(
        dataset=GohrDataset(
            rounds=5,
            differential=(0x0040, 0x0000),
        ),
        model_path="audit/cryptography/evidence/ce1/best5depth10 (10).h5",
    )

    test = CausalInterventionTest()

    engine = CryptographicEngine()

    # NOTE: same caveat as CE3's thresholds -- there is no
    # citable convention for what a "large" necessity gap is in
    # raw output-probability-difference units, unlike CE2's
    # correlation-strength thresholds (Cohen, 1988). Treat these
    # as placeholders. Since CE4 is cheap to run at large N (no
    # training involved, just inference), the principled way to
    # calibrate them is empirically: run CE4 once against a
    # target you expect to be trivially necessary (e.g. flipping
    # every input bit vs. flipping none) to see what the ceiling
    # necessity gap looks like for this model, and scale these
    # thresholds relative to that ceiling rather than picking
    # round numbers.
    evaluator = CryptographicEvaluator(
        supported_threshold=0.10,
        inconclusive_threshold=0.02,
    )

    certificate_generator = CertificateGenerator()

    report_generator = ReportGenerator()

    result = engine.execute(test=test, adapter=adapter)

    evaluation = evaluator.evaluate(
        test,
        result,
    )

    certificate = certificate_generator.generate(
        result=result, evaluation=evaluation,
    )

    with open("ce4_certificate.json", "w", encoding="utf-8") as f:
        json.dump(
            certificate_generator.to_dict(certificate), f, indent=4,
        )

    report_generator.print(result=result, evaluation=evaluation)

    print()
    print("Audit Certificate")
    print("-----------------")
    print(certificate_generator.to_dict(certificate))
    print("Certificate saved to: ce4_certificate.json")


if __name__ == "__main__":
    main()