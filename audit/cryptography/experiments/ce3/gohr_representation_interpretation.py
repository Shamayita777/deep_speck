from __future__ import annotations
import json
from audit.cryptography.adapters.gohr import GohrAdapter
from audit.cryptography.certificate import CertificateGenerator
from audit.cryptography.engine import CryptographicEngine
from audit.cryptography.evaluation import CryptographicEvaluator
from audit.cryptography.reporting import ReportGenerator
from audit.cryptography.test.ce3.representation_interpretation import (
    RepresentationInterpretationTest,
)


def main() -> None:

    adapter = GohrAdapter(
        model_path="audit/cryptography/evidence/ce1/best5depth10 (10).h5",
    )

    test = RepresentationInterpretationTest()

    engine = CryptographicEngine()

    # NOTE: unlike CE2's supported/inconclusive thresholds, these
    # are NOT derived from an established convention -- there is
    # no Cohen's-style citable standard for a selectivity gap in
    # accuracy/R^2 units the way there is for correlation
    # strength. Treat these as placeholders. The principled way
    # to calibrate them is empirically: run CE3 once against the
    # differential-class calibration target discussed earlier
    # (a quantity already known to be encoded, since the output
    # layer must decode it to achieve training accuracy at all)
    # and use its observed selectivity as a reference scale for
    # what "supported" should mean for the primary target, rather
    # than picking round numbers.
    evaluator = CryptographicEvaluator(
        supported_threshold=0.20,
        inconclusive_threshold=0.05,
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

    with open("ce3_certificate.json", "w", encoding="utf-8") as f:
        json.dump(
            certificate_generator.to_dict(certificate), f, indent=4,
        )

    report_generator.print(result=result, evaluation=evaluation)

    print()
    print("Audit Certificate")
    print("-----------------")
    print(certificate_generator.to_dict(certificate))
    print("Certificate saved to: ce3_certificate.json")


if __name__ == "__main__":
    main()