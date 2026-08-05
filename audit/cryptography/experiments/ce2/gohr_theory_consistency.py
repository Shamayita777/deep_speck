from __future__ import annotations
import json
from audit.cryptography.adapters.gohr import GohrAdapter
from audit.cryptography.certificate import CertificateGenerator
from audit.cryptography.engine import CryptographicEngine
from audit.cryptography.evaluation import CryptographicEvaluator
from audit.cryptography.reporting import ReportGenerator
from audit.cryptography.test.ce2.theory_consistency import (
    TheoryConsistencyTest,
)


def main() -> None:

    adapter = GohrAdapter(
        model_path="audit/cryptography/evidence/ce1/best5depth10 (10).h5",
        theory_num_samples=10000,
    )

    test = TheoryConsistencyTest()

    engine = CryptographicEngine()

    evaluator = CryptographicEvaluator(
        supported_threshold=0.30,
        inconclusive_threshold=0.10,
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

    with open("ce2_certificate.json", "w", encoding="utf-8") as f:
        json.dump(
            certificate_generator.to_dict(certificate), f, indent=4,
        )

    report_generator.print(result=result, evaluation=evaluation)

    print()
    print("Audit Certificate")
    print("-----------------")
    print(certificate_generator.to_dict(certificate))
    print("Certificate saved to: ce2_certificate.json")


if __name__ == "__main__":
    main()