"""
Gohr Signal Destruction Experiment
=================================

Reference experiment demonstrating the Cryptographic Evidence
framework using the Gohr neural distinguisher.

Execution Pipeline
------------------
Create Adapter
      ↓
Create SignalDestructionTest
      ↓
Execute Test
      ↓
Evaluate Result
      ↓
Generate Certificate
      ↓
Generate Report
"""

from __future__ import annotations
import json
from audit.cryptography.adapters.gohr import GohrAdapter
from audit.cryptography.certificate import CertificateGenerator
from audit.cryptography.engine import CryptographicEngine
from audit.cryptography.evaluation import CryptographicEvaluator
from audit.cryptography.reporting import ReportGenerator
from audit.cryptography.test.ce1.signal_destruction import (
    SignalDestructionTest,
)


def main() -> None:
    """
    Execute the Gohr Signal Destruction experiment.
    """

    # ---------------------------------------------------------
    # Framework Components
    # ---------------------------------------------------------

    adapter = GohrAdapter()

    test = SignalDestructionTest()

    engine = CryptographicEngine()

    evaluator = CryptographicEvaluator()

    certificate_generator = CertificateGenerator()

    report_generator = ReportGenerator()

    # ---------------------------------------------------------
    # Execute Experiment
    # ---------------------------------------------------------

    result = engine.execute(
        test=test,
        adapter=adapter,
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    evaluation = evaluator.evaluate(result)

    # ---------------------------------------------------------
    # Certificate
    # ---------------------------------------------------------

    certificate = certificate_generator.generate(
        result=result,
        evaluation=evaluation,
    )

    # ---------------------------------------------------------
    # Save Machine-readable Certificate
    # ---------------------------------------------------------

    with open(
        "ce1_certificate.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            certificate_generator.to_dict(certificate),
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    report_generator.print(
        result=result,
        evaluation=evaluation,
    )

    # ---------------------------------------------------------
    # Machine-readable Output
    # ---------------------------------------------------------

    print()

    print("Audit Certificate")

    print("-----------------")

    print(
        certificate_generator.to_dict(
            certificate
        )
    )
    print("Certificate saved to: ce1_certificate.json")

if __name__ == "__main__":
    main()