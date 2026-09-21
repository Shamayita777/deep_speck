from __future__ import annotations
import json
import dataclasses
import numpy as np
from audit.cryptography.adapters.gohr import GohrAdapter
from audit.cryptography.certificate import CertificateGenerator
from audit.cryptography.engine import CryptographicEngine
from audit.cryptography.evaluation import CryptographicEvaluator
from audit.cryptography.reporting import ReportGenerator
from audit.cryptography.test.ce4.causal_intervention import (
    CausalInterventionTest,
)
from audit.cryptography.gohr.dataset import GohrDataset

def calibrate_necessity_ceiling(
    adapter, model, task, *, n_calibration_samples: int = 5000,
) -> float:
    """
    Empirical ceiling for the necessity gap on THIS model and
    dataset: the mean |output change| from flipping every input
    bit, versus flipping none. Used to scale supported/
    inconclusive thresholds relative to what this model's output
    can move by at all, rather than an arbitrary fixed number.
    """
    subset = dataclasses.replace(
        task.dataset, X=task.dataset.X[:n_calibration_samples],
    )
    original_predictions = adapter.compute_model_predictions(model, subset)

    all_flipped = dataclasses.replace(subset, X=subset.X ^ 1)
    flipped_predictions = adapter.compute_model_predictions(model, all_flipped)

    return float(np.mean(np.abs(original_predictions - flipped_predictions)))


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

    # Load the frozen model used by CE4.
    model = adapter.load()

    # Obtain the adapter-declared primary intervention task.
    tasks = adapter.generate_intervention_tasks()
    primary_tasks = [t for t in tasks if t.is_primary]

    if len(primary_tasks) != 1:
        raise ValueError(
            "Exactly one InterventionTask must be marked "
            f"is_primary=True; found {len(primary_tasks)}."
        )

    primary_task = primary_tasks[0]

    # Calibrate the empirical output-change ceiling for this
    # model and dataset before defining CE4 decision thresholds.
    ceiling = calibrate_necessity_ceiling(
        adapter,
        model,
        primary_task,
    )

    evaluator = CryptographicEvaluator(
        supported_threshold=0.10 * ceiling,
        inconclusive_threshold=0.02 * ceiling,
    )
    # Decision thresholds are scaled to the empirical output-change
    # ceiling of this model and dataset, avoiding fixed raw-unit
    # thresholds.

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