"""
Gohr Label Shuffle Experiment

Instantiates the Controlled Perturbation Audit
framework using Gohr's neural distinguisher and
the Label Shuffle perturbation.
"""

import speck as sp

from audit.dataset.adapters.gohr import GohrAdapter

from audit.dataset.perturbations.lable_shuffle import LabelShufflePerturbation

from audit.dataset.d4_controlled_perturbation import (
    run_perturbation,
    evaluate_result,
    generate_certificate,
    print_report,
    save_json,
)


# ============================================================
# Experiment Configuration
# ============================================================

NUM_ROUNDS = 5

DEPTH = 10

EPOCHS = 200

TRAIN_SAMPLES = 10 ** 7

VALIDATION_SAMPLES = 10 ** 6

TEST_SAMPLES = 10 ** 6

EFFECT_THRESHOLD = 20.0


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Generate datasets
    # --------------------------------------------------------

    print("Generating datasets...")

    train_x, train_y = sp.make_train_data(
        TRAIN_SAMPLES,
        NUM_ROUNDS,
    )

    validation_x, validation_y = sp.make_train_data(
        VALIDATION_SAMPLES,
        NUM_ROUNDS,
    )

    test_x, test_y = sp.make_train_data(
        TEST_SAMPLES,
        NUM_ROUNDS,
    )

    # --------------------------------------------------------
    # Create adapter
    # --------------------------------------------------------

    adapter = GohrAdapter(

        validation_x=validation_x,

        validation_y=validation_y,

        test_x=test_x,

        test_y=test_y,

        num_rounds=NUM_ROUNDS,

        depth=DEPTH,

        epochs=EPOCHS,
    )

    # --------------------------------------------------------
    # Select perturbation
    # --------------------------------------------------------

    perturbation = LabelShufflePerturbation()

    # --------------------------------------------------------
    # Execute perturbation audit
    # --------------------------------------------------------

    result = run_perturbation(

        perturbation=perturbation,

        features=train_x,

        labels=train_y,

        adapter=adapter,

        notes=(
            "Gohr neural distinguisher "
            "under label shuffle perturbation."
        ),
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluation = evaluate_result(

        result,

        effect_threshold=EFFECT_THRESHOLD,
    )

    # --------------------------------------------------------
    # Generate certificate
    # --------------------------------------------------------

    certificate = generate_certificate(

        result,

        evaluation,
    )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print_report(

        result,

        evaluation,
    )

    # --------------------------------------------------------
    # Save certificate
    # --------------------------------------------------------

    save_json(

        certificate,

        "gohr_label_shuffle_certificate.json",
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()