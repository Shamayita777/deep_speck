"""
Cryptographic Evidence Reporting
================================

Pretty terminal reporting for Cryptographic Evidence audits.

Responsibilities
----------------
• Format audit results
• Display evaluation decisions
• Produce human-readable terminal output

This module performs presentation only.
"""

from __future__ import annotations

from .results import (
    CryptographicTestResult,
    EvaluationResult,
)


class ReportGenerator:
    """
    Generates terminal reports.
    """

    LINE = "=" * 70

    def generate(
        self,
        result: CryptographicTestResult,
        evaluation: EvaluationResult,
    ) -> str:
        """
        Generate a formatted terminal report.
        """

        report = [
            self.LINE,
            "CRYPTOGRAPHIC EVIDENCE AUDIT REPORT",
            self.LINE,
            "",
            f"Test                 : {result.test_name}",
            f"Decision             : {evaluation.decision.value}",
            "",
            "",
            *(
                [
                    "Theory Consistency",
                    "------------------",
                    f"Correlation (ρ)     : {result.test_score:.6f}",
                    (
                        f"P-value             : {result.p_value}"
                        if result.p_value is not None
                        else "P-value             : N/A"
                    ),
                    (
                        f"Sample Size         : {result.sample_size}"
                        if result.sample_size is not None
                        else "Sample Size         : N/A"
                    ),
                    "",
                ]
                if result.test_name == "Theory Consistency"
                else
                [
                    "Representation Evidence",
                    "-----------------------",
                    f"Primary Selectivity    : {result.test_score:.6f}",
                    f"Calibration Selectivity: "
                    f"{result.metadata.get('calibration_selectivity', float('nan')):.6f}",
                    "",
                ]
                if result.test_name == "Representation Interpretation"
                else
                [
                    "Performance",
                    "-----------",
                    f"Baseline Score       : {result.baseline_score:.6f}",
                    f"Test Score           : {result.test_score:.6f}",
                    f"Performance Drop     : {result.performance_drop:.6f}",
                    f"Relative Difference  : {result.relative_difference * 100:.2f}%",
                    "",
                ]
            ),
            *(
                [
                    "Theory Consistency",
                    "------------------",
                    f"Correlation (ρ)     : {result.test_score:.6f}",
                    (
                        f"P-value             : {result.p_value}"
                        if result.p_value is not None
                        else "P-value             : N/A"
                    ),
                    (
                        f"Sample Size         : {result.sample_size}"
                        if result.sample_size is not None
                        else "Sample Size         : N/A"
                    ),
                    "",
                ]
                if result.test_name == "Theory Consistency"
                else
                [
                    "Representation Interpretation",
                    "------------------------------",
                    f"Target               : {result.metadata.get('target_name', 'N/A')}",
                    f"Metric               : {result.metadata.get('metric_name', 'N/A')}",
                    f"Real Probe Score     : {result.metadata.get('real_probe_score', float('nan')):.6f}",
                    f"Control Probe Score  : {result.metadata.get('control_probe_score', float('nan')):.6f}",
                    f"Effect Size (d_z)    : {result.metadata.get('effect_size', float('nan')):.6f}",
                    (
                        f"95% CI               : [{result.metadata.get('ci_low', float('nan')):.6f}, "
                        f"{result.metadata.get('ci_high', float('nan')):.6f}]"
                    ),
                    f"Statistical Test     : {result.metadata.get('statistical_test', 'N/A')}",
                    (
                        f"P-value              : {result.p_value}"
                        if result.p_value is not None
                        else "P-value              : N/A"
                    ),
                    (
                        f"Independent Replicates: {result.metadata.get('n_replicates', 'N/A')}"
                    ),
                    (
                        f"Folds per Replicate   : {result.metadata.get('n_splits_per_replicate', 'N/A')}"
                    ),
                    "",
                ]
                if result.test_name == "Representation Interpretation"
                else
                [
                    "Performance",
                    "-----------",
                    f"Baseline Score       : {result.baseline_score:.6f}",
                    f"Test Score           : {result.test_score:.6f}",
                    f"Performance Drop     : {result.performance_drop:.6f}",
                    f"Relative Difference  : {result.relative_difference * 100:.2f}%",
                    "",
                ]
            ),
            "Evaluation",
            "----------",
            *(
                [
                    f"Calibration Validated : {result.metadata.get('calibration_validated', False)}",
                    f"Primary Supported     : {result.metadata.get('primary_supported', False)}",
                ]
                if result.test_name == "Representation Interpretation"
                else
                [
                    f"Threshold            : {evaluation.threshold:.6f}",
                ]
            ),
            f"Observed Selectivity : {evaluation.observed_effect:.6f}",
            f"Rationale            : {evaluation.rationale}",
            "",
            "Execution",
            "---------",
            f"Runtime (seconds)    : {result.runtime:.3f}",
            f"Timestamp            : {result.timestamp.isoformat()}",
            "",
            self.LINE,
        ]

        return "\n".join(report)

    def print(
        self,
        result: CryptographicTestResult,
        evaluation: EvaluationResult,
    ) -> None:
        """
        Print a formatted terminal report.
        """

        print(self.generate(result, evaluation))

    @property
    def name(self) -> str:
        return "Report Generator"

    @property
    def description(self) -> str:
        return (
            "Produces human-readable audit reports."
        )