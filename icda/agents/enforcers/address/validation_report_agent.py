"""Validation Report Agent - Comprehensive report generation.

Agent 5 of 6 in the address validation enforcer pipeline.
Generates detailed validation reports for presentation.

Outputs:
- AddressFixSummary - Per-address detailed report
- BatchValidationReport - Comprehensive batch statistics
- format_for_presentation() - ASCII table output for demos
"""

from __future__ import annotations

from typing import Any

from .models import (
    AddressFixSummary,
    BatchValidationReport,
    ComponentQualityStats,
    CompletionMetrics,
    CorrectionMetrics,
    ErrorTypeStats,
    MatchConfidenceMetrics,
    NormalizationMetrics,
)


class ValidationReportAgent:
    """Agent for generating comprehensive validation reports.

    Creates detailed per-address summaries and batch-level statistics
    with presentation-quality ASCII formatting.
    """

    __slots__ = ("_summaries", "_report")

    def __init__(self):
        """Initialize ValidationReportAgent."""
        self._summaries: list[AddressFixSummary] = []
        self._report: BatchValidationReport | None = None

    def add_address_result(
        self,
        address_id: int | str,
        original_address: str,
        final_address: str,
        validation_result: Any,
        error_type: str = "",
        expected_correction: dict[str, Any] | None = None,
        normalization: NormalizationMetrics | None = None,
        completion: CompletionMetrics | None = None,
        correction: CorrectionMetrics | None = None,
        match_confidence: MatchConfidenceMetrics | None = None,
        enforcer_results: list[Any] | None = None,
    ) -> AddressFixSummary:
        """Add a validation result to the report.

        Args:
            address_id: Unique identifier for the address.
            original_address: Original input address.
            final_address: Final validated/corrected address.
            validation_result: Validation result object.
            error_type: Type of error that was present.
            expected_correction: Expected correction details.
            normalization: Normalization metrics.
            completion: Completion metrics.
            correction: Correction metrics.
            match_confidence: Match confidence metrics.
            enforcer_results: List of enforcer results.

        Returns:
            AddressFixSummary for this address.
        """
        summary = AddressFixSummary(
            address_id=address_id,
            original_address=original_address,
            final_address=final_address,
            error_type=error_type,
            expected_correction=expected_correction or {},
            normalization=normalization,
            completion=completion,
            correction=correction,
            match_confidence=match_confidence,
        )

        # Extract validation status
        if validation_result:
            if hasattr(validation_result, "is_valid"):
                summary.is_valid = validation_result.is_valid
            if hasattr(validation_result, "overall_confidence"):
                summary.overall_confidence = validation_result.overall_confidence
            if hasattr(validation_result, "corrections_applied"):
                summary.is_corrected = bool(validation_result.corrections_applied)
            if hasattr(validation_result, "issues"):
                summary.issues = [
                    i.message if hasattr(i, "message") else str(i)
                    for i in (validation_result.issues or [])
                ]

        # Calculate quality score from enforcer results
        if enforcer_results:
            total_passed = sum(len(r.gates_passed) for r in enforcer_results if hasattr(r, "gates_passed"))
            total_failed = sum(len(r.gates_failed) for r in enforcer_results if hasattr(r, "gates_failed"))
            summary.gates_passed = total_passed
            summary.gates_failed = total_failed
            if total_passed + total_failed > 0:
                summary.quality_score = total_passed / (total_passed + total_failed)

        # Check if correction was successful
        summary.correction_successful = self._check_correction_success(
            original_address, final_address, error_type, expected_correction or {}
        )

        self._summaries.append(summary)
        return summary

    def _check_correction_success(
        self,
        original: str,
        corrected: str,
        error_type: str,
        expected: dict[str, Any],
    ) -> bool:
        """Check if the correction was successful based on expected values."""
        corr_upper = corrected.upper()

        if error_type == "misspelled_city":
            expected_city = expected.get("city", "").upper()
            return bool(expected_city and expected_city in corr_upper)

        if error_type == "misspelled_street":
            expected_street = expected.get("street", "").upper()
            if expected_street:
                parts = expected_street.split()
                if len(parts) > 1:
                    return parts[1] in corr_upper
            return False

        if error_type == "missing_zip":
            expected_zip = expected.get("original_zip", "")
            return bool(expected_zip and expected_zip in corr_upper)

        if error_type == "wrong_zip_for_city":
            expected_zip = expected.get("zip", "")
            return bool(expected_zip and expected_zip in corr_upper)

        if error_type == "transposed_zip_digits":
            expected_zip = expected.get("zip", "")
            return bool(expected_zip and expected_zip in corr_upper)

        if error_type == "wrong_state":
            expected_state = expected.get("state", "")
            return bool(expected_state and f", {expected_state} " in corr_upper)

        if error_type == "extra_spaces_formatting":
            return "  " not in corrected

        if error_type == "mixed_case_issues":
            return corrected == corrected.upper() or self._is_title_case(corrected)

        return True

    def _is_title_case(self, text: str) -> bool:
        """Check if text is in proper title case."""
        words = text.replace(",", " ").split()
        for word in words:
            if not word:
                continue
            if word != word.upper() and word != word.title():
                if not word[0].isupper():
                    return False
        return True

    def generate_report(self, total_time_ms: float = 0.0) -> BatchValidationReport:
        """Generate the batch validation report.

        Args:
            total_time_ms: Total processing time in milliseconds.

        Returns:
            BatchValidationReport with all statistics.
        """
        report = BatchValidationReport(
            total_addresses=len(self._summaries),
            total_time_ms=total_time_ms,
            address_summaries=self._summaries,
        )

        if not self._summaries:
            self._report = report
            return report

        # Calculate basic stats
        report.verified = sum(1 for s in self._summaries if s.is_valid)
        report.corrected = sum(1 for s in self._summaries if s.is_corrected)
        report.failed = sum(1 for s in self._summaries if not s.is_valid and not s.is_corrected)
        report.skipped = report.total_addresses - report.verified - report.corrected - report.failed

        # Calculate averages
        confidences = [s.overall_confidence for s in self._summaries if s.overall_confidence > 0]
        report.avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        quality_scores = [s.quality_score for s in self._summaries if s.quality_score > 0]
        report.avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Calculate gate statistics
        report.total_gates_passed = sum(s.gates_passed for s in self._summaries)
        report.total_gates_failed = sum(s.gates_failed for s in self._summaries)

        # Calculate error type statistics
        for summary in self._summaries:
            error_type = summary.error_type or "unknown"

            if error_type not in report.error_type_stats:
                report.error_type_stats[error_type] = ErrorTypeStats(error_type=error_type)

            stats = report.error_type_stats[error_type]
            stats.total += 1
            if summary.is_valid:
                stats.validated += 1
            if summary.is_corrected:
                stats.corrected += 1
            if summary.correction_successful:
                stats.correction_successful += 1

        # Calculate component quality stats
        self._calculate_component_stats(report)

        # Calculate enforcer totals
        for summary in self._summaries:
            if summary.normalization:
                report.normalization_changes += summary.normalization.total_changes
            if summary.completion and summary.completion.has_inferences:
                report.completions_applied += 1
            if summary.correction:
                report.corrections_applied += summary.correction.total_corrections

        # Calculate average time per address
        if report.total_addresses > 0:
            report.avg_time_per_address_ms = total_time_ms / report.total_addresses

        self._report = report
        return report

    def _calculate_component_stats(self, report: BatchValidationReport) -> None:
        """Calculate component-level quality statistics."""
        components = ["street_number", "street_name", "city", "state", "zip"]

        for comp in components:
            stats = ComponentQualityStats(component=comp)
            report.component_stats[comp] = stats

        for summary in self._summaries:
            # Analyze what components were processed/fixed
            error_type = summary.error_type or ""

            if "city" in error_type:
                report.component_stats["city"].total_processed += 1
                if summary.correction_successful:
                    report.component_stats["city"].successful += 1
                    report.component_stats["city"].corrections_applied += 1
                else:
                    report.component_stats["city"].errors_found += 1

            elif "street" in error_type:
                report.component_stats["street_name"].total_processed += 1
                if summary.correction_successful:
                    report.component_stats["street_name"].successful += 1
                    report.component_stats["street_name"].corrections_applied += 1
                else:
                    report.component_stats["street_name"].errors_found += 1

            elif "zip" in error_type:
                report.component_stats["zip"].total_processed += 1
                if summary.correction_successful:
                    report.component_stats["zip"].successful += 1
                    report.component_stats["zip"].corrections_applied += 1
                else:
                    report.component_stats["zip"].errors_found += 1

            elif "state" in error_type:
                report.component_stats["state"].total_processed += 1
                if summary.correction_successful:
                    report.component_stats["state"].successful += 1
                    report.component_stats["state"].corrections_applied += 1
                else:
                    report.component_stats["state"].errors_found += 1

    def format_for_presentation(self, width: int = 80) -> str:
        """Generate ASCII-formatted report for presentation.

        Args:
            width: Width of the output in characters.

        Returns:
            Formatted ASCII string.
        """
        if not self._report:
            self.generate_report()

        report = self._report
        lines: list[str] = []

        # Header
        lines.append("=" * width)
        lines.append("ADDRESS VALIDATION REPORT".center(width))
        lines.append("=" * width)
        lines.append("")

        # Summary section
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Addresses:     {report.total_addresses:>6}")
        lines.append(f"Verified:            {report.verified:>6} ({report.verification_rate * 100:.1f}%)")
        lines.append(f"Corrected:           {report.corrected:>6} ({report.correction_rate * 100:.1f}%)")
        lines.append(f"Failed:              {report.failed:>6} ({report.failure_rate * 100:.1f}%)")
        lines.append(f"Skipped:             {report.skipped:>6}")
        lines.append("")
        lines.append(f"Avg Confidence:      {report.avg_confidence * 100:>6.1f}%")
        lines.append(f"Avg Quality Score:   {report.avg_quality_score * 100:>6.1f}%")
        lines.append(f"Gate Pass Rate:      {report.gate_pass_rate * 100:>6.1f}%")
        lines.append("")
        if report.total_time_ms > 0:
            lines.append(f"Total Time:          {report.total_time_ms:>6.0f} ms")
            lines.append(f"Avg Time/Address:    {report.avg_time_per_address_ms:>6.1f} ms")
        lines.append("")

        # Error Type Breakdown
        lines.append("ERROR TYPE BREAKDOWN")
        lines.append("-" * 40)
        lines.append(f"{'Error Type':<26} {'Fixed':>7}   {'Rate':>8}")

        for error_type in sorted(report.error_type_stats.keys()):
            stats = report.error_type_stats[error_type]
            rate = stats.success_rate * 100
            lines.append(f"  {error_type:<24} {stats.correction_successful:>2}/{stats.total:<2}   ({rate:>5.1f}%)")

        lines.append("")

        # Component Quality
        lines.append("COMPONENT QUALITY")
        lines.append("-" * 40)

        for comp_name, stats in report.component_stats.items():
            if stats.total_processed > 0:
                rate = stats.success_rate
                bar = self._progress_bar(rate, 20)
                lines.append(f"  {comp_name:<14} {bar} {rate * 100:>5.1f}%")

        lines.append("")

        # Enforcer Statistics
        lines.append("ENFORCER STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Normalization Changes:   {report.normalization_changes:>6}")
        lines.append(f"Completions Applied:     {report.completions_applied:>6}")
        lines.append(f"Corrections Applied:     {report.corrections_applied:>6}")
        lines.append(f"Gates Passed:            {report.total_gates_passed:>6}")
        lines.append(f"Gates Failed:            {report.total_gates_failed:>6}")
        lines.append("")

        # Footer
        lines.append("=" * width)

        return "\n".join(lines)

    def _progress_bar(self, ratio: float, width: int = 20) -> str:
        """Generate an ASCII progress bar."""
        filled = int(ratio * width)
        empty = width - filled
        return "[" + "#" * filled + "." * empty + "]"

    def format_address_detail(self, summary: AddressFixSummary, width: int = 80) -> str:
        """Format detailed output for a single address.

        Args:
            summary: Address fix summary.
            width: Width of the output.

        Returns:
            Formatted ASCII string.
        """
        lines: list[str] = []

        lines.append("-" * width)
        lines.append(f"Address ID: {summary.address_id}")
        lines.append(f"Original:   {summary.original_address[:width - 12]}")
        lines.append(f"Final:      {summary.final_address[:width - 12]}")
        lines.append(f"Error Type: {summary.error_type}")
        lines.append(f"Valid: {summary.is_valid} | Corrected: {summary.is_corrected} | Success: {summary.correction_successful}")
        lines.append(f"Confidence: {summary.overall_confidence:.1%} | Quality: {summary.quality_score:.1%}")
        lines.append(f"Gates: {summary.gates_passed} passed, {summary.gates_failed} failed")

        if summary.issues:
            lines.append("Issues:")
            for issue in summary.issues[:3]:
                lines.append(f"  - {issue[:width - 4]}")

        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Export report as JSON-serializable dictionary."""
        if not self._report:
            self.generate_report()

        return {
            "summary": self._report.to_dict(),
            "addresses": [s.to_dict() for s in self._summaries],
        }

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Export results as CSV-compatible rows."""
        rows = []
        for summary in self._summaries:
            rows.append({
                "id": summary.address_id,
                "original": summary.original_address,
                "final": summary.final_address,
                "error_type": summary.error_type,
                "is_valid": summary.is_valid,
                "is_corrected": summary.is_corrected,
                "correction_successful": summary.correction_successful,
                "confidence": summary.overall_confidence,
                "quality_score": summary.quality_score,
                "gates_passed": summary.gates_passed,
                "gates_failed": summary.gates_failed,
            })
        return rows
