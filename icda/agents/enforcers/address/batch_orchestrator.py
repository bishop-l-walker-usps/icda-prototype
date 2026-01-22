"""Batch Orchestrator Agent - Coordinates all address validation enforcers.

Agent 6 of 6 in the address validation enforcer pipeline.
Orchestrates all enforcer agents for batch address processing.

Features:
- Coordinates all 5 other enforcers
- Concurrent processing with semaphore
- Real-time progress tracking
- Aggregates all enforcer results per address
- Generates final BatchValidationReport
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from ..base_enforcer import EnforcerResult
from .completion_enforcer import CompletionEnforcerAgent
from .correction_enforcer import CorrectionEnforcerAgent
from .match_confidence_enforcer import MatchConfidenceEnforcerAgent
from .models import (
    AddressFixSummary,
    AddressInput,
    BatchConfiguration,
    BatchProgress,
    BatchValidationReport,
    CompletionMetrics,
    CorrectionMetrics,
    MatchConfidenceMetrics,
    NormalizationMetrics,
)
from .normalization_enforcer import NormalizationEnforcerAgent
from .validation_report_agent import ValidationReportAgent

logger = logging.getLogger(__name__)


class BatchOrchestratorAgent:
    """Orchestrates all address validation enforcers for batch processing.

    Coordinates the 4 enforcer agents (normalization, completion, correction,
    match confidence) and the report agent for comprehensive address validation.
    """

    __slots__ = (
        "_config",
        "_normalization_enforcer",
        "_completion_enforcer",
        "_correction_enforcer",
        "_match_confidence_enforcer",
        "_report_agent",
        "_progress",
        "_progress_callback",
    )

    def __init__(
        self,
        config: BatchConfiguration | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
    ):
        """Initialize BatchOrchestratorAgent.

        Args:
            config: Configuration for batch processing.
            progress_callback: Optional callback for progress updates.
        """
        self._config = config or BatchConfiguration()
        self._progress_callback = progress_callback
        self._progress = BatchProgress()

        # Initialize enforcers
        self._normalization_enforcer = NormalizationEnforcerAgent(
            enabled=self._config.enable_normalization_enforcer,
            strict_mode=self._config.strict_mode,
        )
        self._completion_enforcer = CompletionEnforcerAgent(
            enabled=self._config.enable_completion_enforcer,
            strict_mode=self._config.strict_mode,
        )
        self._correction_enforcer = CorrectionEnforcerAgent(
            enabled=self._config.enable_correction_enforcer,
            strict_mode=self._config.strict_mode,
        )
        self._match_confidence_enforcer = MatchConfidenceEnforcerAgent(
            enabled=self._config.enable_match_confidence_enforcer,
            strict_mode=self._config.strict_mode,
        )
        self._report_agent = ValidationReportAgent()

    async def process_batch(
        self,
        addresses: list[AddressInput],
        validator: Any = None,
    ) -> BatchValidationReport:
        """Process a batch of addresses through all enforcers.

        Args:
            addresses: List of addresses to validate.
            validator: Optional address validator engine to use.

        Returns:
            BatchValidationReport with all results and statistics.
        """
        start_time = time.time()

        # Initialize progress
        self._progress = BatchProgress(total=len(addresses))
        self._update_progress()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._config.concurrency)

        # Process addresses concurrently
        tasks = [
            self._process_single_with_semaphore(
                addr, idx, semaphore, validator
            )
            for idx, addr in enumerate(addresses)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing address {idx}: {result}")
                self._progress.failed += 1

        # Generate final report
        elapsed_ms = (time.time() - start_time) * 1000
        report = self._report_agent.generate_report(total_time_ms=elapsed_ms)

        return report

    async def _process_single_with_semaphore(
        self,
        address: AddressInput,
        index: int,
        semaphore: asyncio.Semaphore,
        validator: Any,
    ) -> AddressFixSummary | None:
        """Process a single address with semaphore control.

        Args:
            address: Address to process.
            index: Index of address in batch.
            semaphore: Semaphore for concurrency control.
            validator: Address validator engine.

        Returns:
            AddressFixSummary or None if failed.
        """
        async with semaphore:
            try:
                return await self._process_single_address(
                    address, index, validator
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout processing address {address.id}")
                self._progress.failed += 1
                return None
            except Exception as e:
                logger.error(f"Error processing address {address.id}: {e}")
                self._progress.failed += 1
                return None

    async def _process_single_address(
        self,
        address: AddressInput,
        index: int,
        validator: Any,
    ) -> AddressFixSummary:
        """Process a single address through all enforcers.

        Args:
            address: Address to process.
            index: Index of address in batch.
            validator: Address validator engine.

        Returns:
            AddressFixSummary with all results.
        """
        # Update progress
        self._progress.current_index = index
        self._progress.current_address = address.full_address
        self._update_progress()

        # Validate address using engine
        validation_result = None
        standardized = address.full_address

        if validator:
            try:
                # Import ValidationMode if available
                from icda.address_validator_engine import ValidationMode
                validation_result = validator.validate(
                    address.full_address, ValidationMode.CORRECT
                )
                if validation_result and validation_result.standardized:
                    standardized = validation_result.standardized
            except Exception as e:
                logger.debug(f"Validation error for {address.id}: {e}")

        # Build context for enforcers
        context = {
            "original_address": address.full_address,
            "normalized_address": standardized,
            "standardized": standardized,
            "corrected_address": standardized,
            "validation_result": validation_result,
            "original_components": {
                "street": address.address,
                "city": address.city,
                "state": address.state,
                "zip": address.zip,
            },
            "completed_components": {
                "street": address.address,
                "city": address.city,
                "state": address.state,
                "zip": address.zip,
            },
            "corrected_components": {
                "street": address.address,
                "city": address.city,
                "state": address.state,
                "zip": address.zip,
            },
            "components": {
                "street": address.address,
                "city": address.city,
                "state": address.state,
                "zip": address.zip,
            },
            "error_type": address.error_type,
            "expected_correction": address.expected,
            "overall_confidence": (
                validation_result.overall_confidence
                if validation_result and hasattr(validation_result, "overall_confidence")
                else 0.85
            ),
        }

        # Run all enforcers
        enforcer_results: list[EnforcerResult] = []
        normalization_metrics: NormalizationMetrics | None = None
        completion_metrics: CompletionMetrics | None = None
        correction_metrics: CorrectionMetrics | None = None
        match_metrics: MatchConfidenceMetrics | None = None

        # 1. Normalization Enforcer
        if self._config.enable_normalization_enforcer:
            norm_result = await self._normalization_enforcer.enforce(context)
            enforcer_results.append(norm_result)
            if norm_result.metrics and "normalization" in norm_result.metrics:
                norm_data = norm_result.metrics["normalization"]
                if isinstance(norm_data, dict):
                    normalization_metrics = NormalizationMetrics(
                        case_changes=norm_data.get("case_changes", 0),
                        abbreviations_expanded=norm_data.get("abbreviations_expanded", 0),
                        whitespace_fixes=norm_data.get("whitespace_fixes", 0),
                        punctuation_fixes=norm_data.get("punctuation_fixes", 0),
                        directional_fixes=norm_data.get("directional_fixes", 0),
                        unit_format_fixes=norm_data.get("unit_format_fixes", 0),
                        original_text=norm_data.get("original_text", ""),
                        normalized_text=norm_data.get("normalized_text", ""),
                    )

        # 2. Completion Enforcer
        if self._config.enable_completion_enforcer:
            comp_result = await self._completion_enforcer.enforce(context)
            enforcer_results.append(comp_result)
            if comp_result.metrics and "completion" in comp_result.metrics:
                comp_data = comp_result.metrics["completion"]
                if isinstance(comp_data, dict):
                    completion_metrics = CompletionMetrics(
                        zip_inferred=comp_data.get("zip_inferred", False),
                        city_inferred=comp_data.get("city_inferred", False),
                        state_inferred=comp_data.get("state_inferred", False),
                        inferred_zip=comp_data.get("inferred_zip", ""),
                        inferred_city=comp_data.get("inferred_city", ""),
                        inferred_state=comp_data.get("inferred_state", ""),
                        confidence=comp_data.get("confidence", 0.0),
                    )

        # 3. Correction Enforcer
        if self._config.enable_correction_enforcer:
            corr_result = await self._correction_enforcer.enforce(context)
            enforcer_results.append(corr_result)
            if corr_result.metrics and "correction" in corr_result.metrics:
                corr_data = corr_result.metrics["correction"]
                if isinstance(corr_data, dict):
                    correction_metrics = CorrectionMetrics(
                        typos_fixed=corr_data.get("typos_fixed", 0),
                        transpositions_fixed=corr_data.get("transpositions_fixed", 0),
                        misspellings_fixed=corr_data.get("misspellings_fixed", 0),
                        over_corrections=corr_data.get("over_corrections", 0),
                        overall_confidence=corr_data.get("overall_confidence", 0.0),
                    )

        # 4. Match Confidence Enforcer
        if self._config.enable_match_confidence_enforcer:
            match_result = await self._match_confidence_enforcer.enforce(context)
            enforcer_results.append(match_result)
            if match_result.metrics and "match_confidence" in match_result.metrics:
                match_data = match_result.metrics["match_confidence"]
                if isinstance(match_data, dict):
                    match_metrics = MatchConfidenceMetrics(
                        primary_confidence=match_data.get("primary_confidence", 0.0),
                        secondary_confidence=match_data.get("secondary_confidence", 0.0),
                        confidence_gap=match_data.get("confidence_gap", 0.0),
                        ambiguity_score=match_data.get("ambiguity_score", 0.0),
                        match_type=match_data.get("match_type", ""),
                    )

        # Add to report agent
        summary = self._report_agent.add_address_result(
            address_id=address.id,
            original_address=address.full_address,
            final_address=standardized,
            validation_result=validation_result,
            error_type=address.error_type,
            expected_correction=address.expected,
            normalization=normalization_metrics,
            completion=completion_metrics,
            correction=correction_metrics,
            match_confidence=match_metrics,
            enforcer_results=enforcer_results,
        )

        # Update progress
        self._progress.completed += 1
        if summary.is_valid or summary.correction_successful:
            self._progress.succeeded += 1
        else:
            self._progress.failed += 1
        self._progress.elapsed_ms = (time.time() - self._progress.elapsed_ms) * 1000
        self._update_progress()

        return summary

    def _update_progress(self) -> None:
        """Update progress and call callback if set."""
        if self._progress_callback:
            self._progress_callback(self._progress)

    def get_progress(self) -> BatchProgress:
        """Get current progress."""
        return self._progress

    def get_report(self) -> BatchValidationReport | None:
        """Get the generated report."""
        return self._report_agent._report

    def format_report(self, width: int = 80) -> str:
        """Format the report for presentation.

        Args:
            width: Width of the output.

        Returns:
            Formatted ASCII string.
        """
        return self._report_agent.format_for_presentation(width)

    def to_json(self) -> dict[str, Any]:
        """Export report as JSON."""
        return self._report_agent.to_json()

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """Export results as CSV rows."""
        return self._report_agent.to_csv_rows()


def parse_addresses_from_file(
    file_path: str,
    data: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> list[AddressInput]:
    """Parse addresses from file data into AddressInput objects.

    Supports:
    - JSON list of addresses
    - JSON object with "addresses" key
    - CSV-style list of dictionaries

    Auto-maps common field names:
    - address/full_address/addr -> address
    - city/city_name -> city
    - state/st -> state
    - zip/zip_code/postal/postal_code -> zip
    - error_type/category -> error_type

    Args:
        file_path: Path to the file (for format detection).
        data: Pre-loaded data (optional).

    Returns:
        List of AddressInput objects.
    """
    addresses: list[AddressInput] = []

    if data is None:
        return addresses

    # Handle list or dict with addresses key
    if isinstance(data, dict):
        if "addresses" in data:
            raw_addresses = data["addresses"]
        else:
            raw_addresses = [data]
    elif isinstance(data, list):
        raw_addresses = data
    else:
        return addresses

    # Field mapping
    address_fields = ["address", "full_address", "addr", "street"]
    city_fields = ["city", "city_name"]
    state_fields = ["state", "st", "state_code"]
    zip_fields = ["zip", "zip_code", "postal", "postal_code", "zipcode"]
    error_type_fields = ["error_type", "category", "error"]
    id_fields = ["id", "address_id", "ID"]

    for idx, raw in enumerate(raw_addresses):
        if not isinstance(raw, dict):
            continue

        # Auto-map fields
        addr_val = ""
        for field in address_fields:
            if field in raw:
                addr_val = str(raw[field])
                break

        city_val = ""
        for field in city_fields:
            if field in raw:
                city_val = str(raw[field])
                break

        state_val = ""
        for field in state_fields:
            if field in raw:
                state_val = str(raw[field])
                break

        zip_val = ""
        for field in zip_fields:
            if field in raw:
                zip_val = str(raw[field])
                break

        error_type_val = ""
        for field in error_type_fields:
            if field in raw:
                error_type_val = str(raw[field])
                break

        id_val: int | str = idx + 1
        for field in id_fields:
            if field in raw:
                id_val = raw[field]
                break

        # Build expected correction dict
        expected: dict[str, Any] = {}
        if "correct_city" in raw:
            expected["city"] = raw["correct_city"]
        if "correct_street" in raw:
            expected["street"] = raw["correct_street"]
        if "correct_zip" in raw:
            expected["zip"] = raw["correct_zip"]
        if "original_zip" in raw:
            expected["original_zip"] = raw["original_zip"]
        if "correct_address" in raw:
            expected["full_address"] = raw["correct_address"]
        if "correct_state" in raw:
            expected["state"] = raw["correct_state"]
        if "correct_format" in raw:
            expected["format"] = raw["correct_format"]
        if "note" in raw:
            expected["note"] = raw["note"]

        # Create AddressInput
        addr_input = AddressInput(
            id=id_val,
            address=addr_val,
            city=city_val,
            state=state_val,
            zip=zip_val,
            error_type=error_type_val,
            expected=expected,
        )
        addresses.append(addr_input)

    return addresses
