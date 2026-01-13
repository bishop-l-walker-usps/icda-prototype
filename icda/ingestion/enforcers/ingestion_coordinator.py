"""Ingestion enforcer coordinator.

Orchestrates all 5 ingestion enforcers in sequence.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

from icda.ingestion.enforcers.base_ingestion_enforcer import (
    IngestionEnforcerResult,
)
from icda.ingestion.enforcers.schema_enforcer import SchemaEnforcer
from icda.ingestion.enforcers.normalization_enforcer import NormalizationEnforcer
from icda.ingestion.enforcers.duplicate_enforcer import DuplicateEnforcer
from icda.ingestion.enforcers.quality_enforcer import QualityEnforcer
from icda.ingestion.enforcers.approval_enforcer import ApprovalEnforcer
from icda.ingestion.pipeline.ingestion_models import IngestionStatus

if TYPE_CHECKING:
    from icda.address_normalizer import AddressNormalizer
    from icda.address_index import AddressIndex
    from icda.ingestion.pipeline.ingestion_models import IngestionRecord

logger = logging.getLogger(__name__)


class IngestionEnforcerCoordinator:
    """Orchestrates all 5 ingestion enforcers in sequence.

    Execution Order:
    1. SchemaEnforcer - Validate mapped fields
    2. NormalizationEnforcer - Parse and normalize address
    3. DuplicateEnforcer - Check for duplicates
    4. QualityEnforcer - Score quality
    5. ApprovalEnforcer - Final approval gate

    Features:
    - Runs enforcers in sequence
    - Early exit on critical failure (fail_fast mode)
    - Aggregates results into single pass/fail decision
    - Tracks metrics and recommendations
    """

    __slots__ = (
        "_enabled",
        "_fail_fast",
        "_schema_enforcer",
        "_normalization_enforcer",
        "_duplicate_enforcer",
        "_quality_enforcer",
        "_approval_enforcer",
        "_stats",
    )

    def __init__(
        self,
        address_normalizer: AddressNormalizer | None = None,
        address_index: AddressIndex | None = None,
        enabled: bool = True,
        fail_fast: bool = False,
        # Schema enforcer config
        required_fields: list[str] | None = None,
        # Duplicate enforcer config
        similarity_threshold: float = 0.95,
        check_existing_index: bool = True,
        # Quality enforcer config
        min_completeness: float = 0.6,
        min_confidence: float = 0.7,
        require_pr_urbanization: bool = True,
        # Approval enforcer config
        require_embedding: bool = True,
        min_quality_score: float = 0.5,
    ):
        """Initialize coordinator with all enforcers.

        Args:
            address_normalizer: AddressNormalizer instance.
            address_index: AddressIndex instance.
            enabled: Whether enforcer pipeline is enabled.
            fail_fast: Stop at first enforcer failure.
            required_fields: Fields required by schema enforcer.
            similarity_threshold: Duplicate detection threshold.
            check_existing_index: Check duplicates against index.
            min_completeness: Minimum completeness score.
            min_confidence: Minimum confidence threshold.
            require_pr_urbanization: Require URB for PR addresses.
            require_embedding: Require embedding for approval.
            min_quality_score: Minimum quality for approval.
        """
        self._enabled = enabled
        self._fail_fast = fail_fast

        # Initialize all 5 enforcers
        self._schema_enforcer = SchemaEnforcer(
            required_fields=required_fields,
            enabled=enabled,
        )

        self._normalization_enforcer = NormalizationEnforcer(
            normalizer=address_normalizer,
            enabled=enabled,
        )

        self._duplicate_enforcer = DuplicateEnforcer(
            address_index=address_index,
            similarity_threshold=similarity_threshold,
            check_existing_index=check_existing_index,
            enabled=enabled,
        )

        self._quality_enforcer = QualityEnforcer(
            min_completeness=min_completeness,
            min_confidence=min_confidence,
            require_pr_urbanization=require_pr_urbanization,
            enabled=enabled,
        )

        self._approval_enforcer = ApprovalEnforcer(
            require_embedding=require_embedding,
            min_quality_score=min_quality_score,
            enabled=enabled,
        )

        self._stats = {
            "records_processed": 0,
            "approved": 0,
            "rejected": 0,
            "total_gates_passed": 0,
            "total_gates_failed": 0,
        }

    @property
    def enabled(self) -> bool:
        """Check if coordinator is enabled."""
        return self._enabled

    @property
    def stats(self) -> dict[str, int]:
        """Get coordinator statistics."""
        return self._stats.copy()

    def reset_batch(self) -> None:
        """Reset batch state for new batch processing."""
        self._duplicate_enforcer.reset_batch()

    async def enforce(
        self,
        record: IngestionRecord,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run all enforcers on a record.

        Args:
            record: IngestionRecord to validate.
            context: Additional context.

        Returns:
            Dict with:
            - approved: Final approval decision
            - quality_score: Aggregate quality score
            - results: List of enforcer results
            - recommendations: Combined recommendations
            - metrics: Performance metrics
        """
        if not self._enabled:
            # If disabled, auto-approve
            record.status = IngestionStatus.APPROVED
            return {
                "approved": True,
                "quality_score": 1.0,
                "results": [],
                "recommendations": [],
                "metrics": {"enforcer_disabled": True},
            }

        context = context or {}
        start_time = time.time()

        results: list[IngestionEnforcerResult] = []
        current_record = record

        # Run enforcers in sequence
        enforcers = [
            ("schema", self._schema_enforcer),
            ("normalization", self._normalization_enforcer),
            ("duplicate", self._duplicate_enforcer),
            ("quality", self._quality_enforcer),
        ]

        for name, enforcer in enforcers:
            if not enforcer.enabled:
                continue

            result = await enforcer.enforce(current_record, context)
            results.append(result)

            # Update record if modified
            if result.modified_record:
                current_record = result.modified_record

            # Track stats
            self._stats["total_gates_passed"] += len(result.gates_passed)
            self._stats["total_gates_failed"] += len(result.gates_failed)

            # Fail fast if configured
            if self._fail_fast and not result.passed:
                logger.debug(f"Fail-fast triggered at {name} enforcer")
                break

        # Run approval enforcer with prior results
        approval_context = {
            **context,
            "prior_results": results,
        }
        approval_result = await self._approval_enforcer.enforce(
            current_record, approval_context
        )
        results.append(approval_result)

        # Update stats
        self._stats["records_processed"] += 1
        if approval_result.passed:
            self._stats["approved"] += 1
        else:
            self._stats["rejected"] += 1

        # Calculate aggregate metrics
        elapsed_ms = int((time.time() - start_time) * 1000)
        aggregate_quality = self._calculate_aggregate_quality(results)

        # Collect all recommendations
        all_recommendations: list[str] = []
        for r in results:
            all_recommendations.extend(r.recommendations)

        return {
            "approved": approval_result.passed,
            "quality_score": aggregate_quality,
            "results": [r.to_dict() for r in results],
            "recommendations": all_recommendations,
            "metrics": {
                "elapsed_ms": elapsed_ms,
                "enforcers_run": len(results),
                "total_gates_passed": sum(len(r.gates_passed) for r in results),
                "total_gates_failed": sum(len(r.gates_failed) for r in results),
            },
        }

    def _calculate_aggregate_quality(
        self,
        results: list[IngestionEnforcerResult],
    ) -> float:
        """Calculate aggregate quality score from all enforcers."""
        if not results:
            return 1.0

        # Weighted average of enforcer quality scores
        weights = {
            "schema_enforcer": 0.15,
            "normalization_enforcer": 0.25,
            "duplicate_enforcer": 0.15,
            "quality_enforcer": 0.30,
            "approval_enforcer": 0.15,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for result in results:
            weight = weights.get(result.enforcer_name, 0.1)
            weighted_sum += result.quality_score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 1.0

    def get_info(self) -> dict[str, Any]:
        """Get coordinator information."""
        return {
            "enabled": self._enabled,
            "fail_fast": self._fail_fast,
            "enforcers": [
                self._schema_enforcer.name,
                self._normalization_enforcer.name,
                self._duplicate_enforcer.name,
                self._quality_enforcer.name,
                self._approval_enforcer.name,
            ],
            "stats": self._stats,
        }
