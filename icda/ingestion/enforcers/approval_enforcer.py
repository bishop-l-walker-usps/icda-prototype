"""Approval enforcer - Stage 5.

Final gate - approves or rejects records for indexing.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from icda.ingestion.enforcers.base_ingestion_enforcer import (
    BaseIngestionEnforcer,
    IngestionGate,
    IngestionGateResult,
    IngestionEnforcerResult,
)
from icda.ingestion.pipeline.ingestion_models import IngestionStatus

if TYPE_CHECKING:
    from icda.ingestion.pipeline.ingestion_models import IngestionRecord

logger = logging.getLogger(__name__)


class ApprovalEnforcer(BaseIngestionEnforcer):
    """Stage 5: Final approval gate.

    Gates:
    - ALL_GATES_PASSED: Prior enforcers all passed
    - EMBEDDING_AVAILABLE: Embedding is available
    - INDEX_READY: Record is ready for indexing
    - QUALITY_THRESHOLD_MET: Quality score meets threshold

    This is a strict enforcer - all gates must pass for approval.
    """

    __slots__ = (
        "_require_embedding",
        "_min_quality_score",
        "_require_all_prior_passed",
    )

    def __init__(
        self,
        require_embedding: bool = True,
        min_quality_score: float = 0.5,
        require_all_prior_passed: bool = False,
        enabled: bool = True,
    ):
        """Initialize approval enforcer.

        Args:
            require_embedding: Whether embedding is required.
            min_quality_score: Minimum quality score for approval.
            require_all_prior_passed: Whether all prior enforcers must pass.
            enabled: Whether enforcer is active.
        """
        # Approval enforcer is always strict
        super().__init__("approval_enforcer", enabled, strict_mode=True)
        self._require_embedding = require_embedding
        self._min_quality_score = min_quality_score
        self._require_all_prior_passed = require_all_prior_passed

    def get_gates(self) -> list[IngestionGate]:
        """Get gates this enforcer checks."""
        return [
            IngestionGate.ALL_GATES_PASSED,
            IngestionGate.EMBEDDING_AVAILABLE,
            IngestionGate.INDEX_READY,
            IngestionGate.QUALITY_THRESHOLD_MET,
        ]

    async def enforce(
        self,
        record: IngestionRecord,
        context: dict[str, Any],
    ) -> IngestionEnforcerResult:
        """Final approval check.

        Args:
            record: IngestionRecord to approve/reject.
            context: Context including prior enforcer results.

        Returns:
            IngestionEnforcerResult with approval decision.
        """
        gates_passed: list[IngestionGateResult] = []
        gates_failed: list[IngestionGateResult] = []

        # Get prior results from context
        prior_results = context.get("prior_results", [])

        # Gate 1: All gates passed (from prior enforcers)
        if self._require_all_prior_passed:
            all_passed = all(r.passed for r in prior_results)
            total_gates_failed = sum(
                len(r.gates_failed) for r in prior_results
            )

            if all_passed:
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.ALL_GATES_PASSED,
                        "All prior enforcer gates passed",
                    )
                )
            else:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.ALL_GATES_PASSED,
                        f"Prior enforcers had {total_gates_failed} failed gates",
                        details={"failed_count": total_gates_failed},
                    )
                )
        else:
            # Just check if most gates passed
            total_passed = sum(len(r.gates_passed) for r in prior_results)
            total_failed = sum(len(r.gates_failed) for r in prior_results)
            pass_rate = (
                total_passed / (total_passed + total_failed)
                if (total_passed + total_failed) > 0
                else 1.0
            )

            if pass_rate >= 0.5:
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.ALL_GATES_PASSED,
                        f"Prior gate pass rate: {pass_rate:.1%}",
                        score=pass_rate,
                    )
                )
            else:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.ALL_GATES_PASSED,
                        f"Prior gate pass rate too low: {pass_rate:.1%}",
                        score=pass_rate,
                        threshold=0.5,
                    )
                )

        # Gate 2: Embedding available
        if record.has_embedding:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.EMBEDDING_AVAILABLE,
                    f"Embedding available ({len(record.embedding)} dims)",
                    details={"dimension": len(record.embedding)},
                )
            )
        elif not self._require_embedding:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.EMBEDDING_AVAILABLE,
                    "Embedding not required",
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.EMBEDDING_AVAILABLE,
                    "Embedding required but not available",
                )
            )

        # Gate 3: Index ready (has parsed address)
        if record.parsed_address:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.INDEX_READY,
                    "Record has parsed address, ready for index",
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.INDEX_READY,
                    "No parsed address - cannot index",
                )
            )

        # Gate 4: Quality threshold met
        if record.quality_score >= self._min_quality_score:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.QUALITY_THRESHOLD_MET,
                    f"Quality score: {record.quality_score:.2f}",
                    score=record.quality_score,
                    threshold=self._min_quality_score,
                )
            )
        else:
            gates_failed.append(
                self._gate_fail(
                    IngestionGate.QUALITY_THRESHOLD_MET,
                    f"Quality score {record.quality_score:.2f} below threshold",
                    score=record.quality_score,
                    threshold=self._min_quality_score,
                )
            )

        # Create result
        result = self._create_result(gates_passed, gates_failed)

        # Update record status based on approval
        if result.passed:
            record.status = IngestionStatus.APPROVED
            logger.debug(f"Record {record.source_id} approved for indexing")
        else:
            record.status = IngestionStatus.REJECTED
            logger.debug(
                f"Record {record.source_id} rejected: "
                f"{[g.gate.value for g in gates_failed]}"
            )

        return result
