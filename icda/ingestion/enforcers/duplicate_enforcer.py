"""Duplicate enforcer - Stage 3.

Detects duplicates within batch and against existing index.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, TYPE_CHECKING

from icda.ingestion.enforcers.base_ingestion_enforcer import (
    BaseIngestionEnforcer,
    IngestionGate,
    IngestionGateResult,
    IngestionEnforcerResult,
)

if TYPE_CHECKING:
    from icda.address_index import AddressIndex
    from icda.ingestion.pipeline.ingestion_models import IngestionRecord

logger = logging.getLogger(__name__)


class DuplicateEnforcer(BaseIngestionEnforcer):
    """Stage 3: Detects duplicates in batch and against index.

    Gates:
    - NOT_IN_BATCH: Record not duplicate within current batch
    - NOT_IN_INDEX: Record not already in AddressIndex
    - SIMILARITY_BELOW_THRESHOLD: Not too similar to existing

    Uses existing AddressIndex fuzzy matching algorithms.
    """

    __slots__ = (
        "_address_index",
        "_similarity_threshold",
        "_batch_hashes",
        "_check_index",
        "_check_batch",
    )

    def __init__(
        self,
        address_index: AddressIndex | None = None,
        similarity_threshold: float = 0.95,
        check_existing_index: bool = True,
        check_batch_duplicates: bool = True,
        enabled: bool = True,
    ):
        """Initialize duplicate enforcer.

        Args:
            address_index: AddressIndex for lookup.
            similarity_threshold: Threshold above which = duplicate.
            check_existing_index: Whether to check existing index.
            check_batch_duplicates: Whether to check within batch.
            enabled: Whether enforcer is active.
        """
        super().__init__("duplicate_enforcer", enabled)
        self._address_index = address_index
        self._similarity_threshold = similarity_threshold
        self._check_index = check_existing_index
        self._check_batch = check_batch_duplicates
        self._batch_hashes: set[str] = set()

    def get_gates(self) -> list[IngestionGate]:
        """Get gates this enforcer checks."""
        return [
            IngestionGate.NOT_IN_BATCH,
            IngestionGate.NOT_IN_INDEX,
            IngestionGate.SIMILARITY_BELOW_THRESHOLD,
        ]

    def reset_batch(self) -> None:
        """Reset batch tracking for new batch."""
        self._batch_hashes.clear()

    async def enforce(
        self,
        record: IngestionRecord,
        context: dict[str, Any],
    ) -> IngestionEnforcerResult:
        """Check for duplicates.

        Args:
            record: IngestionRecord to check.
            context: Additional context (may include batch_id).

        Returns:
            IngestionEnforcerResult.
        """
        gates_passed: list[IngestionGateResult] = []
        gates_failed: list[IngestionGateResult] = []

        addr = record.parsed_address

        if not addr:
            # No address to check - pass all gates
            gates_passed.extend([
                self._gate_pass(IngestionGate.NOT_IN_BATCH, "No address to check"),
                self._gate_pass(IngestionGate.NOT_IN_INDEX, "No address to check"),
                self._gate_pass(
                    IngestionGate.SIMILARITY_BELOW_THRESHOLD,
                    "No address to check"
                ),
            ])
            return self._create_result(gates_passed, gates_failed)

        # Generate hash for batch duplicate detection
        address_hash = self._generate_hash(addr)

        # Gate 1: Not in batch
        if self._check_batch:
            if address_hash in self._batch_hashes:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.NOT_IN_BATCH,
                        "Duplicate address in current batch",
                        details={"hash": address_hash},
                    )
                )
            else:
                self._batch_hashes.add(address_hash)
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.NOT_IN_BATCH,
                        "Not a batch duplicate",
                    )
                )
        else:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.NOT_IN_BATCH,
                    "Batch duplicate check disabled",
                )
            )

        # Gate 2: Not in index (exact match)
        if self._check_index and self._address_index:
            exact_match = self._address_index.lookup_exact(addr)
            if exact_match:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.NOT_IN_INDEX,
                        f"Exact match found in index: {exact_match.customer_id}",
                        details={
                            "matched_customer": exact_match.customer_id,
                            "matched_address": exact_match.address.single_line
                            if exact_match.address else None,
                        },
                    )
                )
            else:
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.NOT_IN_INDEX,
                        "No exact match in index",
                    )
                )
        else:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.NOT_IN_INDEX,
                    "Index duplicate check disabled or no index",
                )
            )

        # Gate 3: Similarity below threshold (fuzzy match)
        if self._check_index and self._address_index:
            similar = await self._check_similarity(addr)
            if similar:
                gates_failed.append(
                    self._gate_fail(
                        IngestionGate.SIMILARITY_BELOW_THRESHOLD,
                        f"Similar address found (score={similar['score']:.2f})",
                        score=similar["score"],
                        threshold=self._similarity_threshold,
                        details=similar,
                    )
                )
            else:
                gates_passed.append(
                    self._gate_pass(
                        IngestionGate.SIMILARITY_BELOW_THRESHOLD,
                        f"No similar addresses above {self._similarity_threshold}",
                        threshold=self._similarity_threshold,
                    )
                )
        else:
            gates_passed.append(
                self._gate_pass(
                    IngestionGate.SIMILARITY_BELOW_THRESHOLD,
                    "Similarity check disabled or no index",
                )
            )

        return self._create_result(gates_passed, gates_failed)

    def _generate_hash(self, addr: Any) -> str:
        """Generate hash for address deduplication.

        Uses normalized components to generate consistent hash.

        Args:
            addr: ParsedAddress instance.

        Returns:
            Hash string.
        """
        components = [
            (addr.street_number or "").lower().strip(),
            (addr.street_name or "").lower().strip(),
            (addr.city or "").lower().strip(),
            (addr.state or "").upper().strip(),
            (addr.zip_code or "").strip()[:5],  # Only first 5 digits
        ]

        normalized = "|".join(components)
        return hashlib.md5(normalized.encode()).hexdigest()

    async def _check_similarity(
        self,
        addr: Any,
    ) -> dict[str, Any] | None:
        """Check for similar addresses in index.

        Args:
            addr: ParsedAddress to check.

        Returns:
            Dict with similar address info or None.
        """
        if not self._address_index:
            return None

        try:
            # Use existing fuzzy lookup
            matches = self._address_index.lookup_fuzzy(
                addr,
                max_results=1,
                min_score=self._similarity_threshold,
            )

            if matches:
                best = matches[0]
                return {
                    "score": best.score,
                    "customer_id": best.customer_id,
                    "match_type": best.match_type,
                    "matched_address": best.address.single_line if best.address else None,
                }

        except Exception as e:
            logger.warning(f"Similarity check failed: {e}")

        return None
