"""Directory Coverage Enforcer - Validates knowledge directory scanning.

This enforcer ensures that all knowledge directories are being scanned
and indexed properly for RAG retrieval.

Key Gates:
- DIRECTORY_COVERAGE_COMPLETE: All configured directories are scanned
- FILE_TYPE_SUPPORT: All relevant file types are indexed
- INDEX_FRESHNESS: Index is up-to-date with file modifications
- ORPHAN_DETECTION: No orphan files (unindexed but should be indexed)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate, GateResult

logger = logging.getLogger(__name__)


class DirectoryCoverageEnforcer(BaseEnforcer):
    """Enforcer for knowledge directory coverage validation.

    Validates that:
    1. All configured knowledge directories are scanned
    2. All relevant file types are being indexed
    3. Index freshness is maintained
    4. No orphan files exist (unindexed files that should be indexed)
    """

    __slots__ = (
        "_required_directories",
        "_supported_extensions",
        "_max_index_age_hours",
        "_base_path",
    )

    # Default configuration
    DEFAULT_REQUIRED_DIRECTORIES = [
        "knowledge",
        "knowledge/aws-bedrock",
        "knowledge/address-standards",
        "knowledge/examples",
        "knowledge/data",
    ]

    DEFAULT_SUPPORTED_EXTENSIONS = {
        ".md", ".txt", ".json", ".yaml", ".yml",
        ".py", ".ts", ".js", ".tsx", ".jsx",
        ".html", ".css", ".xml", ".csv",
    }

    DEFAULT_MAX_INDEX_AGE_HOURS = 24

    def __init__(
        self,
        enabled: bool = True,
        strict_mode: bool = False,
        base_path: str | Path | None = None,
        required_directories: list[str] | None = None,
        supported_extensions: set[str] | None = None,
        max_index_age_hours: int = DEFAULT_MAX_INDEX_AGE_HOURS,
    ):
        """Initialize DirectoryCoverageEnforcer.

        Args:
            enabled: Whether enforcer is active.
            strict_mode: If True, any gate failure fails entire check.
            base_path: Base path for directory resolution.
            required_directories: Directories that must be scanned.
            supported_extensions: File extensions to index.
            max_index_age_hours: Maximum age of index before stale.
        """
        super().__init__(
            name="DirectoryCoverageEnforcer",
            enabled=enabled,
            strict_mode=strict_mode,
        )
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._required_directories = required_directories or self.DEFAULT_REQUIRED_DIRECTORIES
        self._supported_extensions = supported_extensions or self.DEFAULT_SUPPORTED_EXTENSIONS
        self._max_index_age_hours = max_index_age_hours

    def get_gates(self) -> list[EnforcerGate]:
        """Get list of gates this enforcer checks."""
        return [
            EnforcerGate.DIRECTORY_COVERAGE_COMPLETE,
            EnforcerGate.FILE_TYPE_SUPPORT,
            EnforcerGate.INDEX_FRESHNESS,
            EnforcerGate.ORPHAN_DETECTION,
        ]

    async def enforce(self, context: dict[str, Any]) -> EnforcerResult:
        """Run directory coverage validation gates.

        Args:
            context: Dictionary containing:
                - indexed_files: Set/list of files currently indexed
                - indexed_directories: Directories that have been scanned
                - index_timestamp: When index was last updated
                - file_type_stats: Dict of extension -> count

        Returns:
            EnforcerResult with gate results and recommendations.
        """
        if not self._enabled:
            return EnforcerResult(
                enforcer_name=self._name,
                passed=True,
                quality_score=1.0,
            )

        gates_passed: list[GateResult] = []
        gates_failed: list[GateResult] = []

        # Extract context values
        indexed_files = set(context.get("indexed_files", []))
        indexed_directories = set(context.get("indexed_directories", []))
        index_timestamp = context.get("index_timestamp")
        file_type_stats = context.get("file_type_stats", {})

        # Gate 1: DIRECTORY_COVERAGE_COMPLETE
        gate1 = self._check_directory_coverage(indexed_directories)
        if gate1.passed:
            gates_passed.append(gate1)
        else:
            gates_failed.append(gate1)

        # Gate 2: FILE_TYPE_SUPPORT
        gate2 = self._check_file_type_support(file_type_stats)
        if gate2.passed:
            gates_passed.append(gate2)
        else:
            gates_failed.append(gate2)

        # Gate 3: INDEX_FRESHNESS
        gate3 = self._check_index_freshness(index_timestamp)
        if gate3.passed:
            gates_passed.append(gate3)
        else:
            gates_failed.append(gate3)

        # Gate 4: ORPHAN_DETECTION
        gate4 = await self._check_orphan_files(indexed_files)
        if gate4.passed:
            gates_passed.append(gate4)
        else:
            gates_failed.append(gate4)

        return self._create_result(gates_passed, gates_failed)

    def _check_directory_coverage(
        self, indexed_directories: set[str]
    ) -> GateResult:
        """Check if all required directories are covered."""
        missing_directories: list[str] = []
        covered_directories: list[str] = []

        for required_dir in self._required_directories:
            full_path = self._base_path / required_dir

            # Check if directory exists first
            if not full_path.exists():
                # Directory doesn't exist - not a failure, just skip
                continue

            # Check if it's been indexed (normalize paths for comparison)
            normalized_required = str(full_path.resolve())
            is_covered = any(
                normalized_required in str(Path(idx).resolve())
                or str(Path(idx).resolve()) in normalized_required
                for idx in indexed_directories
            )

            if is_covered:
                covered_directories.append(required_dir)
            else:
                missing_directories.append(required_dir)

        if missing_directories:
            return self._gate_fail(
                EnforcerGate.DIRECTORY_COVERAGE_COMPLETE,
                f"Missing directories: {', '.join(missing_directories)}",
                details={
                    "missing": missing_directories,
                    "covered": covered_directories,
                    "required": self._required_directories,
                },
            )

        coverage = len(covered_directories) / len(self._required_directories) if self._required_directories else 1.0

        return self._gate_pass(
            EnforcerGate.DIRECTORY_COVERAGE_COMPLETE,
            f"All {len(covered_directories)} required directories covered",
            actual_value=coverage,
            details={
                "covered": covered_directories,
                "total_required": len(self._required_directories),
            },
        )

    def _check_file_type_support(
        self, file_type_stats: dict[str, int]
    ) -> GateResult:
        """Check if important file types are being indexed."""
        if not file_type_stats:
            return self._gate_fail(
                EnforcerGate.FILE_TYPE_SUPPORT,
                "No file type statistics available",
                details={"recommendation": "Ensure index tracks file types"},
            )

        indexed_extensions = set(file_type_stats.keys())
        missing_important = self._supported_extensions - indexed_extensions

        # Only fail if ALL important extensions are missing
        important_missing = {".md", ".txt", ".json", ".py"} & missing_important

        if important_missing == {".md", ".txt", ".json", ".py"}:
            return self._gate_fail(
                EnforcerGate.FILE_TYPE_SUPPORT,
                f"Critical file types not indexed: {', '.join(important_missing)}",
                details={
                    "missing_important": list(important_missing),
                    "indexed_types": list(indexed_extensions),
                },
            )

        coverage_ratio = len(indexed_extensions & self._supported_extensions) / len(self._supported_extensions)

        return self._gate_pass(
            EnforcerGate.FILE_TYPE_SUPPORT,
            f"Indexing {len(indexed_extensions)} file types",
            actual_value=coverage_ratio,
            details={
                "indexed_types": list(indexed_extensions),
                "total_files": sum(file_type_stats.values()),
            },
        )

    def _check_index_freshness(
        self, index_timestamp: float | None
    ) -> GateResult:
        """Check if index is fresh (not stale)."""
        if index_timestamp is None:
            return self._gate_fail(
                EnforcerGate.INDEX_FRESHNESS,
                "No index timestamp available",
                details={"recommendation": "Index may not be initialized"},
            )

        current_time = time.time()
        age_hours = (current_time - index_timestamp) / 3600

        if age_hours > self._max_index_age_hours:
            return self._gate_fail(
                EnforcerGate.INDEX_FRESHNESS,
                f"Index is {age_hours:.1f} hours old (max: {self._max_index_age_hours}h)",
                threshold=float(self._max_index_age_hours),
                actual_value=age_hours,
                details={"recommendation": "Trigger index refresh"},
            )

        return self._gate_pass(
            EnforcerGate.INDEX_FRESHNESS,
            f"Index is {age_hours:.1f} hours old (within {self._max_index_age_hours}h limit)",
            threshold=float(self._max_index_age_hours),
            actual_value=age_hours,
        )

    async def _check_orphan_files(
        self, indexed_files: set[str]
    ) -> GateResult:
        """Check for orphan files (should be indexed but aren't)."""
        orphan_files: list[str] = []
        total_files_scanned = 0

        for required_dir in self._required_directories:
            full_path = self._base_path / required_dir
            if not full_path.exists():
                continue

            try:
                for root, _, files in os.walk(full_path):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext not in self._supported_extensions:
                            continue

                        total_files_scanned += 1
                        file_path = str(Path(root) / file)

                        # Check if file is indexed (normalize for comparison)
                        normalized_path = str(Path(file_path).resolve())
                        is_indexed = any(
                            normalized_path in str(Path(idx).resolve())
                            or str(Path(idx).resolve()) in normalized_path
                            for idx in indexed_files
                        )

                        if not is_indexed:
                            # Only track relative path for readability
                            try:
                                rel_path = Path(file_path).relative_to(self._base_path)
                                orphan_files.append(str(rel_path))
                            except ValueError:
                                orphan_files.append(file_path)
            except PermissionError:
                logger.warning(f"Permission denied scanning {full_path}")
                continue

        if orphan_files:
            # Limit to first 20 orphans for readability
            displayed_orphans = orphan_files[:20]
            return self._gate_fail(
                EnforcerGate.ORPHAN_DETECTION,
                f"Found {len(orphan_files)} unindexed files",
                details={
                    "orphan_files": displayed_orphans,
                    "total_orphans": len(orphan_files),
                    "total_scanned": total_files_scanned,
                    "recommendation": "Run index refresh to include these files",
                },
            )

        return self._gate_pass(
            EnforcerGate.ORPHAN_DETECTION,
            f"No orphan files found (scanned {total_files_scanned} files)",
            details={
                "total_scanned": total_files_scanned,
                "total_indexed": len(indexed_files),
            },
        )

    def get_directory_stats(self) -> dict[str, Any]:
        """Get statistics about directory coverage.

        Returns:
            Dict with directory stats for monitoring.
        """
        stats = {
            "base_path": str(self._base_path),
            "required_directories": self._required_directories,
            "supported_extensions": list(self._supported_extensions),
            "max_index_age_hours": self._max_index_age_hours,
            "existing_directories": [],
            "missing_directories": [],
        }

        for dir_path in self._required_directories:
            full_path = self._base_path / dir_path
            if full_path.exists():
                stats["existing_directories"].append(dir_path)
            else:
                stats["missing_directories"].append(dir_path)

        return stats
