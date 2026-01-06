"""Data Source Manager - handles multiple data sources with fallback.

Maintains backward compatibility with customer_data.json while supporting
C library metadata exports for production use.

Priority Order:
    1. customer_data.json (if exists) - for local testing
    2. C library export file (if exists) - for production batch
    3. C library REST API (if configured) - for real-time

This ensures existing tests continue to work while enabling production data.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from icda.indexes.customers_index import CustomerRecord

logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """Available data source types."""
    CUSTOMER_JSON = "customer_json"      # customer_data.json (testing)
    C_LIBRARY_FILE = "c_library_file"    # C library export file
    C_LIBRARY_API = "c_library_api"      # C library REST API
    NONE = "none"


@dataclass(slots=True)
class DataSourceStatus:
    """Status of a data source."""
    source_type: DataSourceType
    available: bool
    path: Optional[str] = None
    record_count: int = 0
    message: str = ""


class DataSourceManager:
    """Manages data sources with automatic fallback.

    Usage:
        manager = DataSourceManager(
            project_root=Path("."),
            c_library_export_path="data/c_library_export.json",
            c_library_api_url="http://java-api/export",
        )

        # Check what's available
        status = manager.detect_sources()
        print(status)

        # Load from best available source
        records = await manager.load_records()
    """

    # Default file names
    CUSTOMER_DATA_FILE = "customer_data.json"
    C_LIBRARY_EXPORT_FILE = "c_library_export.json"

    def __init__(
        self,
        project_root: Path | str = Path("."),
        c_library_export_path: Optional[str] = None,
        c_library_api_url: Optional[str] = None,
    ):
        """Initialize the data source manager.

        Args:
            project_root: Project root directory.
            c_library_export_path: Path to C library export file.
            c_library_api_url: URL for C library REST API.
        """
        self.project_root = Path(project_root)
        self.c_library_export_path = c_library_export_path
        self.c_library_api_url = c_library_api_url

        # Standard paths
        self.customer_data_path = self.project_root / self.CUSTOMER_DATA_FILE

    def detect_sources(self) -> dict[str, DataSourceStatus]:
        """Detect all available data sources.

        Returns:
            Dict of source type -> status.
        """
        sources = {}

        # Check customer_data.json
        sources["customer_json"] = self._check_customer_json()

        # Check C library export file
        sources["c_library_file"] = self._check_c_library_file()

        # Check C library API
        sources["c_library_api"] = self._check_c_library_api()

        return sources

    def _check_customer_json(self) -> DataSourceStatus:
        """Check if customer_data.json exists and is valid."""
        if not self.customer_data_path.exists():
            return DataSourceStatus(
                source_type=DataSourceType.CUSTOMER_JSON,
                available=False,
                path=str(self.customer_data_path),
                message="File not found",
            )

        try:
            with open(self.customer_data_path, "r") as f:
                data = json.load(f)
            count = len(data) if isinstance(data, list) else 0
            return DataSourceStatus(
                source_type=DataSourceType.CUSTOMER_JSON,
                available=True,
                path=str(self.customer_data_path),
                record_count=count,
                message=f"Found {count} customer records (testing data)",
            )
        except Exception as e:
            return DataSourceStatus(
                source_type=DataSourceType.CUSTOMER_JSON,
                available=False,
                path=str(self.customer_data_path),
                message=f"Error reading file: {e}",
            )

    def _check_c_library_file(self) -> DataSourceStatus:
        """Check if C library export file exists."""
        if not self.c_library_export_path:
            return DataSourceStatus(
                source_type=DataSourceType.C_LIBRARY_FILE,
                available=False,
                message="No export path configured",
            )

        path = Path(self.c_library_export_path)
        if not path.exists():
            return DataSourceStatus(
                source_type=DataSourceType.C_LIBRARY_FILE,
                available=False,
                path=str(path),
                message="Export file not found",
            )

        try:
            with open(path, "r") as f:
                data = json.load(f)
            # Handle wrapped format
            if isinstance(data, dict):
                data = data.get("addresses") or data.get("data") or data.get("records") or []
            count = len(data) if isinstance(data, list) else 0
            return DataSourceStatus(
                source_type=DataSourceType.C_LIBRARY_FILE,
                available=True,
                path=str(path),
                record_count=count,
                message=f"Found {count} C library address records",
            )
        except Exception as e:
            return DataSourceStatus(
                source_type=DataSourceType.C_LIBRARY_FILE,
                available=False,
                path=str(path),
                message=f"Error reading file: {e}",
            )

    def _check_c_library_api(self) -> DataSourceStatus:
        """Check if C library API is configured."""
        if not self.c_library_api_url:
            return DataSourceStatus(
                source_type=DataSourceType.C_LIBRARY_API,
                available=False,
                message="No API URL configured",
            )

        # Just check if URL is configured - actual connectivity checked at runtime
        return DataSourceStatus(
            source_type=DataSourceType.C_LIBRARY_API,
            available=True,
            path=self.c_library_api_url,
            message=f"API configured: {self.c_library_api_url}",
        )

    def get_active_source(self) -> DataSourceType:
        """Get the currently active data source (highest priority available).

        Priority: customer_json > c_library_file > c_library_api > none
        """
        sources = self.detect_sources()

        # Priority order for backward compatibility
        if sources["customer_json"].available:
            return DataSourceType.CUSTOMER_JSON

        if sources["c_library_file"].available:
            return DataSourceType.C_LIBRARY_FILE

        if sources["c_library_api"].available:
            return DataSourceType.C_LIBRARY_API

        return DataSourceType.NONE

    def load_customer_json(self) -> list[CustomerRecord]:
        """Load records from customer_data.json (testing data)."""
        if not self.customer_data_path.exists():
            return []

        try:
            with open(self.customer_data_path, "r") as f:
                data = json.load(f)

            records = []
            for item in data:
                records.append(CustomerRecord(
                    crid=item.get("crid", ""),
                    name=item.get("name", ""),
                    address=item.get("address", ""),
                    city=item.get("city", ""),
                    state=item.get("state", ""),
                    zip_code=item.get("zip", item.get("zip_code", "")),
                    customer_type=item.get("customer_type", "residential"),
                    status=item.get("status", "active"),
                    move_count=item.get("move_count", 0),
                    last_move=item.get("last_move"),
                    created_date=item.get("created_date"),
                ))

            logger.info(f"Loaded {len(records)} records from customer_data.json")
            return records

        except Exception as e:
            logger.error(f"Error loading customer_data.json: {e}")
            return []

    def load_c_library_file(self) -> list[CustomerRecord]:
        """Load records from C library export file."""
        if not self.c_library_export_path:
            return []

        path = Path(self.c_library_export_path)
        if not path.exists():
            return []

        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Handle wrapped format
            if isinstance(data, dict):
                data = data.get("addresses") or data.get("data") or data.get("records") or []

            records = []
            for i, item in enumerate(data):
                # Map C library fields to CustomerRecord
                address_id = (
                    item.get("address_id") or
                    item.get("id") or
                    item.get("crid") or
                    f"ADDR-{i+1:06d}"
                )
                records.append(CustomerRecord(
                    crid=str(address_id),
                    name="",  # C library addresses don't have names
                    address=item.get("address") or item.get("street") or "",
                    city=item.get("city", ""),
                    state=(item.get("state") or item.get("st") or "").upper(),
                    zip_code=str(item.get("zip_code") or item.get("zip") or ""),
                    customer_type=(item.get("address_type") or item.get("type") or "residential").lower(),
                    status="active",
                ))

            logger.info(f"Loaded {len(records)} records from C library export")
            return records

        except Exception as e:
            logger.error(f"Error loading C library export: {e}")
            return []

    async def load_c_library_api(self, headers: Optional[dict] = None) -> list[CustomerRecord]:
        """Load records from C library REST API."""
        if not self.c_library_api_url:
            return []

        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.c_library_api_url, headers=headers)
                response.raise_for_status()
                data = response.json()

            # Handle wrapped format
            if isinstance(data, dict):
                data = data.get("addresses") or data.get("data") or data.get("records") or []

            records = []
            for i, item in enumerate(data):
                address_id = (
                    item.get("address_id") or
                    item.get("id") or
                    f"ADDR-{i+1:06d}"
                )
                records.append(CustomerRecord(
                    crid=str(address_id),
                    name="",
                    address=item.get("address") or item.get("street") or "",
                    city=item.get("city", ""),
                    state=(item.get("state") or "").upper(),
                    zip_code=str(item.get("zip_code") or item.get("zip") or ""),
                    customer_type=(item.get("address_type") or item.get("type") or "residential").lower(),
                    status="active",
                ))

            logger.info(f"Loaded {len(records)} records from C library API")
            return records

        except Exception as e:
            logger.error(f"Error loading from C library API: {e}")
            return []

    async def load_records(self, force_source: Optional[DataSourceType] = None) -> list[CustomerRecord]:
        """Load records from the best available source.

        Args:
            force_source: Force a specific source (overrides priority).

        Returns:
            List of CustomerRecord for embedding pipeline.
        """
        source = force_source or self.get_active_source()

        if source == DataSourceType.CUSTOMER_JSON:
            return self.load_customer_json()
        elif source == DataSourceType.C_LIBRARY_FILE:
            return self.load_c_library_file()
        elif source == DataSourceType.C_LIBRARY_API:
            return await self.load_c_library_api()
        else:
            logger.warning("No data source available")
            return []

    def print_status(self) -> str:
        """Print a human-readable status of all data sources."""
        sources = self.detect_sources()
        active = self.get_active_source()

        lines = [
            "=" * 60,
            "DATA SOURCE STATUS",
            "=" * 60,
        ]

        for name, status in sources.items():
            marker = ">>>" if status.source_type == active else "   "
            avail = "YES" if status.available else "NO "
            lines.append(f"{marker} [{avail}] {status.source_type.value}")
            if status.path:
                lines.append(f"         Path: {status.path}")
            if status.record_count:
                lines.append(f"         Records: {status.record_count}")
            lines.append(f"         {status.message}")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"ACTIVE SOURCE: {active.value}")
        lines.append("=" * 60)

        return "\n".join(lines)


def check_data_sources(
    project_root: str = ".",
    c_library_export: Optional[str] = None,
    c_library_api: Optional[str] = None,
) -> dict[str, Any]:
    """Check available data sources and return status.

    Convenience function for scripts and CLI.

    Args:
        project_root: Project root directory.
        c_library_export: Path to C library export file.
        c_library_api: C library API URL.

    Returns:
        Status dict with all source information.
    """
    manager = DataSourceManager(
        project_root=Path(project_root),
        c_library_export_path=c_library_export,
        c_library_api_url=c_library_api,
    )

    sources = manager.detect_sources()
    active = manager.get_active_source()

    return {
        "active_source": active.value,
        "sources": {
            name: {
                "type": status.source_type.value,
                "available": status.available,
                "path": status.path,
                "record_count": status.record_count,
                "message": status.message,
            }
            for name, status in sources.items()
        },
        "summary": manager.print_status(),
    }
