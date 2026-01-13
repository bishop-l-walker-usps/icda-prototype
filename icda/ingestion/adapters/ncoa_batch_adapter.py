"""NCOA Batch Adapter for C library output.

Reads address data and pre-computed embeddings from files
produced by the C library's daily NCOA processing.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from icda.ingestion.adapters.base_adapter import (
    AdapterType,
    BaseStreamAdapter,
)
from icda.ingestion.pipeline.ingestion_models import AddressRecord

logger = logging.getLogger(__name__)


class NCOABatchAdapter(BaseStreamAdapter):
    """Adapter for C library NCOA batch file output.

    Reads address data from JSON/CSV files and optionally pairs
    with pre-computed embeddings from a binary embedding file.

    File Formats:
    - Address data: JSON (list of objects) or CSV
    - Embeddings: Binary (header + float32 vectors)

    Binary Embedding Format:
        Header (12 bytes):
            - dimension: uint32 (4 bytes)
            - count: uint64 (8 bytes)
        Records (repeated):
            - source_id_length: uint16 (2 bytes)
            - source_id: bytes (variable)
            - embedding: float32[dimension]

    Features:
    - Memory-mapped embedding file for efficiency
    - Checkpoint support for resumable processing
    - Source ID matching between address and embedding data
    """

    __slots__ = (
        "_input_path",
        "_embedding_path",
        "_embedding_dim",
        "_checkpoint_path",
        "_file_format",
        "_last_checkpoint",
        "_embeddings_map",
        "_total_records",
    )

    def __init__(
        self,
        input_path: str,
        embedding_path: str | None = None,
        embedding_dim: int = 1024,
        checkpoint_path: str | None = None,
        file_format: str = "json",
    ):
        """Initialize NCOA batch adapter.

        Args:
            input_path: Path to address data file (JSON/CSV).
            embedding_path: Optional path to binary embedding file.
            embedding_dim: Expected embedding dimension.
            checkpoint_path: Optional path for checkpoint persistence.
            file_format: Input format (json, csv).
        """
        super().__init__("ncoa_batch")
        self._input_path = input_path
        self._embedding_path = embedding_path
        self._embedding_dim = embedding_dim
        self._checkpoint_path = checkpoint_path
        self._file_format = file_format.lower()
        self._last_checkpoint: str | None = None
        self._embeddings_map: dict[str, list[float]] = {}
        self._total_records = 0

    @property
    def adapter_type(self) -> AdapterType:
        return AdapterType.NCOA_BATCH

    async def connect(self) -> bool:
        """Validate paths and load checkpoint.

        Returns:
            True if input file exists and is readable.
        """
        # Check input file
        if not os.path.exists(self._input_path):
            logger.error(f"NCOA input file not found: {self._input_path}")
            return False

        # Check embedding file if specified
        if self._embedding_path and not os.path.exists(self._embedding_path):
            logger.warning(
                f"NCOA embedding file not found: {self._embedding_path}, "
                "proceeding without pre-computed embeddings"
            )
            self._embedding_path = None

        # Load embeddings if available
        if self._embedding_path:
            try:
                await self._load_embeddings()
                logger.info(
                    f"Loaded {len(self._embeddings_map)} pre-computed embeddings"
                )
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                self._embedding_path = None

        # Load checkpoint if exists
        if self._checkpoint_path and os.path.exists(self._checkpoint_path):
            try:
                with open(self._checkpoint_path, "r") as f:
                    checkpoint_data = json.load(f)
                    self._last_checkpoint = checkpoint_data.get("last_id")
                    logger.info(f"Resuming from checkpoint: {self._last_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        self._connected = True
        logger.info(
            f"NCOA adapter connected: {self._input_path} "
            f"(embeddings: {self._embedding_path is not None})"
        )
        return True

    async def disconnect(self) -> None:
        """Clean up resources and save checkpoint."""
        # Save checkpoint if path configured
        if self._checkpoint_path and self._last_checkpoint:
            try:
                checkpoint_dir = os.path.dirname(self._checkpoint_path)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                with open(self._checkpoint_path, "w") as f:
                    json.dump(
                        {
                            "last_id": self._last_checkpoint,
                            "timestamp": datetime.utcnow().isoformat(),
                            "records_processed": self._stats.records_read,
                        },
                        f,
                    )
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        self._embeddings_map.clear()
        self._connected = False

    async def read_stream(self) -> AsyncIterator[AddressRecord]:
        """Stream records from NCOA batch file.

        Yields AddressRecords with pre-computed embeddings if available.
        """
        if not self._connected:
            logger.error("Adapter not connected")
            return

        if self._file_format == "json":
            async for record in self._read_json():
                yield record
        elif self._file_format == "csv":
            async for record in self._read_csv():
                yield record
        else:
            logger.error(f"Unsupported file format: {self._file_format}")

    def has_precomputed_embeddings(self) -> bool:
        """Check if embeddings are available."""
        return len(self._embeddings_map) > 0

    async def _load_embeddings(self) -> None:
        """Load embeddings from binary file into memory map."""
        if not self._embedding_path:
            return

        self._embeddings_map.clear()

        with open(self._embedding_path, "rb") as f:
            # Read header
            header = f.read(12)
            if len(header) < 12:
                raise ValueError("Invalid embedding file: header too short")

            dimension = struct.unpack("<I", header[:4])[0]
            count = struct.unpack("<Q", header[4:12])[0]

            logger.info(
                f"Loading embeddings: dimension={dimension}, count={count}"
            )

            if dimension != self._embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                    f"got {dimension}"
                )

            # Read records
            for _ in range(count):
                # Read source_id length
                id_len_bytes = f.read(2)
                if len(id_len_bytes) < 2:
                    break
                id_len = struct.unpack("<H", id_len_bytes)[0]

                # Read source_id
                source_id = f.read(id_len).decode("utf-8")

                # Read embedding
                embedding_bytes = f.read(dimension * 4)
                embedding = list(struct.unpack(f"<{dimension}f", embedding_bytes))

                self._embeddings_map[source_id] = embedding

    async def _read_json(self) -> AsyncIterator[AddressRecord]:
        """Read records from JSON file."""
        try:
            with open(self._input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            self._total_records = len(data)
            skip_until_checkpoint = self._last_checkpoint is not None
            found_checkpoint = False

            for item in data:
                source_id = self._extract_source_id(item)

                # Handle checkpoint resume
                if skip_until_checkpoint:
                    if source_id == self._last_checkpoint:
                        found_checkpoint = True
                        skip_until_checkpoint = False
                    continue

                # Get embedding if available
                embedding = self._embeddings_map.get(source_id)

                # Build raw address string
                raw_address = self._build_raw_address(item)

                record = AddressRecord(
                    source_id=source_id,
                    raw_data=item,
                    raw_address=raw_address,
                    precomputed_embedding=embedding,
                    source_metadata={
                        "file": self._input_path,
                        "format": "json",
                    },
                )

                self._stats.record_read(
                    has_embedding=embedding is not None,
                    bytes_count=len(json.dumps(item)),
                )
                self._last_checkpoint = source_id

                yield record

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            self._stats.record_error(str(e))
        except Exception as e:
            logger.error(f"Error reading JSON file: {e}")
            self._stats.record_error(str(e))

    async def _read_csv(self) -> AsyncIterator[AddressRecord]:
        """Read records from CSV file."""
        import csv

        try:
            with open(self._input_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                skip_until_checkpoint = self._last_checkpoint is not None

                for row in reader:
                    source_id = self._extract_source_id(row)

                    # Handle checkpoint resume
                    if skip_until_checkpoint:
                        if source_id == self._last_checkpoint:
                            skip_until_checkpoint = False
                        continue

                    # Get embedding if available
                    embedding = self._embeddings_map.get(source_id)

                    # Build raw address string
                    raw_address = self._build_raw_address(row)

                    record = AddressRecord(
                        source_id=source_id,
                        raw_data=dict(row),
                        raw_address=raw_address,
                        precomputed_embedding=embedding,
                        source_metadata={
                            "file": self._input_path,
                            "format": "csv",
                        },
                    )

                    self._stats.record_read(
                        has_embedding=embedding is not None,
                    )
                    self._last_checkpoint = source_id

                    yield record

        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            self._stats.record_error(str(e))

    def _extract_source_id(self, data: dict[str, Any]) -> str:
        """Extract source ID from record data.

        Looks for common ID fields in order of preference.
        """
        id_fields = [
            "id", "source_id", "record_id", "crid", "customer_id",
            "ID", "SOURCE_ID", "RECORD_ID", "CRID", "CUSTOMER_ID",
        ]

        for field in id_fields:
            if field in data and data[field]:
                return str(data[field])

        # Generate ID from hash if no ID field
        import hashlib
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _build_raw_address(self, data: dict[str, Any]) -> str:
        """Build raw address string from data fields.

        Handles various field naming conventions.
        """
        # Common field mappings
        field_mappings = {
            "street": ["street", "address", "street_address", "address1", "ADDRESS"],
            "street2": ["street2", "address2", "unit", "apt", "ADDRESS2"],
            "city": ["city", "CITY"],
            "state": ["state", "STATE"],
            "zip": ["zip", "zip_code", "zipcode", "postal_code", "ZIP"],
        }

        parts = []

        for component, fields in field_mappings.items():
            for field in fields:
                if field in data and data[field]:
                    value = str(data[field]).strip()
                    if value:
                        parts.append(value)
                        break

        return ", ".join(parts) if parts else ""

    async def count_records(self) -> int:
        """Count total records in file."""
        if self._total_records > 0:
            return self._total_records

        if self._file_format == "json":
            with open(self._input_path, "r") as f:
                data = json.load(f)
                self._total_records = len(data) if isinstance(data, list) else 1
        elif self._file_format == "csv":
            import csv
            with open(self._input_path, "r") as f:
                self._total_records = sum(1 for _ in csv.reader(f)) - 1  # minus header

        return self._total_records

    def get_info(self) -> dict[str, Any]:
        """Get adapter information."""
        info = super().get_info()
        info.update({
            "input_path": self._input_path,
            "embedding_path": self._embedding_path,
            "embedding_dim": self._embedding_dim,
            "file_format": self._file_format,
            "embeddings_loaded": len(self._embeddings_map),
            "checkpoint": self._last_checkpoint,
        })
        return info
