"""File watcher adapter for address data ingestion.

Monitors directories for new CSV/JSON files and streams to pipeline.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator
from collections import deque

from icda.ingestion.adapters.base_adapter import BaseStreamAdapter, AdapterType
from icda.ingestion.pipeline.ingestion_models import AddressRecord

logger = logging.getLogger(__name__)


class FileWatcherAdapter(BaseStreamAdapter):
    """Adapter that watches directories for address data files.

    Monitors specified directories for new CSV/JSON files,
    parses them, and streams AddressRecords to the pipeline.

    Features:
    - Watch multiple directories
    - Support CSV and JSON formats
    - Track processed files to avoid duplicates
    - Configurable polling interval
    - Move processed files to archive directory
    """

    __slots__ = (
        "_watch_dirs",
        "_archive_dir",
        "_poll_interval",
        "_file_patterns",
        "_processing",
        "_processed_files",
        "_pending_files",
        "_records_queue",
        "_watch_task",
        "_file_stats",
    )

    def __init__(
        self,
        watch_dirs: list[str] | str,
        archive_dir: str | None = None,
        poll_interval: float = 5.0,
        file_patterns: list[str] | None = None,
    ):
        """Initialize file watcher.

        Args:
            watch_dirs: Directory or list of directories to watch.
            archive_dir: Directory to move processed files to.
            poll_interval: Seconds between directory scans.
            file_patterns: File patterns to match (e.g., ["*.csv", "*.json"]).
        """
        if isinstance(watch_dirs, str):
            watch_dirs = [watch_dirs]

        super().__init__(name=f"file_watcher_{watch_dirs[0]}")
        self._watch_dirs = [Path(d) for d in watch_dirs]
        self._archive_dir = Path(archive_dir) if archive_dir else None
        self._poll_interval = poll_interval
        self._file_patterns = file_patterns or ["*.csv", "*.json"]

        self._processing = False
        self._processed_files: set[str] = set()
        self._pending_files: deque[Path] = deque()
        self._records_queue: asyncio.Queue[AddressRecord | None] = asyncio.Queue()
        self._watch_task: asyncio.Task | None = None

        # File-specific stats (separate from base class AdapterStats)
        self._file_stats = {
            "files_discovered": 0,
            "files_processed": 0,
            "files_failed": 0,
            "records_parsed": 0,
            "records_streamed": 0,
        }

    @property
    def adapter_type(self) -> AdapterType:
        """Return adapter type identifier."""
        return AdapterType.FILE_WATCHER

    def has_precomputed_embeddings(self) -> bool:
        """Check if source provides pre-computed embeddings."""
        return False  # Files may or may not have embeddings

    @property
    def source_name(self) -> str:
        """Get source identifier."""
        return f"file_watcher:{','.join(str(d) for d in self._watch_dirs)}"

    @property
    def source_type(self) -> str:
        """Get source type."""
        return "file_watcher"

    async def connect(self) -> bool:
        """Start watching directories.

        Returns:
            True if connected successfully.
        """
        # Validate directories exist
        for watch_dir in self._watch_dirs:
            if not watch_dir.exists():
                logger.warning(f"Watch directory does not exist: {watch_dir}")
                watch_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created watch directory: {watch_dir}")

        # Create archive directory if specified
        if self._archive_dir and not self._archive_dir.exists():
            self._archive_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created archive directory: {self._archive_dir}")

        self._connected = True
        self._processing = True

        # Start background watcher
        self._watch_task = asyncio.create_task(self._watch_loop())

        logger.info(
            f"File watcher connected, watching: "
            f"{[str(d) for d in self._watch_dirs]}"
        )
        return True

    async def disconnect(self) -> None:
        """Stop watching and clean up."""
        self._processing = False
        self._connected = False

        # Cancel watch task
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        # Signal end of stream
        await self._records_queue.put(None)

        logger.info(
            f"File watcher disconnected: "
            f"{self._file_stats['files_processed']} files, "
            f"{self._file_stats['records_streamed']} records"
        )

    async def read_stream(self) -> AsyncIterator[AddressRecord]:
        """Stream records from watched files.

        Yields:
            AddressRecord instances.
        """
        while self._connected or not self._records_queue.empty():
            try:
                record = await asyncio.wait_for(
                    self._records_queue.get(),
                    timeout=1.0,
                )

                if record is None:
                    break

                self._file_stats["records_streamed"] += 1
                yield record

            except asyncio.TimeoutError:
                continue

    async def read_batch(self, batch_size: int = 1000) -> list[AddressRecord]:
        """Read a batch of records.

        Args:
            batch_size: Max records to return.

        Returns:
            List of AddressRecords.
        """
        records: list[AddressRecord] = []

        while len(records) < batch_size:
            try:
                record = await asyncio.wait_for(
                    self._records_queue.get(),
                    timeout=0.5,
                )

                if record is None:
                    break

                records.append(record)
                self._file_stats["records_streamed"] += 1

            except asyncio.TimeoutError:
                break

        return records

    async def _watch_loop(self) -> None:
        """Background loop that watches for new files."""
        logger.debug("File watcher loop started")

        while self._processing:
            try:
                # Scan directories for new files
                await self._scan_directories()

                # Process pending files
                while self._pending_files and self._processing:
                    file_path = self._pending_files.popleft()
                    await self._process_file(file_path)

                # Wait before next scan
                await asyncio.sleep(self._poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
                await asyncio.sleep(self._poll_interval)

        logger.debug("File watcher loop stopped")

    async def _scan_directories(self) -> None:
        """Scan watch directories for new files."""
        for watch_dir in self._watch_dirs:
            if not watch_dir.exists():
                continue

            for pattern in self._file_patterns:
                for file_path in watch_dir.glob(pattern):
                    if not file_path.is_file():
                        continue

                    file_key = str(file_path.absolute())

                    if file_key not in self._processed_files:
                        self._pending_files.append(file_path)
                        self._processed_files.add(file_key)
                        self._file_stats["files_discovered"] += 1

                        logger.info(f"Discovered new file: {file_path.name}")

    async def _process_file(self, file_path: Path) -> None:
        """Process a single file.

        Args:
            file_path: Path to file to process.
        """
        try:
            logger.info(f"Processing file: {file_path.name}")

            # Determine format from extension
            suffix = file_path.suffix.lower()

            if suffix == ".csv":
                records = await self._parse_csv(file_path)
            elif suffix in (".json", ".jsonl"):
                records = await self._parse_json(file_path)
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return

            # Queue records
            for record in records:
                await self._records_queue.put(record)
                self._file_stats["records_parsed"] += 1

            self._file_stats["files_processed"] += 1

            # Archive file if configured
            if self._archive_dir:
                await self._archive_file(file_path)
            else:
                # Delete processed file
                file_path.unlink()

            logger.info(
                f"Processed {file_path.name}: {len(records)} records"
            )

        except Exception as e:
            self._file_stats["files_failed"] += 1
            logger.error(f"Failed to process {file_path}: {e}")

    async def _parse_csv(self, file_path: Path) -> list[AddressRecord]:
        """Parse CSV file into AddressRecords.

        Args:
            file_path: Path to CSV file.

        Returns:
            List of AddressRecords.
        """
        records: list[AddressRecord] = []

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(
            None,
            self._read_csv_sync,
            file_path,
        )

        for i, row in enumerate(rows):
            record = self._row_to_record(
                row,
                source_id=f"{file_path.stem}_{i}",
                source_file=file_path.name,
            )
            records.append(record)

        return records

    def _read_csv_sync(self, file_path: Path) -> list[dict[str, Any]]:
        """Synchronously read CSV file.

        Args:
            file_path: Path to CSV file.

        Returns:
            List of row dictionaries.
        """
        rows: list[dict[str, Any]] = []

        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up keys and values
                cleaned = {
                    k.strip(): v.strip() if isinstance(v, str) else v
                    for k, v in row.items()
                    if k is not None
                }
                rows.append(cleaned)

        return rows

    async def _parse_json(self, file_path: Path) -> list[AddressRecord]:
        """Parse JSON/JSONL file into AddressRecords.

        Args:
            file_path: Path to JSON file.

        Returns:
            List of AddressRecords.
        """
        records: list[AddressRecord] = []

        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            self._read_json_sync,
            file_path,
        )

        # Handle both single object and array
        if isinstance(data, dict):
            data = [data]

        for i, item in enumerate(data):
            record = self._row_to_record(
                item,
                source_id=item.get("id") or f"{file_path.stem}_{i}",
                source_file=file_path.name,
            )
            records.append(record)

        return records

    def _read_json_sync(self, file_path: Path) -> list[dict[str, Any]]:
        """Synchronously read JSON file.

        Args:
            file_path: Path to JSON file.

        Returns:
            List of data objects.
        """
        suffix = file_path.suffix.lower()

        with open(file_path, "r", encoding="utf-8") as f:
            if suffix == ".jsonl":
                # JSON Lines format
                return [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]
            else:
                # Regular JSON
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return [data]

    def _row_to_record(
        self,
        row: dict[str, Any],
        source_id: str,
        source_file: str,
    ) -> AddressRecord:
        """Convert row data to AddressRecord.

        Args:
            row: Raw row data.
            source_id: Record identifier.
            source_file: Source file name.

        Returns:
            AddressRecord instance.
        """
        # Try to extract raw address
        raw_address = None
        for key in [
            "address", "full_address", "street_address",
            "Address", "ADDRESS", "FullAddress",
        ]:
            if key in row:
                raw_address = str(row[key])
                break

        # Check for embedding
        embedding = None
        for key in ["embedding", "vector", "embeddings"]:
            if key in row:
                emb = row[key]
                if isinstance(emb, list):
                    embedding = [float(x) for x in emb]
                elif isinstance(emb, str):
                    # Try parsing JSON array
                    try:
                        embedding = json.loads(emb)
                    except json.JSONDecodeError:
                        pass
                break

        return AddressRecord(
            source_id=source_id,
            raw_data=row,
            raw_address=raw_address,
            precomputed_embedding=embedding,
            source_metadata={
                "source": "file_watcher",
                "file": source_file,
                "processed_at": time.time(),
            },
        )

    async def _archive_file(self, file_path: Path) -> None:
        """Move processed file to archive directory.

        Args:
            file_path: File to archive.
        """
        if not self._archive_dir:
            return

        # Add timestamp to avoid collisions
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        archive_path = self._archive_dir / archive_name

        # Move file
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: file_path.rename(archive_path),
        )

        logger.debug(f"Archived {file_path.name} to {archive_path}")

    def add_file(self, file_path: str | Path) -> bool:
        """Manually add a file to process.

        Args:
            file_path: Path to file.

        Returns:
            True if added to queue.
        """
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"File does not exist: {path}")
            return False

        file_key = str(path.absolute())

        if file_key in self._processed_files:
            logger.warning(f"File already processed: {path.name}")
            return False

        self._pending_files.append(path)
        self._processed_files.add(file_key)
        self._file_stats["files_discovered"] += 1

        logger.info(f"Manually added file: {path.name}")
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self._file_stats,
            "watch_dirs": [str(d) for d in self._watch_dirs],
            "pending_files": len(self._pending_files),
            "queue_size": self._records_queue.qsize(),
            "connected": self._connected,
        }

    def get_info(self) -> dict[str, Any]:
        """Get adapter information."""
        return {
            "source_name": self.source_name,
            "source_type": self.source_type,
            "watch_dirs": [str(d) for d in self._watch_dirs],
            "archive_dir": str(self._archive_dir) if self._archive_dir else None,
            "poll_interval": self._poll_interval,
            "file_patterns": self._file_patterns,
            "connected": self._connected,
            "stats": self.get_stats(),
        }
