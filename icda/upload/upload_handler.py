"""
ICDA Upload Handler - Unlimited File Size Support
==================================================
Handles streaming uploads of any size, routes to appropriate processing
pipeline based on upload mode (index vs validate).

Author: Bishop Walker / Salt Water Coder
Project: ICDA Prototype
"""

import asyncio
import hashlib
import tempfile
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import aiofiles
import logging

logger = logging.getLogger(__name__)


class UploadMode(str, Enum):
    """Upload operation modes"""
    INDEX = "index"      # Parse and index into RAG + Titan
    VALIDATE = "validate"  # Use RAG index to validate addresses


class FileType(str, Enum):
    """Supported file types"""
    CSV = "csv"
    EXCEL = "xlsx"
    EXCEL_OLD = "xls"
    JSON = "json"
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    YAML = "yaml"
    XML = "xml"


@dataclass
class UploadProgress:
    """Track upload progress for large files"""
    upload_id: str
    filename: str
    total_bytes: int
    bytes_received: int = 0
    status: str = "uploading"
    started_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    
    @property
    def percent_complete(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return round((self.bytes_received / self.total_bytes) * 100, 2)
    
    def to_dict(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "total_bytes": self.total_bytes,
            "bytes_received": self.bytes_received,
            "percent_complete": self.percent_complete,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "error": self.error
        }


@dataclass
class UploadResult:
    """Result of upload processing"""
    upload_id: str
    success: bool
    mode: UploadMode
    filename: str
    file_type: FileType
    records_processed: int = 0
    records_indexed: int = 0
    records_validated: int = 0
    validation_errors: list = field(default_factory=list)
    address_corrections: list = field(default_factory=list)
    processing_time_ms: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "success": self.success,
            "mode": self.mode.value,
            "filename": self.filename,
            "file_type": self.file_type.value,
            "records_processed": self.records_processed,
            "records_indexed": self.records_indexed,
            "records_validated": self.records_validated,
            "validation_errors": self.validation_errors,
            "address_corrections": self.address_corrections,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error
        }


class StreamingUploadHandler:
    """
    Handles streaming uploads of any file size.
    Uses chunked transfer to avoid memory issues with large files.
    """
    
    # Chunk size for streaming (1MB chunks)
    CHUNK_SIZE = 1024 * 1024
    
    # Temp directory for large files
    TEMP_DIR = Path(tempfile.gettempdir()) / "icda_uploads"
    
    def __init__(self):
        self.active_uploads: dict[str, UploadProgress] = {}
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    async def start_upload(
        self,
        filename: str,
        total_bytes: int,
        content_type: str
    ) -> str:
        """
        Initialize a new upload session.
        Returns upload_id for tracking.
        """
        upload_id = str(uuid.uuid4())
        
        self.active_uploads[upload_id] = UploadProgress(
            upload_id=upload_id,
            filename=filename,
            total_bytes=total_bytes
        )
        
        logger.info(f"Started upload {upload_id}: {filename} ({total_bytes} bytes)")
        return upload_id
    
    async def receive_chunk(
        self,
        upload_id: str,
        chunk: bytes,
        chunk_index: int
    ) -> UploadProgress:
        """
        Receive and store a chunk of the upload.
        Streams to disk to handle unlimited file sizes.
        """
        if upload_id not in self.active_uploads:
            raise ValueError(f"Unknown upload_id: {upload_id}")
        
        progress = self.active_uploads[upload_id]
        temp_path = self.TEMP_DIR / f"{upload_id}.tmp"
        
        try:
            # Append chunk to temp file
            async with aiofiles.open(temp_path, "ab") as f:
                await f.write(chunk)
            
            progress.bytes_received += len(chunk)
            
            # Check if upload is complete
            if progress.bytes_received >= progress.total_bytes:
                progress.status = "complete"
                logger.info(f"Upload {upload_id} complete: {progress.bytes_received} bytes")
            
            return progress
            
        except Exception as e:
            progress.status = "error"
            progress.error = str(e)
            logger.error(f"Upload {upload_id} error: {e}")
            raise
    
    async def stream_upload(
        self,
        upload_id: str,
        stream: AsyncIterator[bytes],
        progress_callback: Optional[Callable[[UploadProgress], None]] = None
    ) -> Path:
        """
        Stream an entire upload to disk.
        Returns path to the completed file.
        """
        if upload_id not in self.active_uploads:
            raise ValueError(f"Unknown upload_id: {upload_id}")
        
        progress = self.active_uploads[upload_id]
        temp_path = self.TEMP_DIR / f"{upload_id}.tmp"
        
        try:
            async with aiofiles.open(temp_path, "wb") as f:
                async for chunk in stream:
                    await f.write(chunk)
                    progress.bytes_received += len(chunk)
                    
                    if progress_callback:
                        progress_callback(progress)
            
            progress.status = "complete"
            return temp_path
            
        except Exception as e:
            progress.status = "error"
            progress.error = str(e)
            raise
    
    def get_progress(self, upload_id: str) -> Optional[UploadProgress]:
        """Get current upload progress"""
        return self.active_uploads.get(upload_id)
    
    def get_temp_path(self, upload_id: str) -> Path:
        """Get the temp file path for a completed upload"""
        return self.TEMP_DIR / f"{upload_id}.tmp"
    
    async def cleanup(self, upload_id: str):
        """Clean up temp files after processing"""
        temp_path = self.TEMP_DIR / f"{upload_id}.tmp"
        if temp_path.exists():
            temp_path.unlink()
        
        if upload_id in self.active_uploads:
            del self.active_uploads[upload_id]
    
    def detect_file_type(self, filename: str, content_type: str = None) -> FileType:
        """Detect file type from filename or content type"""
        ext = Path(filename).suffix.lower().lstrip(".")
        
        type_map = {
            "csv": FileType.CSV,
            "xlsx": FileType.EXCEL,
            "xls": FileType.EXCEL_OLD,
            "json": FileType.JSON,
            "pdf": FileType.PDF,
            "docx": FileType.DOCX,
            "txt": FileType.TXT,
            "md": FileType.MD,
            "yaml": FileType.YAML,
            "yml": FileType.YAML,
            "xml": FileType.XML
        }
        
        if ext in type_map:
            return type_map[ext]
        
        # Fallback to content type
        content_map = {
            "text/csv": FileType.CSV,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileType.EXCEL,
            "application/vnd.ms-excel": FileType.EXCEL_OLD,
            "application/json": FileType.JSON,
            "application/pdf": FileType.PDF,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
            "text/plain": FileType.TXT,
            "text/markdown": FileType.MD
        }
        
        if content_type in content_map:
            return content_map[content_type]
        
        raise ValueError(f"Unsupported file type: {filename} ({content_type})")


class UploadOrchestrator:
    """
    Orchestrates the full upload pipeline:
    1. Receive file (streaming for large files)
    2. Parse content based on file type
    3. Route to INDEX or VALIDATE pipeline
    4. Return results with address corrections and confidence scores
    """
    
    def __init__(
        self,
        parser_service,
        rag_indexer,
        address_validator,
        nova_corrector
    ):
        self.upload_handler = StreamingUploadHandler()
        self.parser = parser_service
        self.rag_indexer = rag_indexer
        self.address_validator = address_validator
        self.nova_corrector = nova_corrector
    
    async def process_upload(
        self,
        upload_id: str,
        mode: UploadMode,
        progress_callback: Optional[Callable] = None
    ) -> UploadResult:
        """
        Process a completed upload based on mode.
        """
        import time
        start_time = time.time()
        
        progress = self.upload_handler.get_progress(upload_id)
        if not progress:
            return UploadResult(
                upload_id=upload_id,
                success=False,
                mode=mode,
                filename="unknown",
                file_type=FileType.TXT,
                error="Upload not found"
            )
        
        file_path = self.upload_handler.get_temp_path(upload_id)
        file_type = self.upload_handler.detect_file_type(progress.filename)
        
        try:
            # Parse the file
            records = await self.parser.parse_file(file_path, file_type)
            records_processed = len(records)
            
            if mode == UploadMode.INDEX:
                # Index into RAG + Titan
                result = await self._process_index(records, progress_callback)
            else:
                # Validate using RAG index
                result = await self._process_validate(records, progress_callback)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return UploadResult(
                upload_id=upload_id,
                success=True,
                mode=mode,
                filename=progress.filename,
                file_type=file_type,
                records_processed=records_processed,
                records_indexed=result.get("indexed", 0),
                records_validated=result.get("validated", 0),
                validation_errors=result.get("errors", []),
                address_corrections=result.get("corrections", []),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Processing error for {upload_id}: {e}")
            return UploadResult(
                upload_id=upload_id,
                success=False,
                mode=mode,
                filename=progress.filename,
                file_type=file_type,
                error=str(e)
            )
        finally:
            await self.upload_handler.cleanup(upload_id)
    
    async def _process_index(
        self,
        records: list[dict],
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        Index records into RAG and Titan.
        """
        indexed_count = 0
        
        # Batch for efficiency
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            # Index each record
            for record in batch:
                await self.rag_indexer.index_record(record)
                indexed_count += 1
            
            if progress_callback:
                progress_callback({
                    "stage": "indexing",
                    "processed": indexed_count,
                    "total": len(records)
                })
        
        return {"indexed": indexed_count}
    
    async def _process_validate(
        self,
        records: list[dict],
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        Validate records using RAG index and Nova Pro corrections.
        """
        validated = 0
        errors = []
        corrections = []
        
        for i, record in enumerate(records):
            # Extract address fields from record
            address_data = self._extract_address(record)
            
            if not address_data:
                validated += 1
                continue
            
            # Validate against RAG index
            validation_result = await self.address_validator.validate(address_data)
            
            if validation_result.is_valid:
                validated += 1
            else:
                # Get Nova Pro corrections with confidence scores
                correction_result = await self.nova_corrector.suggest_corrections(
                    original_address=address_data,
                    validation_context=validation_result.context
                )
                
                errors.append({
                    "row": i + 1,
                    "original": address_data,
                    "error": validation_result.error_message,
                    "corrections": correction_result.suggestions
                })
                
                corrections.extend([
                    {
                        "row": i + 1,
                        "original": address_data,
                        **suggestion
                    }
                    for suggestion in correction_result.suggestions
                ])
            
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback({
                    "stage": "validating",
                    "processed": i + 1,
                    "total": len(records)
                })
        
        return {
            "validated": validated,
            "errors": errors,
            "corrections": corrections
        }
    
    def _extract_address(self, record: dict) -> Optional[dict]:
        """Extract address fields from a record"""
        # Common address field patterns
        address_fields = {
            "street": ["street", "address", "address1", "street_address", "addr"],
            "street2": ["street2", "address2", "apt", "suite", "unit"],
            "city": ["city", "town", "municipality"],
            "state": ["state", "st", "province", "region"],
            "zip": ["zip", "zipcode", "zip_code", "postal", "postal_code"],
            "country": ["country", "nation"]
        }
        
        result = {}
        record_lower = {k.lower(): v for k, v in record.items()}
        
        for field_type, patterns in address_fields.items():
            for pattern in patterns:
                if pattern in record_lower and record_lower[pattern]:
                    result[field_type] = str(record_lower[pattern]).strip()
                    break
        
        # Return None if no address fields found
        if not any(k in result for k in ["street", "city", "zip"]):
            return None
        
        return result
