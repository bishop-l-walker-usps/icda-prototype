"""
ICDA Upload Router - FastAPI Endpoints
======================================
Routes for file upload, validation, and address correction.
Integrates with existing ICDA infrastructure.

Author: Bishop Walker / Salt Water Coder
Project: ICDA Prototype
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging

from .upload_handler import StreamingUploadHandler, UploadOrchestrator, UploadMode, FileType
from .document_parser import DocumentParserService
from .address_validator import AddressValidatorService
from .nova_corrector import NovaProAddressCorrector, NovaProClient

logger = logging.getLogger(__name__)


# Pydantic models
class UploadInitRequest(BaseModel):
    filename: str
    total_bytes: int
    content_type: str = "application/octet-stream"
    mode: str = Field(default="index", description="'index' or 'validate'")


class ValidationRequest(BaseModel):
    street: str
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None


class BatchValidationRequest(BaseModel):
    addresses: list[dict]
    include_corrections: bool = True


class AddressCorrectionRequest(BaseModel):
    street: str
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None


# Global services (initialized from main.py)
_upload_handler: StreamingUploadHandler = None
_parser: DocumentParserService = None
_validator: AddressValidatorService = None
_corrector: NovaProAddressCorrector = None
_orchestrator: UploadOrchestrator = None


def configure_upload_services(
    embedding_client=None,
    opensearch_client=None,
    nova_client=None
):
    """Configure upload services with existing ICDA components"""
    global _upload_handler, _parser, _validator, _corrector, _orchestrator
    
    _upload_handler = StreamingUploadHandler()
    _parser = DocumentParserService()
    
    _validator = AddressValidatorService(
        embedding_client=embedding_client,
        opensearch_indexer=opensearch_client
    )
    
    _corrector = NovaProAddressCorrector(
        nova_client=nova_client or NovaProClient(),
        embedding_client=embedding_client,
        opensearch_indexer=opensearch_client
    )
    
    _orchestrator = UploadOrchestrator(
        parser_service=_parser,
        rag_indexer=None,  # Use existing knowledge manager
        address_validator=_validator,
        nova_corrector=_corrector
    )
    
    logger.info("Upload services configured")


# Create router
router = APIRouter(prefix="/api/upload", tags=["Upload & Validation"])


@router.post("/init")
async def init_upload(request: UploadInitRequest):
    """Initialize a new chunked upload session"""
    if not _upload_handler:
        raise HTTPException(status_code=503, detail="Upload service not initialized")
    
    try:
        mode = UploadMode(request.mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {request.mode}. Use 'index' or 'validate'"
        )
    
    upload_id = await _upload_handler.start_upload(
        filename=request.filename,
        total_bytes=request.total_bytes,
        content_type=request.content_type
    )
    
    return {
        "upload_id": upload_id,
        "filename": request.filename,
        "mode": mode.value,
        "chunk_size": _upload_handler.CHUNK_SIZE,
        "message": f"Upload initialized. Send chunks to /api/upload/{upload_id}/chunk"
    }


@router.post("/{upload_id}/chunk")
async def upload_chunk(
    upload_id: str,
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...)
):
    """Upload a chunk of a file"""
    if not _upload_handler:
        raise HTTPException(status_code=503, detail="Upload service not initialized")
    
    if upload_id not in _upload_handler.active_uploads:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    chunk_data = await chunk.read()
    
    progress = await _upload_handler.receive_chunk(
        upload_id=upload_id,
        chunk=chunk_data,
        chunk_index=chunk_index
    )
    
    return {
        "upload_id": upload_id,
        "chunk_index": chunk_index,
        "bytes_received": progress.bytes_received,
        "percent_complete": progress.percent_complete,
        "status": progress.status
    }


@router.post("/direct")
async def direct_upload(
    file: UploadFile = File(...),
    mode: str = Form(default="validate")
):
    """Direct file upload with immediate processing (for smaller files)"""
    if not _upload_handler or not _orchestrator:
        raise HTTPException(status_code=503, detail="Upload service not initialized")
    
    try:
        upload_mode = UploadMode(mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Use 'index' or 'validate'"
        )
    
    # Get file size
    file.file.seek(0, 2)
    total_bytes = file.file.tell()
    file.file.seek(0)
    
    # Initialize upload
    upload_id = await _upload_handler.start_upload(
        filename=file.filename,
        total_bytes=total_bytes,
        content_type=file.content_type
    )
    
    # Write entire file
    content = await file.read()
    await _upload_handler.receive_chunk(upload_id, content, 0)
    
    # Process
    result = await _orchestrator.process_upload(
        upload_id=upload_id,
        mode=upload_mode
    )
    
    return result.to_dict()


@router.post("/{upload_id}/process")
async def process_upload(upload_id: str, mode: str = "validate"):
    """Process a completed upload"""
    if not _upload_handler or not _orchestrator:
        raise HTTPException(status_code=503, detail="Upload service not initialized")
    
    progress = _upload_handler.get_progress(upload_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    if progress.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Upload not complete. Status: {progress.status}"
        )
    
    try:
        upload_mode = UploadMode(mode.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Use 'index' or 'validate'"
        )
    
    result = await _orchestrator.process_upload(
        upload_id=upload_id,
        mode=upload_mode
    )
    
    return result.to_dict()


@router.get("/{upload_id}/status")
async def get_upload_status(upload_id: str):
    """Get status of an upload"""
    if not _upload_handler:
        raise HTTPException(status_code=503, detail="Upload service not initialized")
    
    progress = _upload_handler.get_progress(upload_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return progress.to_dict()


# Validation endpoints
validation_router = APIRouter(prefix="/api/validate", tags=["Address Validation"])


@validation_router.post("/address")
async def validate_address(request: ValidationRequest):
    """Validate a single address"""
    if not _validator:
        raise HTTPException(status_code=503, detail="Validation service not initialized")
    
    address = {
        "street": request.street,
        "city": request.city,
        "state": request.state,
        "zip": request.zip
    }
    
    result = await _validator.validate(address)
    return result.to_dict()


@validation_router.post("/batch")
async def validate_batch(request: BatchValidationRequest):
    """Validate multiple addresses in batch"""
    if not _validator or not _corrector:
        raise HTTPException(status_code=503, detail="Validation service not initialized")
    
    results = []
    valid_count = 0
    invalid_count = 0
    
    for i, addr in enumerate(request.addresses):
        validation = await _validator.validate(addr)
        
        result_entry = {
            "index": i,
            "original": addr,
            "is_valid": validation.is_valid,
            "confidence": validation.confidence,
            "errors": [e.message for e in validation.errors]
        }
        
        if validation.is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            
            if request.include_corrections:
                corrections = await _corrector.suggest_corrections(
                    original_address=addr,
                    validation_context=validation.to_dict()
                )
                result_entry["suggestions"] = corrections.suggestions[:3]
        
        results.append(result_entry)
    
    return {
        "total_processed": len(request.addresses),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "results": results
    }


@validation_router.post("/correct")
async def correct_address(request: AddressCorrectionRequest):
    """Get correction suggestions for an address"""
    if not _validator or not _corrector:
        raise HTTPException(status_code=503, detail="Validation service not initialized")
    
    address = {
        "street": request.street,
        "city": request.city,
        "state": request.state,
        "zip": request.zip
    }
    
    # First validate
    validation = await _validator.validate(address)
    
    # Get corrections
    corrections = await _corrector.suggest_corrections(
        original_address=address,
        validation_context=validation.to_dict()
    )
    
    return {
        "original": address,
        "validation": validation.to_dict(),
        "corrections": corrections.to_dict()
    }


# Combined router for easy inclusion
def create_upload_validation_router(
    embedding_client=None,
    opensearch_client=None,
    nova_client=None
) -> tuple[APIRouter, APIRouter]:
    """
    Create and configure the upload and validation routers.
    
    Usage in main.py:
        from icda.upload.router import create_upload_validation_router
        
        upload_router, validation_router = create_upload_validation_router(
            embedding_client=_embedder,
            opensearch_client=_vector_index.client if _vector_index else None,
            nova_client=None
        )
        app.include_router(upload_router)
        app.include_router(validation_router)
    """
    configure_upload_services(
        embedding_client=embedding_client,
        opensearch_client=opensearch_client,
        nova_client=nova_client
    )
    
    return router, validation_router
