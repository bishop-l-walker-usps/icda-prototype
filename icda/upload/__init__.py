"""
ICDA Upload & Validation Services
==================================
Complete upload, parsing, indexing, validation, and correction pipeline.

Author: Bishop Walker / Salt Water Coder
Project: ICDA Prototype v2.0
"""

from .upload_handler import (
    StreamingUploadHandler,
    UploadOrchestrator,
    UploadMode,
    UploadProgress,
    UploadResult,
    FileType
)

from .document_parser import (
    DocumentParserService,
    ParsedRecord
)

from .address_validator import (
    AddressValidatorService,
    ValidationResult,
    ValidationError,
    USStateValidator,
    ZipCodeValidator
)

from .nova_corrector import (
    NovaProAddressCorrector,
    NovaProClient,
    AddressCorrection,
    CorrectionResult,
    CorrectionSource
)

__version__ = "2.0.0"
__author__ = "Bishop Walker / Salt Water Coder"

__all__ = [
    # Upload
    "StreamingUploadHandler",
    "UploadOrchestrator",
    "UploadMode",
    "UploadProgress",
    "UploadResult",
    "FileType",
    
    # Parser
    "DocumentParserService",
    "ParsedRecord",
    
    # Validator
    "AddressValidatorService",
    "ValidationResult",
    "ValidationError",
    "USStateValidator",
    "ZipCodeValidator",
    
    # Corrector
    "NovaProAddressCorrector",
    "NovaProClient",
    "AddressCorrection",
    "CorrectionResult",
    "CorrectionSource",
]
