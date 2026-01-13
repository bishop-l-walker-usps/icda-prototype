# ICDA Upload & Validation Module Integration

## Files Installed

```
icda/upload/
├── __init__.py           # Module exports
├── upload_handler.py     # Streaming upload handler (unlimited file sizes)
├── document_parser.py    # Multi-format parser (CSV, Excel, JSON, PDF, DOCX)
├── address_validator.py  # Address validation with confidence scoring
├── nova_corrector.py     # Nova Pro intelligent address correction
├── router.py             # FastAPI routes for integration
└── requirements-upload.txt # Additional dependencies
```

## Installation

1. Install additional dependencies:
```bash
pip install aiofiles pdfplumber
```

2. Add routes to your `main.py`:

```python
# At the top with other imports
from icda.upload.router import create_upload_validation_router

# In the lifespan function, after services are initialized (around line 280)
# Add this after _orchestrator is created:
upload_router, validation_router = create_upload_validation_router(
    embedding_client=_embedder,
    opensearch_client=_vector_index.client if _vector_index else None,
    nova_client=None  # Will auto-create
)

# After app = FastAPI(...) line (around line 410)
# Include the routers:
app.include_router(upload_router)
app.include_router(validation_router)
```

## API Endpoints

### Upload (chunked for large files)
- `POST /api/upload/init` - Initialize upload session
- `POST /api/upload/{id}/chunk` - Upload chunk
- `POST /api/upload/direct` - Direct upload (small files)
- `POST /api/upload/{id}/process` - Process uploaded file
- `GET /api/upload/{id}/status` - Get progress

### Validation
- `POST /api/validate/address` - Validate single address
- `POST /api/validate/batch` - Batch validate with corrections
- `POST /api/validate/correct` - Get correction suggestions

## Usage Examples

### Validate & Correct Address
```bash
curl -X POST http://localhost:8000/api/validate/correct \
  -H "Content-Type: application/json" \
  -d '{
    "street": "101 turkey",
    "city": "",
    "state": "",
    "zip": "22030"
  }'
```

Response:
```json
{
  "original": {"street": "101 turkey", "city": "", "state": "", "zip": "22030"},
  "validation": {
    "is_valid": false,
    "confidence": 0.35,
    "errors": [{"type": "missing_required", "field": "city", "message": "Missing required field: city"}]
  },
  "corrections": {
    "suggestions": [{
      "corrected_address": {
        "street": "101 Turkey Run Rd",
        "city": "McLean",
        "state": "VA",
        "zip": "22030"
      },
      "confidence": 0.92,
      "source": "ensemble",
      "reasoning": "Multiple sources agree: rag_index, nova_inference",
      "changes_made": ["street: '101 turkey' → '101 Turkey Run Rd'", "city: '' → 'McLean'", "state: '' → 'VA'"]
    }]
  }
}
```

### Upload CSV for Validation
```bash
curl -X POST http://localhost:8000/api/upload/direct \
  -F "file=@addresses.csv" \
  -F "mode=validate"
```

## Confidence Scoring

| Factor | Weight | Description |
|--------|--------|-------------|
| Format | 15% | Required fields present |
| State | 15% | Valid US state/territory |
| ZIP | 15% | Valid ZIP format |
| Cross-field | 20% | ZIP matches state range |
| RAG Index | 35% | Vector similarity ≥0.75 |

**Thresholds**: ≥80% (high), 50-79% (medium), <50% (low)
