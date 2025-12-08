"""
ICDA Prototype - Intelligent Customer Data Access
Run with: uvicorn main:app --reload --port 8000
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from icda.config import cfg
from icda.cache import RedisCache
from icda.embeddings import EmbeddingClient
from icda.vector_index import VectorIndex
from icda.database import CustomerDB
from icda.nova import NovaClient
from icda.router import Router
from icda.session import SessionManager
from icda.address_index import AddressIndex
from icda.address_completer import NovaAddressCompleter
from icda.address_pipeline import AddressPipeline
from icda.address_router import router as address_router, configure_router

load_dotenv()

BASE_DIR = Path(__file__).parent

# Globals
_cache: RedisCache = None
_embedder: EmbeddingClient = None
_vector_index: VectorIndex = None
_db: CustomerDB = None
_nova: NovaClient = None
_router: Router = None
_sessions: SessionManager = None
_address_index: AddressIndex = None
_address_completer: NovaAddressCompleter = None
_address_pipeline: AddressPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache, _embedder, _vector_index, _db, _nova, _router, _sessions
    global _address_index, _address_completer, _address_pipeline

    # Startup
    _cache = RedisCache(cfg.cache_ttl)
    await _cache.connect(cfg.redis_url)

    _sessions = SessionManager(_cache, ttl=3600)  # 1 hour session TTL
    print(f"Session manager initialized (TTL: 1 hour)")

    _embedder = EmbeddingClient(cfg.aws_region, cfg.titan_embed_model, cfg.embed_dimensions)

    _vector_index = VectorIndex(_embedder, cfg.opensearch_index)
    await _vector_index.connect(cfg.opensearch_host, cfg.aws_region)

    _db = CustomerDB(BASE_DIR / "customer_data.json")

    _nova = NovaClient(cfg.aws_region, cfg.nova_model, _db)

    _router = Router(_cache, _vector_index, _db, _nova, _sessions)

    # Initialize address verification pipeline
    _address_index = AddressIndex()
    _address_index.build_from_customers(_db.customers)
    print(f"Address index built with {_address_index.stats()['total_addresses']} addresses")

    _address_completer = NovaAddressCompleter(cfg.aws_region, cfg.nova_model, _address_index)

    _address_pipeline = AddressPipeline(_address_index, _address_completer)
    configure_router(_address_pipeline)

    yield

    # Shutdown
    await _cache.close()
    await _vector_index.close()


app = FastAPI(title="ICDA", version="0.6.0", lifespan=lifespan)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount address verification router
app.include_router(address_router)


class GuardrailSettings(BaseModel):
    pii: bool = True
    financial: bool = True
    credentials: bool = True
    offtopic: bool = True


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    bypass_cache: bool = False
    guardrails: GuardrailSettings | None = None
    session_id: str | None = None  # Pass to maintain conversation context


@app.post("/api/query")
async def query(req: QueryRequest):
    """Process a text query with optional session context."""
    guards = req.guardrails.model_dump() if req.guardrails else None
    return await _router.route(req.query, req.bypass_cache, guards, req.session_id)


ALLOWED_EXTENSIONS = {".json", ".md"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    query: str = Form(default=""),
    bypass_cache: bool = Form(default=False),
    session_id: str | None = Form(default=None),
):
    """
    Upload a .json or .md file for processing.
    Optionally include a query to ask about the file content.
    """
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only {', '.join(ALLOWED_EXTENSIONS)} files allowed")

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB")

    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 encoded")

    # Parse JSON if applicable
    file_data = None
    if ext == ".json":
        try:
            file_data = json.loads(text_content)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid JSON: {e}")

    # Build the query with file context
    if query:
        full_query = f"Based on the uploaded {ext} file content:\n\n{text_content[:5000]}\n\nUser question: {query}"
    else:
        full_query = f"Analyze this {ext} file:\n\n{text_content[:5000]}"

    # Route through the system
    result = await _router.route(full_query, bypass_cache, None, session_id)
    result["file_name"] = file.filename
    result["file_type"] = ext
    result["file_size"] = len(content)

    return result


@app.post("/api/query-with-file")
async def query_with_file(
    query: str = Form(...),
    file: UploadFile | None = File(default=None),
    bypass_cache: bool = Form(default=False),
    session_id: str | None = Form(default=None),
    pii: bool = Form(default=True),
    financial: bool = Form(default=True),
    credentials: bool = Form(default=True),
    offtopic: bool = Form(default=True),
):
    """
    Unified endpoint: accepts text query with optional file upload.
    Supports both .json and .md files.
    """
    guards = {"pii": pii, "financial": financial, "credentials": credentials, "offtopic": offtopic}

    # If no file, just process the text query
    if file is None or file.filename == "":
        return await _router.route(query, bypass_cache, guards, session_id)

    # Validate file
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only {', '.join(ALLOWED_EXTENSIONS)} files allowed")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB")

    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 encoded")

    # Build combined query
    full_query = f"File: {file.filename}\nContent:\n{text_content[:5000]}\n\nQuestion: {query}"

    result = await _router.route(full_query, bypass_cache, guards, session_id)
    result["file_name"] = file.filename
    result["file_type"] = ext

    return result


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "redis": _cache.available if _cache else False,
        "opensearch": _vector_index.available if _vector_index else False,
        "embedder": _embedder.available if _embedder else False,
        "nova": _nova.available if _nova else False,
        "customers": len(_db.customers) if _db else 0
    }


@app.get("/api/cache/stats")
async def cache_stats():
    return await _cache.stats()


@app.delete("/api/cache")
async def clear_cache():
    """Clear query cache (not sessions)."""
    await _cache.clear()
    return {"status": "cleared"}


# Session Management Endpoints
@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session (clears conversation history)."""
    await _sessions.delete(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.delete("/api/sessions")
async def clear_all_sessions():
    """Clear all sessions."""
    count = await _sessions.clear_all()
    return {"status": "cleared", "count": count}


@app.post("/api/session/new")
async def new_session():
    """Create a new empty session and return its ID."""
    import uuid
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.get("/", response_class=HTMLResponse)
async def root():
    return (BASE_DIR / "templates/index.html").read_text()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
