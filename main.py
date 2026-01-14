"""
ICDA Prototype - Intelligent Customer Data Access
Run with: uvicorn main:app --reload --port 8000

Supports two modes:
- LITE MODE: No AWS credentials - basic search, autocomplete, keyword knowledge search
- FULL MODE: With AWS credentials - AI queries, semantic search, vector embeddings
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging
import sys

from dotenv import load_dotenv
load_dotenv()  # Must be before importing config

# Configure logging - stdout only (no CloudWatch, no file handlers)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("icda")

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import shutil

from icda.config import Config
from icda.cache import RedisCache
from icda.embedding_client import EmbeddingClient
from icda.vector_index import VectorIndex
from icda.database import CustomerDB
from icda.nova import NovaClient
from icda.router import Router
from icda.session import SessionManager
from icda.knowledge import KnowledgeManager
from icda.redis_stack import RedisStackClient
from icda.knowledge_watcher import KnowledgeWatcher
from icda.knowledge_index_state import (
    compute_file_hash, load_index_state, save_index_state,
    update_file_state, remove_file_state, needs_reindex,
    get_orphaned_entries, mark_full_reindex
)
from icda.download_tokens import DownloadTokenManager
from icda.progress_tracker import ProgressTracker, format_bytes, format_duration

# LLM Enforcer imports (provider-agnostic)
from icda.llm import LLMEnforcer, create_llm_client

# Address verification imports
from icda.address_index import AddressIndex
from icda.address_completer import NovaAddressCompleter
from icda.address_pipeline import AddressPipeline
from icda.address_router import router as address_router, configure_router
from icda.upload.router import (
    router as upload_router,
    validation_router,
    configure_upload_services,
)
from icda.agents.orchestrator import AddressAgentOrchestrator
from icda.agents.models import AgentCoreMemoryConfig
from icda.indexes.zip_database import ZipDatabase
from icda.indexes.address_vector_index import AddressVectorIndex
from icda.indexes.redis_vector_index import RedisAddressIndex
from icda.indexes.index_federation import IndexFederation
from icda.embeddings.address_embedder import AddressEmbedder
from icda.llm.nova_reranker import NovaAddressReranker
from icda.address_completion_pipeline import AddressCompletionPipeline, CompletionSource
from icda.completion_router import router as completion_router, configure_completion_router
from icda.redis_stack.router import router as redis_stack_router, configure_router as configure_redis_stack_router
from icda.auth import (
    EmailAuth,
    EmailAuthConfig,
    User,
    init_email_auth,
    get_email_auth,
    get_current_user,
    get_current_user_optional,
)
from icda.ingestion import (
    IngestionPipeline,
    IngestionConfig,
    IngestionBatchResult,
)

cfg = Config()  # Fresh instance after dotenv loaded

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
KNOWLEDGE_DATA_DIR = KNOWLEDGE_DIR / "data"
KNOWLEDGE_UPLOADED_DIR = KNOWLEDGE_DIR / "data-uploaded"
INDEX_STATE_FILE = KNOWLEDGE_DIR / ".index_state.json"

# Supported knowledge file extensions for auto-indexing
KNOWLEDGE_EXTENSIONS = {".md", ".txt", ".json", ".pdf", ".docx", ".doc", ".odt", ".odf", ".csv", ".xls", ".xlsx"}

# Globals
_cache: RedisCache = None
_embedder: EmbeddingClient = None
_vector_index: VectorIndex = None
_db: CustomerDB = None
_nova: NovaClient = None
_sessions: SessionManager = None
_router: Router = None
_knowledge: KnowledgeManager = None
_knowledge_watcher: KnowledgeWatcher = None
_index_state: dict = None

# Address verification globals
_address_index: AddressIndex = None
_address_completer: NovaAddressCompleter = None
_address_pipeline: AddressPipeline = None
_zip_database: ZipDatabase = None
_address_vector_index: AddressVectorIndex = None
_orchestrator: AddressAgentOrchestrator = None

# Address completion pipeline (Redis + Titan + Nova)
_completion_pipeline: AddressCompletionPipeline = None
_address_embedder: AddressEmbedder = None
_redis_vector_index: RedisAddressIndex = None
_nova_reranker: NovaAddressReranker = None

# LLM Enforcer global (supports any secondary LLM provider)
_enforcer: LLMEnforcer = None

# Download Token Manager global
_download_manager: DownloadTokenManager = None

# Progress Tracker global (for real-time indexing progress)
_progress_tracker: ProgressTracker = None

# Redis Stack unified client (provides all Redis Stack modules)
_redis_stack: RedisStackClient = None

# Index Federation (unified federated search across all indexes)
_federation: IndexFederation = None

# Email Authentication (simple email-based auth with session tokens)
_email_auth: EmailAuth = None

# Address Data Ingestion Pipeline (NCOA, webhooks, file drops)
_ingestion_pipeline: IngestionPipeline = None


def _extract_tags_from_content(content: str, filepath: Path) -> list[str]:
    """Extract tags from file content or infer from filename/path."""
    tags = []

    # Extract from YAML frontmatter if present
    if content.startswith("---"):
        lines = content.split("\n")
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                if not in_frontmatter:
                    in_frontmatter = True
                    continue
                else:
                    break
            if in_frontmatter and line.startswith("tags:"):
                tag_part = line.split(":", 1)[1].strip()
                if tag_part.startswith("["):
                    tags.extend([t.strip().strip('"\'') for t in tag_part.strip("[]").split(",")])

    # Infer from filepath
    filename_lower = filepath.stem.lower()
    if "puerto" in filename_lower or "pr-" in filename_lower:
        tags.extend(["puerto-rico", "urbanization"])
    if "address" in filename_lower:
        tags.append("addressing")
    if "example" in filename_lower:
        tags.append("examples")

    return list(set(tags))


async def auto_index_knowledge_documents(knowledge_manager: KnowledgeManager, state: dict = None) -> dict:
    """
    Auto-index all documents from /knowledge directory using content hash tracking.

    Uses hash comparison to avoid re-indexing unchanged files and cleans up orphaned entries.
    """
    if not knowledge_manager or not knowledge_manager.available:
        return {"indexed": 0, "skipped": 0, "failed": 0, "orphans_removed": 0}

    if not KNOWLEDGE_DIR.exists():
        return {"indexed": 0, "skipped": 0, "failed": 0, "orphans_removed": 0}

    # Load or use provided state
    if state is None:
        state = load_index_state(INDEX_STATE_FILE)

    indexed = 0
    skipped = 0
    failed = 0

    # Ensure data directories exist
    KNOWLEDGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_UPLOADED_DIR.mkdir(parents=True, exist_ok=True)

    # Recursively find all knowledge files
    for filepath in KNOWLEDGE_DIR.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in KNOWLEDGE_EXTENSIONS:
            continue
        if filepath.name.startswith(".") or filepath.name.lower() == "readme.md":
            continue

        relative_path = filepath.relative_to(KNOWLEDGE_DIR)
        filename = str(relative_path).replace("\\", "/")

        # Check if file needs reindexing using content hash
        try:
            current_hash = compute_file_hash(filepath)
        except Exception as e:
            logger.error(f"Hash failed: {filename} - {e}")
            failed += 1
            continue

        if not needs_reindex(state, filename, current_hash):
            skipped += 1
            continue

        category = filepath.parent.name if filepath.parent != KNOWLEDGE_DIR else "general"

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            tags = _extract_tags_from_content(content, filepath)

            result = await knowledge_manager.index_document(
                content=filepath,
                filename=filename,
                tags=tags,
                category=category
            )

            if result.get("success"):
                logger.info(f"Indexed: {filename} ({result.get('chunks_indexed', 0)} chunks)")
                update_file_state(
                    state,
                    filename,
                    result.get("doc_id", ""),
                    current_hash,
                    result.get("chunks_indexed", 0)
                )
                indexed += 1
            else:
                logger.warning(f"Index failed: {filename} - {result.get('error')}")
                failed += 1
        except Exception as e:
            logger.error(f"Index error: {filename} - {e}")
            failed += 1

    # Clean up orphaned entries (files that were deleted)
    orphans = get_orphaned_entries(state, KNOWLEDGE_DIR)
    orphans_removed = 0
    for orphan_path, doc_id in orphans:
        if doc_id:
            try:
                await knowledge_manager.delete_document(doc_id)
                logger.info(f"Removed orphan: {orphan_path}")
                orphans_removed += 1
            except Exception as e:
                logger.error(f"Failed to remove orphan: {orphan_path} - {e}")
        remove_file_state(state, orphan_path)

    # Save updated state
    save_index_state(INDEX_STATE_FILE, state)

    return {"indexed": indexed, "skipped": skipped, "failed": failed, "orphans_removed": orphans_removed}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache, _embedder, _vector_index, _db, _nova, _sessions, _router, _knowledge, _knowledge_watcher
    global _address_index, _address_completer, _address_pipeline
    global _zip_database, _address_vector_index, _orchestrator
    global _enforcer, _download_manager, _index_state, _progress_tracker, _redis_stack
    global _federation, _email_auth, _ingestion_pipeline

    logger.info("=" * 50)
    logger.info("ICDA Startup")
    logger.info("=" * 50)

    # Startup - Redis with graceful degradation
    _cache = RedisCache(cfg.cache_ttl)
    await _cache.connect(cfg.redis_url)
    if _cache.available:
        # Clear stale cache entries on startup to ensure fresh responses
        await _cache.clear()
        logger.info("Cache cleared on startup (removing stale entries)")
    else:
        logger.warning("Redis not available at startup - using in-memory fallback")
        logger.warning("Redis will auto-reconnect when available (checked on each health request)")
        logger.info("Start Redis with: docker-compose up -d redis")

    # Initialize progress tracker for real-time indexing feedback
    _progress_tracker = ProgressTracker(_cache)
    logger.info(f"Progress tracker: {'enabled' if _progress_tracker.available else 'disabled (no Redis)'}")

    # Initialize Redis Stack unified client (provides all Redis Stack modules)
    _redis_stack = RedisStackClient()
    redis_modules = await _redis_stack.connect(cfg.redis_url)
    if _redis_stack._connected:
        enabled = [k for k, v in redis_modules.items() if v and k != "connected"]
        logger.info(f"Redis Stack modules: {', '.join(enabled) if enabled else 'base only'}")
        # Configure Redis Stack router (will also get _db later)
        configure_redis_stack_router(_redis_stack)
    else:
        logger.info("Redis Stack: connection failed, using base Redis only")

    _embedder = EmbeddingClient(cfg.aws_region, cfg.titan_embed_model, cfg.embed_dimensions)

    # OpenSearch is REQUIRED
    _vector_index = VectorIndex(_embedder, cfg.opensearch_index)
    await _vector_index.connect(cfg.opensearch_host, cfg.aws_region)
    if not _vector_index.available:
        logger.critical("OpenSearch is REQUIRED but not available! Start OpenSearch with: docker-compose up -d opensearch")
        raise RuntimeError("OpenSearch is required for ICDA")

    _db = CustomerDB(BASE_DIR / "customer_data.json")
    logger.info(f"Customer database: {len(_db.customers)} customers loaded")
    logger.info(f"Available states: {', '.join(_db.get_available_states()[:10])}{'...' if len(_db.available_states) > 10 else ''}")

    # Update Redis Stack router with database for suggestion building
    if _redis_stack and _redis_stack._connected:
        configure_redis_stack_router(_redis_stack, _db)

    # ============================================================
    # CUSTOMER INDEX: Check index status only (no auto-reindex)
    # Reindexing is only triggered via API endpoint, not on startup.
    # This prevents long startup times and unnecessary reindexing.
    # ============================================================
    if _vector_index.available and _embedder.available:
        try:
            indexed_count = await asyncio.wait_for(
                _vector_index.customer_count(),
                timeout=10.0  # Quick timeout - don't block startup
            )
            db_count = len(_db.customers)
            if indexed_count == 0:
                logger.warning(f"Customer index: EMPTY (POST /api/admin/index/reindex?index_name=customers to populate)")
            elif indexed_count != db_count:
                logger.info(f"Customer index: {indexed_count:,} customers (DB: {db_count:,})")
            else:
                logger.info(f"Customer index: {indexed_count:,} customers (in sync)")
        except asyncio.TimeoutError:
            logger.warning("Customer index: count check timed out (OpenSearch may be slow)")
        except Exception as e:
            logger.warning(f"Customer index: status check failed - {e}")
    else:
        logger.info("Customer indexing skipped (OpenSearch or embeddings not available)")

    _sessions = SessionManager(_cache)

    # Initialize Email Authentication (simple email-based auth with Redis sessions)
    _email_auth = init_email_auth(EmailAuthConfig(), _cache)
    if _email_auth.available:
        logger.info(f"Email Auth: enabled (session TTL={_email_auth.config.session_ttl}s)")
    else:
        if not _email_auth.config.enabled:
            logger.info("Email Auth: disabled (set AUTH_ENABLED=true to enable)")
        else:
            logger.info("Email Auth: disabled (requires Redis)")

    # Initialize download token manager
    _download_manager = DownloadTokenManager(_cache)
    logger.info(f"Download token manager: threshold={_download_manager.pagination_threshold}, preview={_download_manager.preview_size}")

    # Initialize address verification
    logger.info("Initializing address verification...")
    _address_index = AddressIndex()
    _address_index.build_from_customers(_db.customers)
    logger.info(f"Address index: {_address_index.total_addresses} addresses")

    _zip_database = ZipDatabase()
    _zip_database.build_from_customers(_db.customers)
    logger.info(f"ZIP database: {_zip_database.total_zips} ZIPs")

    _address_completer = NovaAddressCompleter(cfg.aws_region, cfg.nova_model, _address_index)
    _address_pipeline = AddressPipeline(_address_index, _address_completer)

    _address_vector_index = None
    if cfg.opensearch_host and _embedder.available:
        _address_vector_index = AddressVectorIndex(_embedder)
        connected = await _address_vector_index.connect(cfg.opensearch_host, cfg.aws_region)
        if connected:
            logger.info("Address vector index: connected")
        else:
            _address_vector_index = None

    _orchestrator = AddressAgentOrchestrator(
        address_index=_address_index,
        zip_database=_zip_database,
        vector_index=_address_vector_index,
    )
    configure_router(_address_pipeline, _orchestrator)
    logger.info("Address verification: ready")

    # Initialize address completion pipeline (Redis + Titan + Nova)
    logger.info("Initializing address completion pipeline...")
    _address_embedder = AddressEmbedder(
        redis_client=_cache.client if _cache.available else None,
        region=cfg.aws_region,
        model=cfg.titan_embed_model,
        dimensions=cfg.embed_dimensions
    )

    _redis_vector_index = None
    if _cache.available and _address_embedder.available:
        _redis_vector_index = RedisAddressIndex(_cache.client, _address_embedder)
        try:
            redis_vector_available = await asyncio.wait_for(
                _redis_vector_index.initialize(),
                timeout=10.0  # 10 second timeout for Redis Stack initialization
            )
        except asyncio.TimeoutError:
            logger.warning("Redis vector index: initialization timed out (10s)")
            redis_vector_available = False

        if redis_vector_available:
            logger.info("  Redis vector index: available (Redis Stack)")
        else:
            logger.info("  Redis vector index: not available (install Redis Stack for vector search)")
            _redis_vector_index = None

    _nova_reranker = NovaAddressReranker(
        region=cfg.aws_region,
        model=cfg.nova_lite_model
    )

    _completion_pipeline = AddressCompletionPipeline(
        redis_client=_cache.client if _cache.available else None,
        embedder=_address_embedder,
        vector_index=_redis_vector_index,
        reranker=_nova_reranker,
        cache_ttl=cfg.cache_ttl,
        vector_confidence_threshold=cfg.completion_vector_threshold,
        min_confidence=cfg.completion_min_confidence
    )

    configure_completion_router(
        pipeline=_completion_pipeline,
        vector_index=_redis_vector_index,
        embedder=_address_embedder,
        db=_db
    )
    logger.info(f"  Address completion pipeline: {'ready' if _completion_pipeline.available else 'limited (no embeddings)'}")

    # Initialize bulk upload/validation services
    logger.info("Initializing bulk upload/validation services...")
    configure_upload_services(
        embedding_client=_embedder,
        opensearch_client=_vector_index.client if _vector_index and _vector_index.available else None,
        nova_client=None  # Will initialize lazily
    )
    logger.info("Bulk upload/validation: ready")

    # Initialize knowledge base BEFORE NovaClient so it can use RAG
    logger.info("Initializing knowledge base...")
    opensearch_client = _vector_index.client if _vector_index.available else None
    _knowledge = KnowledgeManager(_embedder, opensearch_client)
    await _knowledge.ensure_index()

    # ============================================================
    # KNOWLEDGE INDEX: Incremental indexing (NOT full scan on every startup)
    # - First run (no state files): Do initial full index
    # - Subsequent runs: Trust saved state, rely on file watcher
    # - Manual reindex: POST /api/knowledge/reindex?force=true
    #
    # ROOT CAUSE FIX: Only check if files exist in state, NOT last_full_index
    # The last_full_index field was often null in legacy state files, causing
    # every startup to think it was "first run" and do a full scan.
    # ============================================================
    _index_state = load_index_state(INDEX_STATE_FILE)
    state_has_files = len(_index_state.get("files", {})) > 0

    if KNOWLEDGE_DIR.exists():
        if state_has_files:
            # Existing index state - skip startup scan, rely on file watcher
            state_stats = {
                "tracked_files": len(_index_state.get("files", {})),
                "total_chunks": sum(f.get("chunks_indexed", 0) for f in _index_state.get("files", {}).values())
            }
            logger.info(f"Knowledge index: {state_stats['tracked_files']} files, {state_stats['total_chunks']} chunks (using saved state)")
            logger.info("  Incremental mode: file watcher active, NO startup scan")
            # Ensure last_full_index is set (fixes legacy state files with null)
            if _index_state.get("last_full_index") is None:
                mark_full_reindex(_index_state)
                save_index_state(INDEX_STATE_FILE, _index_state)
                logger.info("  Fixed legacy state: set last_full_index marker")
        else:
            # First run - no files in state, do initial full index
            logger.info("Knowledge index: FIRST RUN - indexing all documents...")
            result = await auto_index_knowledge_documents(_knowledge, _index_state)
            if result["indexed"]:
                logger.info(f"  Indexed: {result['indexed']} files")
            if result.get("orphans_removed"):
                logger.info(f"  Orphans removed: {result['orphans_removed']}")
            if result["failed"]:
                logger.warning(f"  Failed: {result['failed']}")
            # Mark that we did the initial full index
            mark_full_reindex(_index_state)
            save_index_state(INDEX_STATE_FILE, _index_state)

    stats = await _knowledge.get_stats()
    logger.info(f"Knowledge base: {stats.get('unique_documents', 0)} docs, {stats.get('total_chunks', 0)} chunks ({stats.get('backend', 'unknown')})")

    # Start knowledge file watcher for auto-indexing new files
    async def index_file_callback(filepath: Path) -> dict:
        """Callback for file watcher to index new/modified files."""
        global _index_state
        try:
            relative_path = str(filepath.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
            content_hash = compute_file_hash(filepath)

            # Check if actually changed
            if _index_state and not needs_reindex(_index_state, relative_path, content_hash):
                return {"success": True, "skipped": True, "reason": "unchanged"}

            content = filepath.read_text(encoding="utf-8", errors="ignore")
            tags = _extract_tags_from_content(content, filepath)
            category = filepath.parent.name if filepath.parent != KNOWLEDGE_DIR else "general"

            result = await _knowledge.index_document(
                content=filepath,
                filename=relative_path,
                tags=tags,
                category=category
            )

            # Update index state on success
            if result.get("success") and _index_state is not None:
                update_file_state(
                    _index_state,
                    relative_path,
                    result.get("doc_id", ""),
                    content_hash,
                    result.get("chunks_indexed", 0)
                )
                save_index_state(INDEX_STATE_FILE, _index_state)

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    _knowledge_watcher = KnowledgeWatcher(KNOWLEDGE_DIR, index_file_callback)
    _knowledge_watcher.start()

    # Initialize Index Federation (unified federated search across all indexes)
    if cfg.enable_federation and _vector_index.available:
        logger.info("Initializing Index Federation...")
        _federation = IndexFederation(
            opensearch_client=_vector_index.client,
            embedder=_embedder,
            config=cfg.get_index_config()  # Uses configured index names
        )
        # Ensure all federation indexes exist
        try:
            index_results = await asyncio.wait_for(
                _federation.ensure_all_indexes(),
                timeout=15.0
            )
            enabled_indexes = [k for k, v in index_results.items() if v]
            logger.info(f"Index Federation: {len(enabled_indexes)}/4 indexes ready ({', '.join(enabled_indexes)})")
        except asyncio.TimeoutError:
            logger.warning("Index Federation: index creation timed out, federation may be limited")
        except Exception as e:
            logger.warning(f"Index Federation: failed to ensure indexes - {e}")
    else:
        logger.info(f"Index Federation: disabled (enable_federation={cfg.enable_federation}, opensearch={_vector_index.available})")

    # Initialize Address Data Ingestion Pipeline (optional)
    # Supports: NCOA batch files, REST webhooks, file drops
    # Features: AI schema mapping, multi-provider embeddings, 5-stage quality enforcers
    try:
        ingestion_config = IngestionConfig()
        _ingestion_pipeline = IngestionPipeline(
            config=ingestion_config,
            address_index=_address_index,
            vector_index=_address_vector_index,
            embedding_client=_embedder,
            nova_client=None,  # Will be set after NovaClient is created
        )
        await _ingestion_pipeline.initialize()
        logger.info("Ingestion Pipeline: initialized")
        logger.info(f"  - Schema mapping: {'AI-enabled' if ingestion_config.enable_ai_schema_mapping else 'disabled'}")
        logger.info(f"  - Quality enforcers: {'enabled' if ingestion_config.enforcers.enabled else 'disabled'}")
    except Exception as e:
        logger.warning(f"Ingestion Pipeline: initialization failed - {e}")
        _ingestion_pipeline = None

    # Initialize LLM Enforcer FIRST (optional, graceful degradation if no API key)
    # This must be done before NovaClient so it can be passed to the orchestrator
    # Supports any secondary LLM: Gemini, OpenAI, Claude, OpenRouter
    print("\nInitializing LLM Enforcer...")
    llm_client = create_llm_client(
        provider=cfg.secondary_llm_provider,
        model=cfg.secondary_llm_model if cfg.secondary_llm_model else None,
    )
    _enforcer = LLMEnforcer(
        client=llm_client,
        chunk_threshold=cfg.enforcer_chunk_threshold,
        query_sample_rate=cfg.enforcer_query_sample_rate,
        validation_interval_hours=cfg.enforcer_validation_interval,
    )
    if _enforcer.available:
        print(f"  Enforcer: enabled (provider: {llm_client.provider}, model: {llm_client.config.model})")
        print(f"  - L1 Chunk Gate: threshold {cfg.enforcer_chunk_threshold}")
        print(f"  - L2 Index Validation: every {cfg.enforcer_validation_interval}h")
        print(f"  - L3 Query Review: {int(cfg.enforcer_query_sample_rate * 100)}% sample")
    else:
        print("  Enforcer: disabled (no LLM API key found)")

    # Initialize NovaClient with 11-agent pipeline + LLM enforcer
    print("\nInitializing AI query pipeline...")

    # Build model routing config from settings
    model_config = {
        "nova_lite_model": cfg.nova_lite_model,
        "nova_pro_model": cfg.nova_pro_model,
        "model_routing_threshold": cfg.model_routing_threshold,
    }
    print(f"  Model routing: micro={cfg.nova_model}, lite={cfg.nova_lite_model}, pro={cfg.nova_pro_model}")

    # Create AgentCore memory configuration
    agentcore_config = AgentCoreMemoryConfig(
        memory_id=cfg.agentcore_memory_id if cfg.agentcore_memory_id else None,
        region=cfg.agentcore_region,
        use_ltm=cfg.agentcore_use_ltm,
        stm_retention_days=cfg.agentcore_stm_retention_days,
        ltm_retention_days=cfg.agentcore_ltm_retention_days,
        enabled=cfg.enable_agentcore_memory,
    )

    _nova = NovaClient(
        region=cfg.aws_region,
        model=cfg.nova_model,
        db=_db,
        vector_index=_vector_index,
        knowledge=_knowledge,  # Pass knowledge manager for RAG
        address_orchestrator=_orchestrator,
        session_store=_sessions,
        llm_enforcer=_enforcer,  # Pass LLM enforcer for quality validation
        use_orchestrator=True,  # Enable 11-agent pipeline
        download_manager=_download_manager,  # Pass download manager for pagination
        model_config=model_config,  # Pass model routing config
        cache=_cache,  # Pass cache for memory storage
        agentcore_config=agentcore_config,  # Pass AgentCore memory config
    )
    if _nova.available:
        if _nova.orchestrator:
            enforcer_status = f" + {llm_client.provider} enforcer" if _enforcer.available else ""
            memory_status = "enabled" if _nova.orchestrator._memory_agent.available else "disabled"
            agentcore_status = "enabled" if agentcore_config.enabled else "disabled"
            print(f"  Nova AI: enabled with 11-agent orchestrator{enforcer_status}")
            print(f"    - KnowledgeAgent: {'enabled' if _nova.orchestrator._knowledge_agent.available else 'disabled'}")
            print(f"    - MemoryAgent: {memory_status}")
            print(f"    - UnifiedMemoryLayer: {agentcore_status} (AgentCore)")
            print(f"    - EnforcerCoordinator: enabled (5 memory enforcers)")
            print(f"    - PersonalityAgent: Witty Expert")
        else:
            print(f"  Nova AI: enabled (simple mode)")
    else:
        print("  Nova AI: disabled (no AWS credentials)")

    _router = Router(_cache, _vector_index, _db, _nova, _sessions)

    # Print mode summary
    mode = "FULL" if _nova.available and _embedder.available else "LITE"
    print("\n" + "="*50)
    print(f"  ICDA Running in {mode} MODE")
    print("="*50)
    if mode == "LITE":
        print("  (Add AWS credentials to .env for AI features)")
    print(f"\n  API: http://localhost:8000")
    print(f"  Docs: http://localhost:8000/docs\n")

    yield

    # Shutdown
    if _knowledge_watcher:
        _knowledge_watcher.stop()
    if _redis_stack:
        await _redis_stack.close()
    await _cache.close()
    await _vector_index.close()
    if _address_vector_index:
        await _address_vector_index.close()


app = FastAPI(title="ICDA", version="0.8.0", lifespan=lifespan)
app.include_router(address_router)
app.include_router(upload_router)
app.include_router(validation_router)
app.include_router(completion_router)
app.include_router(redis_stack_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Sanitized global exception handler - never exposes internal details."""
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later."
            }
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )


class GuardrailSettings(BaseModel):
    pii: bool = True
    financial: bool = True
    credentials: bool = True
    offtopic: bool = True


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    bypass_cache: bool = False
    guardrails: GuardrailSettings | None = None
    session_id: str | None = None  # For conversation continuity


@app.post("/api/query")
async def query(req: QueryRequest):
    guards = req.guardrails.model_dump() if req.guardrails else None
    return await _router.route(req.query, req.bypass_cache, guards, req.session_id)


@app.get("/api/query/download/{token}")
async def download_results(token: str, format: str = "json"):
    """Download full results using download token.

    Args:
        token: Download token from paginated query response.
        format: Output format ('json' or 'csv').

    Returns:
        Full result set in requested format.
    """
    if not _download_manager:
        return {"success": False, "error": "Download manager not available"}

    result = await _download_manager.get_full_results_async(token)
    if not result:
        return {"success": False, "error": "Invalid or expired download token"}

    if format == "csv":
        import csv
        import io
        from fastapi.responses import StreamingResponse

        data = result["data"]
        if not data:
            return {"success": False, "error": "No data to download"}

        output = io.StringIO()
        if data:
            # Get all unique keys from all records
            all_keys = set()
            for record in data:
                all_keys.update(record.keys())
            fieldnames = sorted(all_keys)

            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=icda_results_{token[:8]}.csv"}
        )

    # Default: JSON format
    return result


@app.get("/api/query/paginate/{token}")
async def paginate_results(token: str, offset: int = 0, limit: int = 15):
    """Get a page of results using download token for inline pagination.

    Args:
        token: Download token from paginated query response.
        offset: Starting index for results (0-based).
        limit: Maximum number of results to return.

    Returns:
        Paginated result set with metadata.
    """
    if not _download_manager:
        return {"success": False, "error": "Download manager not available"}

    result = await _download_manager.get_full_results_async(token)
    if not result:
        return {"success": False, "error": "Invalid or expired download token"}

    data = result.get("data", [])
    total = len(data)
    page_data = data[offset:offset + limit]

    return {
        "success": True,
        "data": page_data,
        "offset": offset,
        "limit": limit,
        "total": total,
        "has_more": offset + limit < total,
        "remaining": max(0, total - offset - limit),
    }


# =============================================================================
# AUTH ENDPOINTS - Simple email-based authentication
# =============================================================================

class LoginRequest(BaseModel):
    """Email login request."""
    email: str = Field(..., description="User's email address")


@app.post("/api/auth/login")
async def auth_login(request: LoginRequest):
    """
    Login or register user by email.

    Returns a session token that should be included in subsequent requests
    via X-Session-ID header or Authorization: Bearer <token>.
    """
    auth = get_email_auth()

    if not auth.config.enabled:
        return {
            "success": False,
            "error": "Authentication is disabled. Set AUTH_ENABLED=true to enable.",
        }

    if not auth.available:
        return {
            "success": False,
            "error": "Authentication service unavailable (requires Redis)",
        }

    user, error = await auth.login(request.email)

    if not user:
        return {
            "success": False,
            "error": error,
        }

    return {
        "success": True,
        "user": {
            "email": user.email,
            "username": user.username,
            "id": user.id,
        },
        "session_id": user.session_id,
        "expires_in": auth.config.session_ttl,
    }


@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    """
    Logout user by invalidating session.

    Session ID can be provided via:
    - X-Session-ID header
    - Authorization: Bearer <session_id> header
    """
    auth = get_email_auth()

    if not auth.available:
        return {"success": False, "error": "Authentication service unavailable"}

    # Get session ID from headers
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            session_id = auth_header[7:]

    if not session_id:
        return {"success": False, "error": "No session ID provided"}

    success = await auth.logout(session_id)
    return {"success": success}


@app.get("/api/auth/me")
async def auth_me(request: Request):
    """
    Get current authenticated user info.

    Returns user details if authenticated, or anonymous status if not.
    """
    auth = get_email_auth()

    if not auth.config.enabled:
        return {
            "authenticated": False,
            "auth_enabled": False,
            "user": None,
        }

    if not auth.available:
        return {
            "authenticated": False,
            "auth_enabled": True,
            "auth_available": False,
            "user": None,
        }

    # Try to get session from headers
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            session_id = auth_header[7:]

    if not session_id:
        return {
            "authenticated": False,
            "auth_enabled": True,
            "auth_available": True,
            "user": None,
        }

    user = await auth.validate_session(session_id)

    if not user:
        return {
            "authenticated": False,
            "auth_enabled": True,
            "auth_available": True,
            "user": None,
            "error": "Invalid or expired session",
        }

    return {
        "authenticated": True,
        "auth_enabled": True,
        "auth_available": True,
        "user": {
            "email": user.email,
            "username": user.username,
            "id": user.id,
            "created_at": user.created_at.isoformat(),
            "last_active": user.last_active.isoformat(),
        },
    }


@app.get("/api/auth/status")
async def auth_status():
    """Get authentication system status."""
    auth = get_email_auth()

    return {
        "enabled": auth.config.enabled,
        "available": auth.available,
        "session_ttl": auth.config.session_ttl,
        "allow_anonymous": auth.config.allow_anonymous,
        "require_verification": auth.config.require_verification,
        "allowed_domains": list(auth.config.get_allowed_domains()) if auth.config.allowed_domains else None,
    }


# =============================================================================
# SESSION ENDPOINTS - Conversation session management
# =============================================================================

@app.post("/api/session/new")
async def session_new(request: Request):
    """
    Create a new conversation session.

    If user is authenticated (via X-Session-ID or Authorization header),
    the session will be linked to their user account.
    """
    import uuid

    # Check for authenticated user
    auth = get_email_auth()
    user = None
    if auth.available:
        session_id = request.headers.get("X-Session-ID")
        if not session_id:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                session_id = auth_header[7:]
        if session_id:
            user = await auth.validate_session(session_id)

    # Create new session
    new_session_id = str(uuid.uuid4())

    if user:
        session = await _sessions.get_or_create_for_user(
            new_session_id,
            user_id=user.id,
            user_email=user.email,
        )
    else:
        session = await _sessions.get(new_session_id)

    await _sessions.save(session)

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "created_at": session.created_at,
    }


@app.delete("/api/session/{session_id}")
async def session_delete(session_id: str):
    """Delete a specific conversation session."""
    await _sessions.delete(session_id)
    return {"success": True, "session_id": session_id}


@app.delete("/api/sessions")
async def sessions_clear(request: Request):
    """
    Clear all sessions.

    If user is authenticated, only their sessions are cleared.
    Otherwise, clears all sessions (admin operation).
    """
    auth = get_email_auth()
    user = None

    if auth.available:
        session_id = request.headers.get("X-Session-ID")
        if not session_id:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                session_id = auth_header[7:]
        if session_id:
            user = await auth.validate_session(session_id)

    if user:
        # Clear only this user's sessions
        count = await _sessions.delete_user_sessions(user.id)
        return {"status": "cleared", "count": count, "scope": "user"}
    else:
        # Clear all sessions (anonymous/admin)
        count = await _sessions.clear_all()
        return {"status": "cleared", "count": count, "scope": "all"}


@app.get("/api/sessions")
async def sessions_list(request: Request):
    """
    List all sessions for the authenticated user.

    Returns empty list if not authenticated.
    """
    auth = get_email_auth()

    if not auth.available:
        return {"sessions": [], "error": "Authentication not available"}

    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            session_id = auth_header[7:]

    if not session_id:
        return {"sessions": [], "error": "Not authenticated"}

    user = await auth.validate_session(session_id)
    if not user:
        return {"sessions": [], "error": "Invalid session"}

    sessions = await _sessions.get_user_sessions(user.id)

    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "message_count": len(s.messages),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "preview": s.messages[0].content[:100] if s.messages else None,
            }
            for s in sessions
        ],
        "count": len(sessions),
    }


@app.get("/api/health")
async def health():
    """Health check with mode status - flat structure for frontend compatibility.

    Automatically attempts to reconnect to Redis if connection was lost.
    """
    nova_ok = _nova.available if _nova else False
    embedder_ok = _embedder.available if _embedder else False

    # Try to reconnect to Redis if not connected
    redis_ok = False
    if _cache:
        if not _cache.available:
            await _cache.reconnect()
        redis_ok = _cache.available

    opensearch_ok = _vector_index.available if _vector_index else False
    mode = "FULL" if nova_ok and embedder_ok else "LITE"

    # Flat structure expected by frontend
    return {
        "status": "healthy",
        "mode": mode,
        "redis": redis_ok,
        "opensearch": opensearch_ok,
        "embedder": embedder_ok,
        "nova": nova_ok,
        "customers": len(_db.customers) if _db else 0,
        # Extended info
        "knowledge": _knowledge.available if _knowledge else False,
    }


@app.get("/api/cache/stats")
async def cache_stats():
    return await _cache.stats()


@app.delete("/api/cache")
async def clear_cache():
    await _cache.clear()
    return {"status": "cleared"}


# =============================================================================
# FRONTEND CONVENIENCE ENDPOINTS - Aliases for frontend API compatibility
# =============================================================================

class AddressVerificationRequest(BaseModel):
    """Request for batch address verification."""
    addresses: list[dict] = Field(..., description="List of address objects with street, city, state, zip")
    include_corrections: bool = Field(default=True, description="Include Nova corrections for failed matches")


@app.post("/api/verify-addresses")
async def verify_addresses(request: AddressVerificationRequest):
    """
    Verify a batch of addresses.

    Alias for /api/validate/batch for frontend compatibility.
    """
    # Forward to the validation router's batch endpoint
    if not _address_pipeline:
        return {"success": False, "error": "Address verification not initialized"}

    results = []
    for addr in request.addresses:
        try:
            result = await _address_pipeline.verify(
                street=addr.get("street", ""),
                city=addr.get("city"),
                state=addr.get("state"),
                zip_code=addr.get("zip") or addr.get("zip_code"),
            )
            results.append({
                "input": addr,
                "match": result.get("match"),
                "confidence": result.get("confidence", 0),
                "status": "verified" if result.get("match") else "not_found",
            })
        except Exception as e:
            results.append({
                "input": addr,
                "status": "error",
                "error": str(e),
            })

    verified_count = sum(1 for r in results if r.get("status") == "verified")
    return {
        "success": True,
        "total": len(results),
        "verified": verified_count,
        "failed": len(results) - verified_count,
        "results": results,
    }


@app.post("/api/upload-addresses")
async def upload_addresses(
    file: UploadFile = File(...),
    save_to_pipeline: bool = Form(default=False),
):
    """
    Upload a file containing addresses for verification.

    Supports CSV, Excel, and text files with one address per line.
    """
    import csv
    from io import StringIO

    # Read file content
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception:
            return {"success": False, "error": "Unable to decode file (unsupported encoding)"}

    # Parse addresses based on file type
    addresses = []
    filename = file.filename.lower() if file.filename else ""

    if filename.endswith(".csv"):
        # Parse CSV
        reader = csv.DictReader(StringIO(text))
        for row in reader:
            # Try to find address fields
            addr = {}
            for key in ["street", "address", "street_address", "address1"]:
                if key in row:
                    addr["street"] = row[key]
                    break
            for key in ["city"]:
                if key in row:
                    addr["city"] = row[key]
            for key in ["state", "st"]:
                if key in row:
                    addr["state"] = row[key]
            for key in ["zip", "zip_code", "zipcode", "postal"]:
                if key in row:
                    addr["zip"] = row[key]

            if addr.get("street"):
                addresses.append(addr)
    else:
        # Plain text - one address per line
        for line in text.strip().split("\n"):
            line = line.strip()
            if line:
                addresses.append({"street": line})

    if not addresses:
        return {"success": False, "error": "No valid addresses found in file"}

    # Verify addresses
    results = []
    for addr in addresses:
        try:
            result = await _address_pipeline.verify(
                street=addr.get("street", ""),
                city=addr.get("city"),
                state=addr.get("state"),
                zip_code=addr.get("zip"),
            )
            results.append({
                "input": addr,
                "match": result.get("match"),
                "confidence": result.get("confidence", 0),
                "status": "verified" if result.get("match") else "not_found",
            })
        except Exception as e:
            results.append({
                "input": addr,
                "status": "error",
                "error": str(e),
            })

    verified_count = sum(1 for r in results if r.get("status") == "verified")
    return {
        "success": True,
        "filename": file.filename,
        "total": len(results),
        "verified": verified_count,
        "failed": len(results) - verified_count,
        "results": results,
    }


@app.post("/api/query-with-file")
async def query_with_file(
    query: str = Form(...),
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    bypass_cache: bool = Form(False),
    pii: bool = Form(True),
    financial: bool = Form(True),
    credentials: bool = Form(True),
    offtopic: bool = Form(True),
):
    """
    Query with an attached file for context.

    The file content is extracted and used as additional context for the query.
    """
    # Read file content
    content = await file.read()
    try:
        file_text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            file_text = content.decode("latin-1")
        except Exception:
            return {"success": False, "error": "Unable to decode file (unsupported encoding)"}

    # Truncate if too long
    max_context_len = 8000  # Keep context reasonable
    if len(file_text) > max_context_len:
        file_text = file_text[:max_context_len] + f"\n\n[... truncated {len(file_text) - max_context_len} characters ...]"

    # Construct enhanced query with file context
    enhanced_query = f"""User query: {query}

File context ({file.filename}):
---
{file_text}
---

Please answer the query considering the file content above."""

    # Route through main query endpoint
    from pydantic import BaseModel as PydanticBaseModel

    class InternalQueryRequest(PydanticBaseModel):
        query: str
        session_id: Optional[str] = None
        bypass_cache: bool = False
        guardrails: Optional[dict] = None

    internal_request = InternalQueryRequest(
        query=enhanced_query,
        session_id=session_id,
        bypass_cache=bypass_cache,
        guardrails={
            "pii": pii,
            "financial": financial,
            "credentials": credentials,
            "offtopic": offtopic,
        },
    )

    # Get session
    session = await _sessions.get(session_id)

    # Process query through router
    result = await _router.route(
        internal_request.query,
        session.get_history() if session else [],
        {
            "pii": pii,
            "financial": financial,
            "credentials": credentials,
            "offtopic": offtopic,
        },
        bypass_cache=bypass_cache,
    )

    # Save to session
    if session:
        session.add_message("user", query)  # Original query, not enhanced
        if result.get("response"):
            session.add_message("assistant", result["response"])
        await _sessions.save(session)

    return {
        **result,
        "session_id": session.session_id if session else None,
        "file_processed": file.filename,
    }


@app.get("/api/autocomplete/{field}")
async def autocomplete(field: str, q: str, limit: int = 10, fuzzy: bool = False):
    """Autocomplete for address, name, or city fields."""
    if fuzzy:
        return _db.autocomplete_fuzzy(field, q, limit)
    return _db.autocomplete(field, q, limit)


@app.get("/api/search/semantic")
async def semantic_search(q: str, limit: int = 10, state: str = None, city: str = None,
                          min_moves: int = None, status: str = None, customer_type: str = None):
    """Semantic search (requires OpenSearch + embeddings)"""
    filters = {}
    if state:
        filters["state"] = state
    if city:
        filters["city"] = city
    if min_moves:
        filters["min_moves"] = min_moves
    if status:
        filters["status"] = status
    if customer_type:
        filters["customer_type"] = customer_type

    return await _vector_index.search_customers_semantic(q, limit, filters if filters else None)


@app.get("/api/search/hybrid")
async def hybrid_search(q: str, limit: int = 10, state: str = None, min_moves: int = None):
    """Hybrid text + semantic search"""
    filters = {}
    if state:
        filters["state"] = state
    if min_moves:
        filters["min_moves"] = min_moves

    return await _vector_index.search_customers_hybrid(q, limit, filters if filters else None)


@app.get("/api/index/status")
async def index_status():
    """Get comprehensive index status including sync state and active operations.

    Returns:
        - opensearch_available: Whether OpenSearch is connected
        - indexed_count: Number of customers in the index
        - db_count: Number of customers in the database
        - sync_status: 'synced', 'out_of_sync', or 'empty'
        - sync_diff: Difference between index and DB counts
        - indexing_active: Whether a reindex operation is in progress
        - indexing_progress: Progress details if indexing is active
    """
    indexed_count = 0
    db_count = len(_db.customers) if _db else 0

    if _vector_index and _vector_index.available:
        try:
            indexed_count = await _vector_index.customer_count()
        except Exception:
            indexed_count = 0

    # Determine sync status
    if indexed_count == 0:
        sync_status = "empty"
    elif indexed_count == db_count:
        sync_status = "synced"
    else:
        sync_status = "out_of_sync"

    # Check for active indexing operations
    indexing_active = False
    indexing_progress = None

    if _progress_tracker:
        operations = await _progress_tracker.get_active_operations()
        for op in operations:
            if op.operation_type == "customer_index":
                indexing_active = True
                indexing_progress = {
                    "operation_id": op.operation_id,
                    "processed": op.processed_items,
                    "total": op.total_items,
                    "percent": round((op.processed_items / op.total_items * 100) if op.total_items > 0 else 0, 1),
                    "phase": op.current_phase,
                    "elapsed_seconds": op.elapsed_seconds,
                    "items_per_second": round(op.items_per_second, 1) if op.items_per_second else 0,
                    "estimated_remaining_seconds": op.estimated_remaining_seconds,
                }
                break

    return {
        "opensearch_available": _vector_index.available if _vector_index else False,
        "customer_index": _vector_index.customer_index if _vector_index else None,
        "indexed_count": indexed_count,
        "db_count": db_count,
        "sync_status": sync_status,
        "sync_diff": indexed_count - db_count,
        "indexing_active": indexing_active,
        "indexing_progress": indexing_progress,
    }


@app.get("/api/data/states")
async def get_available_states():
    """Get all states available in the current data source.
    
    This is dynamically derived from the actual data, not hardcoded.
    Use this to know what states you can query for.
    """
    if not _db:
        return {"success": False, "error": "Database not loaded"}
    
    states = _db.get_available_states()
    counts = _db.get_state_counts()
    
    return {
        "success": True,
        "total_states": len(states),
        "total_customers": len(_db.customers),
        "states": [
            {
                "code": code,
                "name": _db.STATE_CODE_TO_NAME.get(code, code),
                "customer_count": counts.get(code, 0)
            }
            for code in states
        ]
    }


@app.post("/api/data/reindex")
async def reindex_customer_data(force: bool = False, async_mode: bool = False):
    """Trigger reindexing of customer data into OpenSearch.

    Use this when:
    - You've switched to a new data source
    - You've updated the customer data file
    - The index seems out of sync

    Args:
        force: If True, delete and recreate the index. If False, only reindex if counts differ.
        async_mode: If True, run in background and return operation_id for progress tracking.
    """
    if not _vector_index or not _vector_index.available:
        return {"success": False, "error": "OpenSearch not available"}
    if not _embedder or not _embedder.available:
        return {"success": False, "error": "Embeddings not available"}
    if not _db:
        return {"success": False, "error": "Database not loaded"}

    indexed_count = await _vector_index.customer_count()
    db_count = len(_db.customers)

    if not force and indexed_count == db_count:
        return {
            "success": True,
            "message": "Index already in sync",
            "indexed": indexed_count,
            "db_count": db_count
        }

    # Async mode: run in background with progress tracking
    if async_mode and _progress_tracker:
        op_id = await _progress_tracker.start_operation(
            operation_type="customer_index",
            total_items=db_count,
            total_batches=db_count // 100 + 1,
        )

        async def run_reindex():
            try:
                if force:
                    await _progress_tracker.update_progress(op_id, phase="Deleting old index")
                    await _vector_index.delete_customer_index()
                    await _cache.clear()

                await _progress_tracker.update_progress(op_id, phase="Indexing customers")

                # Progress callback for batch updates
                async def on_progress(processed: int, total: int):
                    batch = processed // 100
                    bytes_est = processed * 200  # ~200 bytes per customer
                    embeddings_est = processed  # 1 embedding per customer
                    await _progress_tracker.update_progress(
                        op_id,
                        processed=processed,
                        batch=batch,
                        bytes_processed=bytes_est,
                        embeddings=embeddings_est,
                        message=f"Indexed {processed:,} of {total:,} customers",
                    )

                # Create sync wrapper for async callback
                def sync_progress(processed: int, total: int):
                    asyncio.create_task(on_progress(processed, total))

                result = await _vector_index.index_customers(
                    _db.customers,
                    batch_size=100,
                    progress_callback=sync_progress,
                )

                await _progress_tracker.complete_operation(
                    op_id,
                    success=True,
                    message=f"Indexed {result.get('indexed', 0):,} customers ({result.get('errors', 0)} errors)",
                )
            except Exception as e:
                await _progress_tracker.complete_operation(
                    op_id,
                    success=False,
                    error=str(e),
                )

        # Start background task
        asyncio.create_task(run_reindex())

        return {
            "success": True,
            "async": True,
            "operation_id": op_id,
            "total_items": db_count,
            "stream_url": f"/api/data/reindex/stream/{op_id}",
            "status_url": f"/api/data/reindex/status/{op_id}",
        }

    # Synchronous mode (original behavior)
    if force:
        await _vector_index.delete_customer_index()
        await _cache.clear()

    result = await _vector_index.index_customers(_db.customers, batch_size=100)

    return {
        "success": True,
        "indexed": result.get("indexed", 0),
        "errors": result.get("errors", 0),
        "previous_count": indexed_count,
        "db_count": db_count,
        "available_states": _db.get_available_states()
    }


@app.post("/api/data/sync")
async def sync_customer_data(async_mode: bool = False):
    """Incremental sync - only index NEW/CHANGED customers (delta-based).

    This is much faster than full reindex when you've added a few records.
    It compares CRIDs in the database vs OpenSearch and only processes the delta.

    Use this when:
    - You've added new customers to customer_data.json
    - You want fast sync without re-embedding all 50k+ records

    Use /api/data/reindex?force=true for full reindex when:
    - Index is corrupted
    - You want to regenerate all embeddings

    Args:
        async_mode: If True, run in background and return operation_id.

    Returns:
        Sync results with counts of added/deleted records.
    """
    if not _vector_index or not _vector_index.available:
        return {"success": False, "error": "OpenSearch not available"}
    if not _embedder or not _embedder.available:
        return {"success": False, "error": "Embeddings not available"}
    if not _db:
        return {"success": False, "error": "Database not loaded"}

    # First, compute the delta to show what will be synced
    delta = await _vector_index.compute_index_delta(_db.customers)

    if delta["in_sync"]:
        return {
            "success": True,
            "message": "Index already in sync - no changes needed",
            "db_count": delta["db_count"],
            "indexed_count": delta["indexed_count"],
            "to_add": 0,
            "to_delete": 0
        }

    # Async mode: run in background with progress tracking
    if async_mode and _progress_tracker:
        total_work = delta["to_add_count"] + delta["to_delete_count"]
        op_id = await _progress_tracker.start_operation(
            operation_type="customer_sync",
            total_items=total_work,
            total_batches=total_work // 50 + 1,
        )

        async def run_sync():
            try:
                await _progress_tracker.update_progress(op_id, phase="Syncing customers (incremental)")

                async def on_progress(processed: int, total: int):
                    await _progress_tracker.update_progress(
                        op_id,
                        processed=processed,
                        message=f"Synced {processed:,} of {total:,} new customers",
                    )

                result = await _vector_index.index_customers_incremental(
                    _db.customers,
                    batch_size=50,
                    progress_callback=on_progress
                )

                await _progress_tracker.complete_operation(
                    op_id,
                    message=f"Sync complete: {result.get('indexed', 0)} added, {result.get('deleted', 0)} deleted"
                )
            except Exception as e:
                await _progress_tracker.fail_operation(op_id, str(e))

        import asyncio
        asyncio.create_task(run_sync())

        return {
            "success": True,
            "message": "Incremental sync started in background",
            "operation_id": op_id,
            "delta": {
                "to_add": delta["to_add_count"],
                "to_delete": delta["to_delete_count"],
            },
            "stream_url": f"/api/data/reindex/stream/{op_id}",
            "status_url": f"/api/data/reindex/status/{op_id}",
        }

    # Synchronous mode
    result = await _vector_index.index_customers_incremental(_db.customers, batch_size=50)

    return {
        "success": result.get("success", False),
        "message": result.get("message", "Sync complete"),
        "indexed": result.get("indexed", 0),
        "deleted": result.get("deleted", 0),
        "failed": result.get("failed", 0),
        "total_in_index": result.get("total_in_index", 0),
        "db_count": len(_db.customers),
        "available_states": _db.get_available_states()
    }


@app.get("/api/data/delta")
async def get_index_delta():
    """Preview what would be synced without actually syncing.

    Use this to check if sync is needed before running it.

    Returns:
        Delta information showing what's new/deleted.
    """
    if not _vector_index or not _vector_index.available:
        return {"success": False, "error": "OpenSearch not available"}
    if not _db:
        return {"success": False, "error": "Database not loaded"}

    delta = await _vector_index.compute_index_delta(_db.customers)

    # Get sample CRIDs for preview (don't return all 2000)
    sample_to_add = list(delta["to_add"])[:10]
    sample_to_delete = list(delta["to_delete"])[:10]

    return {
        "success": True,
        "in_sync": delta["in_sync"],
        "db_count": delta["db_count"],
        "indexed_count": delta["indexed_count"],
        "to_add_count": delta["to_add_count"],
        "to_delete_count": delta["to_delete_count"],
        "sample_to_add": sample_to_add,
        "sample_to_delete": sample_to_delete,
        "recommendation": "Use POST /api/data/sync to apply changes" if not delta["in_sync"] else "No sync needed"
    }


@app.get("/api/data/reindex/stream/{operation_id}")
async def stream_reindex_progress(operation_id: str):
    """Stream reindex progress via Server-Sent Events.

    Connect to this endpoint to receive real-time progress updates.
    Events: 'progress' (ongoing), 'complete' (finished), 'error' (failed)
    """
    if not _progress_tracker:
        return {"success": False, "error": "Progress tracking not available"}

    async def event_stream():
        async for event in _progress_tracker.stream_progress(operation_id):
            yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/data/reindex/status/{operation_id}")
async def get_reindex_status(operation_id: str):
    """Get current status of a reindex operation."""
    if not _progress_tracker:
        return {"success": False, "error": "Progress tracking not available"}

    state = await _progress_tracker.get_progress(operation_id)
    if not state:
        return {"success": False, "error": "Operation not found"}

    return {
        "success": True,
        "operation": state.to_dict(),
        "formatted": {
            "elapsed": format_duration(state.elapsed_seconds),
            "remaining": format_duration(state.estimated_remaining_seconds) if state.estimated_remaining_seconds > 0 else "calculating...",
            "data_processed": format_bytes(state.bytes_processed),
            "rate": f"{state.items_per_second:.0f} items/sec" if state.items_per_second > 0 else "starting...",
        },
    }


@app.get("/api/data/reindex/active")
async def get_active_reindex_operations():
    """Get all active reindex operations."""
    if not _progress_tracker:
        return {"success": False, "error": "Progress tracking not available", "operations": []}

    operations = await _progress_tracker.get_active_operations()
    return {
        "success": True,
        "operations": [op.to_dict() for op in operations],
        "count": len(operations),
    }


# ==================== Customer Index CRUD API ====================
# Incremental indexing: add, update, delete individual customers
# without requiring full reindex

class CustomerData(BaseModel):
    """Customer data for indexing."""
    crid: str | None = None  # Optional - can be set from path param
    name: str
    address: str
    city: str
    state: str
    zip: str
    customer_type: str = "RESIDENTIAL"
    status: str = "ACTIVE"
    move_count: int = 0
    last_move_date: str | None = None
    created_date: str | None = None


class BulkCustomersRequest(BaseModel):
    """Request body for bulk customer indexing."""
    customers: list[CustomerData]


class BulkDeleteRequest(BaseModel):
    """Request body for bulk customer deletion."""
    crids: list[str]


# IMPORTANT: Bulk routes MUST come before {crid} routes to avoid path parameter matching "bulk"

@app.post("/api/customers/bulk")
async def bulk_add_customers_to_index(req: BulkCustomersRequest):
    """Add multiple customers to the search index incrementally.

    This is for INCREMENTAL bulk adds - when adding a batch of new customers.
    Use /api/data/reindex for full reindex when data source changes completely.

    Args:
        req: BulkCustomersRequest with list of customers to add/update.

    Returns:
        Count of successfully indexed customers.
    """
    if not _vector_index or not _vector_index.available:
        return {"success": False, "error": "OpenSearch not available"}

    if not req.customers:
        return {"success": False, "error": "No customers provided"}

    indexed = 0
    failed = 0
    errors = []

    for customer in req.customers:
        crid = customer.crid
        if not crid:
            failed += 1
            errors.append("Missing crid")
            continue

        customer_dict = customer.model_dump()
        success = await _vector_index.index_customer(customer_dict)
        if success:
            indexed += 1
        else:
            failed += 1
            errors.append(f"Failed: {crid}")

    return {
        "success": True,
        "indexed": indexed,
        "failed": failed,
        "total": len(req.customers),
        "errors": errors[:10] if errors else []  # Limit error messages
    }


@app.post("/api/customers/bulk/delete")
async def bulk_delete_customers_from_index(req: BulkDeleteRequest):
    """Remove multiple customers from the search index.

    Args:
        req: BulkDeleteRequest with list of customer IDs to remove.

    Returns:
        Count of successfully deleted customers.
    """
    if not _vector_index or not _vector_index.available:
        return {"success": False, "error": "OpenSearch not available"}

    if not req.crids:
        return {"success": False, "error": "No customer IDs provided"}

    deleted = 0
    failed = 0

    for crid in req.crids:
        success = await _vector_index.delete_customer(crid)
        if success:
            deleted += 1
        else:
            failed += 1

    return {
        "success": True,
        "deleted": deleted,
        "failed": failed,
        "total": len(req.crids)
    }


# Single customer operations (path parameter routes come AFTER bulk routes)

@app.post("/api/customers/{crid}")
async def add_or_update_customer_index(crid: str, customer: CustomerData):
    """Add or update a single customer in the search index.

    This is for INCREMENTAL indexing - add/update one customer at a time.
    Use /api/data/reindex for full reindex when data source changes.

    Args:
        crid: Customer ID (path parameter)
        customer: CustomerData with name, address, city, state, zip, etc.
    """
    if not _vector_index or not _vector_index.available:
        return {"success": False, "error": "OpenSearch not available"}

    # Convert to dict and ensure crid matches path
    customer_dict = customer.model_dump()
    customer_dict["crid"] = crid

    success = await _vector_index.index_customer(customer_dict)
    if success:
        return {
            "success": True,
            "message": f"Customer {crid} indexed successfully",
            "crid": crid
        }
    return {"success": False, "error": f"Failed to index customer {crid}"}


@app.delete("/api/customers/{crid}")
async def delete_customer_from_index(crid: str):
    """Delete a single customer from the search index.

    This is for INCREMENTAL removal - delete one customer at a time.
    The customer data in the source database is not affected.

    Args:
        crid: Customer ID to remove from index.
    """
    if not _vector_index or not _vector_index.available:
        return {"success": False, "error": "OpenSearch not available"}

    success = await _vector_index.delete_customer(crid)
    if success:
        return {
            "success": True,
            "message": f"Customer {crid} removed from index",
            "crid": crid
        }
    return {"success": False, "error": f"Failed to delete customer {crid}"}


@app.get("/api/customers/{crid}/indexed")
async def check_customer_indexed(crid: str):
    """Check if a customer exists in the search index.

    Args:
        crid: Customer ID to check.

    Returns:
        Whether customer is in the index and their indexed data.
    """
    if not _vector_index or not _vector_index.available:
        return {"success": False, "error": "OpenSearch not available", "indexed": False}

    exists = await _vector_index.customer_exists(crid)
    result = {"success": True, "crid": crid, "indexed": exists}

    if exists:
        customer = await _vector_index.get_customer(crid)
        if customer:
            # Remove embedding from response (too large)
            customer.pop("embedding", None)
            result["customer"] = customer

    return result


# ==================== Knowledge Base API ====================

@app.get("/api/knowledge/stats")
async def knowledge_stats():
    if not _knowledge or not _knowledge.available:
        return {"available": False, "error": "Knowledge base not initialized"}
    return await _knowledge.get_stats()


@app.get("/api/knowledge/documents")
async def list_knowledge_documents(category: Optional[str] = None, limit: int = 50):
    if not _knowledge or not _knowledge.available:
        return {"success": False, "documents": [], "error": "Not available"}
    docs = await _knowledge.list_documents(category=category, limit=limit)
    return {"success": True, "documents": docs, "count": len(docs)}


@app.post("/api/knowledge/upload")
async def upload_knowledge_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    category: str = Form("uploaded")
):
    """Upload and persist a document to the knowledge base."""
    global _index_state

    if not _knowledge or not _knowledge.available:
        return {"success": False, "error": "Knowledge base not available"}

    # Ensure upload directory exists
    KNOWLEDGE_UPLOADED_DIR.mkdir(parents=True, exist_ok=True)

    # Generate unique filename (handle duplicates)
    original_name = Path(file.filename).name
    target_path = KNOWLEDGE_UPLOADED_DIR / original_name
    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        counter = 1
        while target_path.exists():
            target_path = KNOWLEDGE_UPLOADED_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

    # Write file permanently
    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    relative_path = str(target_path.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")

    try:
        # Compute hash and index
        content_hash = compute_file_hash(target_path)

        result = await _knowledge.index_document(
            content=target_path,
            filename=relative_path,
            tags=tag_list,
            category=category
        )

        # Update index state
        if result.get("success") and _index_state is not None:
            update_file_state(
                _index_state,
                relative_path,
                result.get("doc_id", ""),
                content_hash,
                result.get("chunks_indexed", 0)
            )
            save_index_state(INDEX_STATE_FILE, _index_state)

        return {
            **result,
            "persisted_path": relative_path,
            "content_hash": content_hash
        }
    except Exception as e:
        # Clean up file on failure
        target_path.unlink(missing_ok=True)
        return {"success": False, "error": str(e)}


@app.post("/api/knowledge/upload-text")
async def upload_knowledge_text(
    title: str = Form(...),
    content: str = Form(...),
    tags: Optional[str] = Form(None),
    category: str = Form("general")
):
    """Upload raw text to the knowledge base."""
    if not _knowledge or not _knowledge.available:
        return {"success": False, "error": "Knowledge base not available"}

    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    return await _knowledge.index_document(
        content=content,
        filename=title,
        tags=tag_list,
        category=category
    )


class KnowledgeSearchRequest(BaseModel):
    query: str
    limit: int = 5
    tags: list[str] | None = None
    category: str | None = None


@app.post("/api/knowledge/search")
async def search_knowledge(req: KnowledgeSearchRequest):
    """Search the knowledge base."""
    if not _knowledge or not _knowledge.available:
        return {"success": False, "hits": [], "error": "Not available"}
    return await _knowledge.search(query=req.query, limit=req.limit, tags=req.tags, category=req.category)


@app.get("/api/knowledge/search")
async def search_knowledge_get(
    q: str,
    limit: int = 5,
    tags: Optional[str] = None,
    category: Optional[str] = None
):
    """Search the knowledge base (GET version)."""
    if not _knowledge or not _knowledge.available:
        return {"success": False, "hits": [], "error": "Not available"}
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()] if tags else None
    return await _knowledge.search(query=q, limit=limit, tags=tag_list, category=category)


@app.delete("/api/knowledge/documents/{doc_id}")
async def delete_knowledge_document(doc_id: str):
    """Delete a document from the knowledge base."""
    if not _knowledge or not _knowledge.available:
        return {"success": False, "error": "Not available"}
    result = await _knowledge.delete_document(doc_id)
    return {"success": result.get("deleted", 0) > 0, "doc_id": doc_id, **result}


@app.post("/api/knowledge/reindex")
async def reindex_knowledge_documents(force: bool = False):
    """
    Manually trigger re-indexing of knowledge documents.

    Args:
        force: If True, delete all documents and reindex everything.
               If False, only index new/modified files (incremental).
    """
    global _index_state

    if not _knowledge or not _knowledge.available:
        return {"success": False, "error": "Knowledge base not available"}

    if force:
        # Delete all documents and clear index state
        docs = await _knowledge.list_documents(limit=1000)
        deleted = 0
        for doc in docs:
            await _knowledge.delete_document(doc["doc_id"])
            deleted += 1
        # Clear index state for full reindex
        from icda.knowledge_index_state import create_empty_state, mark_full_reindex
        _index_state = create_empty_state()
        mark_full_reindex(_index_state)
        print(f"  Force reindex: deleted {deleted} documents")

    # Reload state for incremental reindex
    if _index_state is None:
        _index_state = load_index_state(INDEX_STATE_FILE)

    result = await auto_index_knowledge_documents(_knowledge, _index_state)
    return {"success": True, "mode": "full" if force else "incremental", **result}


# ==================== Admin API ====================

class AdminChunkUpdate(BaseModel):
    """Model for updating chunk metadata."""
    tags: list[str] | None = None
    category: str | None = None
    quality_score: float | None = None


class AdminSearchTest(BaseModel):
    """Model for testing search queries."""
    query: str
    limit: int = 10
    index: str | None = None
    filters: dict | None = None
    explain: bool = False


class AdminSavedQuery(BaseModel):
    """Model for saving test queries."""
    name: str
    query: str
    index: str | None = None
    filters: dict | None = None
    notes: str | None = None


# In-memory store for saved queries (would be Redis in production)
_saved_queries: dict = {}
_saved_query_counter: int = 0


@app.get("/api/admin/chunks")
async def admin_list_chunks(
    offset: int = 0,
    limit: int = 50,
    category: str | None = None,
    min_quality: float | None = None,
    max_quality: float | None = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """List all chunks with pagination and filtering."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}
    if not _knowledge or not _knowledge.available:
        return {"success": False, "chunks": [], "error": "Knowledge base not available"}

    all_docs = await _knowledge.list_documents(category=category, limit=1000)
    chunks = []

    for doc in all_docs:
        doc_chunks = await _knowledge.get_document_chunks(doc.get("doc_id", ""))
        for chunk in doc_chunks:
            quality = chunk.get("quality_score", 1.0)
            if min_quality and quality < min_quality:
                continue
            if max_quality and quality > max_quality:
                continue
            chunks.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "doc_id": doc.get("doc_id", ""),
                "filename": doc.get("filename", "unknown"),
                "content": chunk.get("content", "")[:500],
                "content_length": len(chunk.get("content", "")),
                "category": doc.get("category", "general"),
                "tags": doc.get("tags", []),
                "quality_score": quality,
                "created_at": chunk.get("created_at", ""),
            })

    if sort_by == "quality_score":
        chunks.sort(key=lambda x: x.get("quality_score", 0), reverse=(sort_order == "desc"))
    elif sort_by == "content_length":
        chunks.sort(key=lambda x: x.get("content_length", 0), reverse=(sort_order == "desc"))
    else:
        chunks.sort(key=lambda x: x.get("created_at", ""), reverse=(sort_order == "desc"))

    total = len(chunks)
    paginated = chunks[offset:offset + limit]

    return {
        "success": True,
        "chunks": paginated,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total
    }


@app.get("/api/admin/chunks/{chunk_id}")
async def admin_get_chunk(chunk_id: str):
    """Get detailed information about a specific chunk."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}
    if not _knowledge or not _knowledge.available:
        return {"success": False, "error": "Knowledge base not available"}

    all_docs = await _knowledge.list_documents(limit=1000)
    for doc in all_docs:
        doc_chunks = await _knowledge.get_document_chunks(doc.get("doc_id", ""))
        for chunk in doc_chunks:
            if chunk.get("chunk_id") == chunk_id:
                return {
                    "success": True,
                    "chunk": {
                        "chunk_id": chunk_id,
                        "doc_id": doc.get("doc_id", ""),
                        "filename": doc.get("filename", "unknown"),
                        "content": chunk.get("content", ""),
                        "content_length": len(chunk.get("content", "")),
                        "category": doc.get("category", "general"),
                        "tags": doc.get("tags", []),
                        "quality_score": chunk.get("quality_score", 1.0),
                        "embedding_preview": chunk.get("embedding", [])[:10] if chunk.get("embedding") else None,
                        "embedding_dimensions": len(chunk.get("embedding", [])) if chunk.get("embedding") else 0,
                        "created_at": chunk.get("created_at", ""),
                        "metadata": chunk.get("metadata", {})
                    }
                }

    return {"success": False, "error": "Chunk not found"}


@app.patch("/api/admin/chunks/{chunk_id}")
async def admin_update_chunk(chunk_id: str, update: AdminChunkUpdate):
    """Update chunk metadata (tags, category, quality_score)."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}
    if not _knowledge or not _knowledge.available:
        return {"success": False, "error": "Knowledge base not available"}

    return {"success": False, "error": "Chunk updates not yet implemented"}


@app.delete("/api/admin/chunks/{chunk_id}")
async def admin_delete_chunk(chunk_id: str):
    """Delete a specific chunk."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}
    if not _knowledge or not _knowledge.available:
        return {"success": False, "error": "Knowledge base not available"}

    return {"success": False, "error": "Individual chunk deletion not yet implemented"}


@app.post("/api/admin/chunks/{chunk_id}/reembed")
async def admin_reembed_chunk(chunk_id: str):
    """Regenerate embedding for a specific chunk."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}
    if not _knowledge or not _knowledge.available:
        return {"success": False, "error": "Knowledge base not available"}
    if not _embedder or not _embedder.available:
        return {"success": False, "error": "Embedding service not available"}

    return {"success": False, "error": "Re-embedding not yet implemented"}


@app.get("/api/admin/chunks/embeddings/visualization")
async def admin_embedding_visualization(sample_size: int = 100):
    """Get 2D projection of chunk embeddings for visualization (t-SNE/UMAP style)."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}
    if not _knowledge or not _knowledge.available:
        return {"success": False, "points": [], "error": "Knowledge base not available"}

    return {
        "success": True,
        "points": [],
        "message": "Embedding visualization requires sklearn/umap - not yet implemented"
    }


@app.get("/api/admin/index/stats")
async def admin_index_stats():
    """Get detailed statistics for all indexes."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    stats = {
        "knowledge": {},
        "customers": {},
        "addresses": {},
        "services": {},
        "enforcer": {}
    }

    if _knowledge and _knowledge.available:
        kb_stats = await _knowledge.get_stats()
        stats["knowledge"] = kb_stats

    if _vector_index and _vector_index.available:
        customer_count = await _vector_index.customer_count()
        stats["customers"] = {
            "indexed": customer_count,
            "index_name": _vector_index.customer_index
        }

    if _address_vector_index:
        stats["addresses"] = {
            "available": True,
            "total_addresses": _address_index.total_addresses if _address_index else 0
        }

    stats["services"] = {
        "redis": _cache.available if _cache else False,
        "opensearch": _vector_index.available if _vector_index else False,
        "embeddings": _embedder.available if _embedder else False,
        "nova_ai": _nova.available if _nova else False
    }

    if _enforcer:
        stats["enforcer"] = _enforcer.get_metrics()

    return {"success": True, "stats": stats}


@app.get("/api/admin/index/health")
async def admin_index_health():
    """Get health status of all indexes."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    health = {
        "overall": "healthy",
        "indexes": {},
        "issues": []
    }

    if _knowledge and _knowledge.available:
        kb_stats = await _knowledge.get_stats()
        kb_health = "healthy"
        if kb_stats.get("total_chunks", 0) == 0:
            kb_health = "empty"
            health["issues"].append("Knowledge base has no chunks indexed")
        health["indexes"]["knowledge"] = {
            "status": kb_health,
            "chunks": kb_stats.get("total_chunks", 0),
            "documents": kb_stats.get("unique_documents", 0)
        }
    else:
        health["indexes"]["knowledge"] = {"status": "unavailable"}
        health["issues"].append("Knowledge base not available")

    if _vector_index and _vector_index.available:
        customer_count = await _vector_index.customer_count()
        health["indexes"]["customers"] = {
            "status": "healthy" if customer_count > 0 else "empty",
            "count": customer_count
        }
    else:
        health["indexes"]["customers"] = {"status": "unavailable"}
        health["issues"].append("Customer index not available")

    if len(health["issues"]) > 0:
        health["overall"] = "degraded"

    return {"success": True, "health": health}


@app.post("/api/admin/index/reindex")
async def admin_trigger_reindex(index_name: str = "all"):
    """Trigger re-indexing of specified index."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    results = {}

    if index_name in ("all", "knowledge"):
        if _knowledge and _knowledge.available:
            state = load_index_state(INDEX_STATE_FILE) if _index_state is None else _index_state
            result = await auto_index_knowledge_documents(_knowledge, state)
            results["knowledge"] = result
        else:
            results["knowledge"] = {"error": "Not available"}

    if index_name in ("all", "customers"):
        if _vector_index and _vector_index.available and _db:
            count = await _vector_index.index_customers(_db.customers)
            results["customers"] = {"indexed": count}
        else:
            results["customers"] = {"error": "Not available"}

    return {"success": True, "results": results}


@app.delete("/api/admin/index/{index_name}")
async def admin_clear_index(index_name: str):
    """Clear all data from specified index."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    if index_name == "knowledge":
        if _knowledge and _knowledge.available:
            docs = await _knowledge.list_documents(limit=1000)
            deleted = 0
            for doc in docs:
                result = await _knowledge.delete_document(doc.get("doc_id", ""))
                deleted += result.get("deleted", 0)
            return {"success": True, "deleted": deleted}
        return {"success": False, "error": "Knowledge base not available"}

    return {"success": False, "error": f"Unknown index: {index_name}"}


@app.get("/api/admin/index/export")
async def admin_export_stats():
    """Export comprehensive index statistics for reporting."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    from datetime import datetime

    stats = await admin_index_stats()
    health = await admin_index_health()

    return {
        "success": True,
        "export": {
            "timestamp": datetime.utcnow().isoformat(),
            "stats": stats.get("stats", {}),
            "health": health.get("health", {}),
            "config": {
                "opensearch_host": cfg.opensearch_host,
                "redis_url": bool(cfg.redis_url),
                "federation_enabled": cfg.enable_federation,
                "enforcer_enabled": cfg.enable_llm_enforcer,
                "admin_enabled": cfg.admin_enabled
            }
        }
    }


@app.get("/api/admin/config/embeddings")
async def admin_embedding_config():
    """Get current embedding configuration and provider status.

    Returns embedding provider settings, model info, and availability status.
    """
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    # Get embedding client info
    embedding_info = {
        "available": _embedder.available if _embedder else False,
        "provider": "titan",
        "model": cfg.titan_model if hasattr(cfg, 'titan_model') else "amazon.titan-embed-text-v2:0",
        "dimensions": 1024,
        "region": cfg.aws_region,
    }

    # Check if we can get more details from the embedder
    if _embedder and hasattr(_embedder, 'model_id'):
        embedding_info["model"] = _embedder.model_id

    # Try to get ingestion config if available
    ingestion_config = None
    try:
        from icda.ingestion import IngestionConfig, EmbeddingProviderConfig
        config = IngestionConfig.from_env()
        ingestion_config = {
            "primary_provider": config.embeddings.primary_provider,
            "fallback_order": config.embeddings.fallback_order,
            "target_dimension": config.embeddings.target_dimension,
            "enable_normalization": config.embeddings.enable_normalization,
            "batch_size": config.embeddings.batch_size,
            "titan_model": config.embeddings.titan_model,
            "circuit_breaker_threshold": config.embeddings.circuit_breaker_threshold,
            "circuit_breaker_timeout": config.embeddings.circuit_breaker_timeout,
        }
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not load ingestion config: {e}")

    return {
        "success": True,
        "config": {
            "current_provider": embedding_info,
            "ingestion": ingestion_config,
            "services": {
                "titan_embeddings": _embedder.available if _embedder else False,
                "opensearch_vectors": _vector_index.available if _vector_index else False,
                "address_embeddings": _address_embedder.available if _address_embedder else False,
            }
        }
    }


@app.post("/api/admin/search/test")
async def admin_test_search(req: AdminSearchTest):
    """Test a search query with debug information."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    import time
    start = time.time()

    results = []
    debug_info = {
        "query": req.query,
        "index": req.index or "knowledge",
        "filters": req.filters,
        "explain": req.explain
    }

    if (req.index or "knowledge") == "knowledge":
        if _knowledge and _knowledge.available:
            search_result = await _knowledge.search(
                query=req.query,
                limit=req.limit,
                tags=req.filters.get("tags") if req.filters else None,
                category=req.filters.get("category") if req.filters else None
            )
            results = search_result.get("hits", [])
            debug_info["backend"] = search_result.get("backend", "unknown")
            debug_info["search_type"] = search_result.get("search_type", "unknown")
    elif req.index == "customers":
        if _vector_index and _vector_index.available:
            search_result = await _vector_index.search_customers_semantic(
                req.query,
                req.limit,
                req.filters
            )
            results = search_result.get("hits", [])

    elapsed = time.time() - start
    debug_info["elapsed_ms"] = round(elapsed * 1000, 2)
    debug_info["result_count"] = len(results)

    return {
        "success": True,
        "results": results,
        "debug": debug_info if req.explain else None
    }


@app.post("/api/admin/search/saved")
async def admin_save_query(req: AdminSavedQuery):
    """Save a test query for later use."""
    global _saved_query_counter

    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    _saved_query_counter += 1
    query_id = f"sq_{_saved_query_counter}"

    from datetime import datetime
    _saved_queries[query_id] = {
        "id": query_id,
        "name": req.name,
        "query": req.query,
        "index": req.index,
        "filters": req.filters,
        "notes": req.notes,
        "created_at": datetime.utcnow().isoformat()
    }

    return {"success": True, "query_id": query_id}


@app.get("/api/admin/search/saved")
async def admin_list_saved_queries():
    """List all saved test queries."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    return {
        "success": True,
        "queries": list(_saved_queries.values()),
        "count": len(_saved_queries)
    }


@app.delete("/api/admin/search/saved/{query_id}")
async def admin_delete_saved_query(query_id: str):
    """Delete a saved test query."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    if query_id in _saved_queries:
        del _saved_queries[query_id]
        return {"success": True, "deleted": query_id}

    return {"success": False, "error": "Query not found"}


@app.post("/api/admin/search/saved/{query_id}/run")
async def admin_run_saved_query(query_id: str):
    """Run a saved test query."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    if query_id not in _saved_queries:
        return {"success": False, "error": "Query not found"}

    saved = _saved_queries[query_id]
    return await admin_test_search(AdminSearchTest(
        query=saved["query"],
        index=saved.get("index"),
        filters=saved.get("filters"),
        explain=True
    ))


@app.get("/api/admin/enforcer/metrics")
async def admin_enforcer_metrics():
    """Get LLM Enforcer pipeline metrics."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    if not _enforcer:
        return {
            "success": True,
            "available": False,
            "metrics": None,
            "message": "Enforcer not initialized"
        }

    return {
        "success": True,
        "available": _enforcer.available,
        "metrics": _enforcer.get_metrics(),
        "config": {
            "provider": cfg.secondary_llm_provider,
            "model": cfg.secondary_llm_model or "auto",
            "chunk_threshold": cfg.enforcer_chunk_threshold,
            "query_sample_rate": cfg.enforcer_query_sample_rate,
            "validation_interval_hours": cfg.enforcer_validation_interval
        }
    }


@app.get("/api/admin/chunks/quality")
async def admin_chunks_quality(threshold: float = 0.6, limit: int = 50):
    """Get chunks below quality threshold for review."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}
    if not _knowledge or not _knowledge.available:
        return {"success": False, "chunks": [], "error": "Knowledge base not available"}

    all_docs = await _knowledge.list_documents(limit=1000)
    low_quality = []

    for doc in all_docs:
        doc_chunks = await _knowledge.get_document_chunks(doc.get("doc_id", ""))
        for chunk in doc_chunks:
            quality = chunk.get("quality_score", 1.0)
            if quality < threshold:
                low_quality.append({
                    "chunk_id": chunk.get("chunk_id", ""),
                    "doc_id": doc.get("doc_id", ""),
                    "filename": doc.get("filename", "unknown"),
                    "content_preview": chunk.get("content", "")[:200],
                    "quality_score": quality,
                    "category": doc.get("category", "general"),
                })

    low_quality.sort(key=lambda x: x.get("quality_score", 0))
    return {
        "success": True,
        "chunks": low_quality[:limit],
        "total_below_threshold": len(low_quality),
        "threshold": threshold
    }


@app.post("/api/admin/enforcer/evaluate-chunk")
async def admin_evaluate_chunk(chunk_id: str, content: str, source: str = "manual"):
    """Manually trigger L1 chunk quality evaluation."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    if not _enforcer or not _enforcer.available:
        return {"success": False, "error": "Enforcer not available"}

    result = await _enforcer.evaluate_chunk(
        chunk_id=chunk_id,
        content=content,
        source=source,
        content_type="text"
    )

    return {
        "success": True,
        "result": {
            "passed": result.passed,
            "overall_score": result.overall_score,
            "coherence": result.quality.coherence,
            "completeness": result.quality.completeness,
            "relevance": result.quality.relevance,
            "issues": result.quality.issues,
            "suggestions": result.quality.suggestions
        }
    }


@app.post("/api/admin/enforcer/validate-index")
async def admin_validate_index():
    """Manually trigger L2 index validation."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    if not _enforcer or not _enforcer.available:
        return {"success": False, "error": "Enforcer not available"}

    if not _knowledge or not _knowledge.available:
        return {"success": False, "error": "Knowledge base not available"}

    all_docs = await _knowledge.list_documents(limit=500)
    chunks = []
    for doc in all_docs:
        doc_chunks = await _knowledge.get_document_chunks(doc.get("doc_id", ""))
        for chunk in doc_chunks:
            chunks.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "content": chunk.get("content", ""),
                "source": doc.get("filename", "unknown"),
                "category": doc.get("category", "general")
            })

    report = await _enforcer.validate_index(chunks)

    return {
        "success": True,
        "report": {
            "health_score": report.health_score,
            "total_chunks": report.total_chunks,
            "sampled_chunks": report.sampled_chunks,
            "duplicate_groups": len(report.duplicate_groups),
            "stale_chunks": len(report.stale_chunks),
            "coverage_gaps": report.coverage_gaps,
            "recommendations": report.recommendations
        }
    }


# ==================== Index Federation ====================

class FederatedSearchRequest(BaseModel):
    """Request model for federated search."""
    query: str = Field(..., description="Search query")
    indexes: Optional[list[str]] = Field(None, description="Specific indexes to search (null for auto-route)")
    k: int = Field(default=10, ge=1, le=100, description="Results per index")
    deduplicate: bool = Field(default=True, description="Deduplicate results across indexes")


@app.post("/api/federated/search")
async def federated_search(req: FederatedSearchRequest):
    """
    Execute a federated search across multiple indexes.

    Auto-routes query to relevant indexes (code, knowledge, customers) based on
    semantic routing from the master index. Returns merged, deduplicated results
    with source attribution.
    """
    if not _federation:
        return {
            "success": False,
            "error": "Index Federation not enabled. Set ENABLE_FEDERATION=true in .env"
        }

    import time
    start = time.time()

    try:
        result = await _federation.search(
            query=req.query,
            indexes=req.indexes,
            k=req.k,
            deduplicate=req.deduplicate,
        )

        # Convert results to serializable format
        hits = []
        for r in result.results:
            hits.append({
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "text": r.text[:500] if r.text else "",  # Truncate for response
                "score": round(r.score, 4),
                "source_index": r.source_index,
                "is_deduplicated": r.is_deduplicated,
                "alternate_sources": r.alternate_sources,
            })

        return {
            "success": True,
            "results": hits,
            "total": result.total_hits,
            "searched_indexes": result.searched_indexes,
            "routing_scores": {k: round(v, 4) for k, v in result.routing_scores.items()},
            "deduplicated_count": result.deduplicated_count,
            "processing_ms": result.processing_time_ms,
            "query_time_ms": round((time.time() - start) * 1000, 2),
        }
    except Exception as e:
        logger.error(f"Federated search failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/federated/search/code")
async def federated_search_code(
    q: str,
    language: Optional[str] = None,
    limit: int = 10
):
    """Search code index only."""
    if not _federation:
        return {"success": False, "error": "Index Federation not enabled"}

    try:
        results = await _federation.search_code(q, language=language, k=limit)
        hits = [
            {
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "text": r.text[:500] if r.text else "",
                "score": round(r.score, 4),
                "filename": r.metadata.get("filename", ""),
                "language": r.metadata.get("language", ""),
                "symbols": r.metadata.get("symbols", []),
            }
            for r in results
        ]
        return {"success": True, "results": hits, "total": len(hits)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/federated/search/knowledge")
async def federated_search_knowledge(
    q: str,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 10
):
    """Search knowledge index only."""
    if not _federation:
        return {"success": False, "error": "Index Federation not enabled"}

    try:
        tag_list = tags.split(",") if tags else None
        results = await _federation.search_knowledge(q, category=category, tags=tag_list, k=limit)
        hits = [
            {
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "text": r.text[:500] if r.text else "",
                "score": round(r.score, 4),
                "filename": r.metadata.get("filename", ""),
                "category": r.metadata.get("category", ""),
                "tags": r.metadata.get("tags", []),
            }
            for r in results
        ]
        return {"success": True, "results": hits, "total": len(hits)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/federated/search/customers")
async def federated_search_customers(
    q: str,
    state: Optional[str] = None,
    limit: int = 10
):
    """Search customers index only."""
    if not _federation:
        return {"success": False, "error": "Index Federation not enabled"}

    try:
        results = await _federation.search_customers(q, state=state, k=limit)
        hits = [
            {
                "crid": r.metadata.get("crid", r.doc_id),
                "name": r.metadata.get("name", ""),
                "address": r.metadata.get("address", ""),
                "city": r.metadata.get("city", ""),
                "state": r.metadata.get("state", ""),
                "zip": r.metadata.get("zip", ""),
                "score": round(r.score, 4),
            }
            for r in results
        ]
        return {"success": True, "results": hits, "total": len(hits)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/admin/federation/stats")
async def admin_federation_stats():
    """Get statistics for all federated indexes."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    if not _federation:
        return {
            "success": False,
            "error": "Index Federation not enabled. Set ENABLE_FEDERATION=true in .env"
        }

    try:
        stats = await _federation.get_all_stats()
        return {
            "success": True,
            "federation_enabled": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get federation stats: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/admin/federation/sync")
async def admin_federation_sync_document(
    doc_id: str,
    source_index: str,
    title: str,
    summary: str,
    chunk_count: int = 0,
    tags: list[str] = None,
    category: str = "general"
):
    """Manually sync a document to the master index."""
    if not cfg.admin_enabled:
        return {"success": False, "error": "Admin API disabled"}

    if not _federation:
        return {"success": False, "error": "Index Federation not enabled"}

    try:
        success = await _federation.sync_to_master(
            doc_id=doc_id,
            source_index=source_index,
            title=title,
            summary=summary,
            chunk_count=chunk_count,
            tags=tags or [],
            category=category,
        )
        return {"success": success}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# INGESTION API ENDPOINTS - Address data ingestion pipeline
# =============================================================================

class IngestionFileRequest(BaseModel):
    """Request to ingest addresses from a file."""
    file_path: str = Field(..., description="Path to address data file (JSON/CSV)")
    embedding_path: Optional[str] = Field(None, description="Optional path to pre-computed embeddings")
    file_format: str = Field("json", description="File format: 'json' or 'csv'")


@app.get("/api/ingestion/status")
async def ingestion_status():
    """Get ingestion pipeline status and statistics."""
    if not _ingestion_pipeline:
        return {
            "available": False,
            "error": "Ingestion pipeline not initialized",
        }

    return {
        "available": True,
        "initialized": _ingestion_pipeline._initialized,
        "stats": _ingestion_pipeline._stats,
        "config": {
            "enable_ai_schema_mapping": _ingestion_pipeline._config.enable_ai_schema_mapping,
            "enforcers_enabled": _ingestion_pipeline._config.enforcers.enabled,
            "max_concurrent": _ingestion_pipeline._config.max_concurrent,
            "batch_size": _ingestion_pipeline._config.batch_size,
        },
    }


@app.post("/api/ingestion/file")
async def ingest_file(request: IngestionFileRequest):
    """
    Ingest address data from a file.

    Supports JSON and CSV formats. Optionally accepts pre-computed embeddings.
    """
    if not _ingestion_pipeline:
        return {"success": False, "error": "Ingestion pipeline not initialized"}

    try:
        result: IngestionBatchResult = await _ingestion_pipeline.ingest_file(
            file_path=request.file_path,
            embedding_path=request.embedding_path,
            file_format=request.file_format,
        )

        return {
            "success": True,
            "summary": result.summary.to_dict() if result.summary else None,
            "batch_id": result.batch_id,
            "source": result.source,
            "total_records": len(result.records),
            "approved": sum(1 for r in result.records if r.status.value == "approved"),
            "rejected": sum(1 for r in result.records if r.status.value == "rejected"),
        }
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {request.file_path}"}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/ingestion/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    file_format: str = Form("auto"),
):
    """
    Upload and ingest an address file directly.

    The file is saved temporarily and processed through the ingestion pipeline.
    """
    import tempfile
    import os

    if not _ingestion_pipeline:
        return {"success": False, "error": "Ingestion pipeline not initialized"}

    # Determine format from filename if auto
    if file_format == "auto":
        filename = file.filename.lower() if file.filename else ""
        if filename.endswith(".csv"):
            file_format = "csv"
        elif filename.endswith(".json"):
            file_format = "json"
        else:
            return {"success": False, "error": "Cannot determine file format. Specify 'json' or 'csv'."}

    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=f".{file_format}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Process through pipeline
        result = await _ingestion_pipeline.ingest_file(
            file_path=tmp_path,
            file_format=file_format,
        )

        # Clean up
        os.unlink(tmp_path)

        return {
            "success": True,
            "filename": file.filename,
            "summary": result.summary.to_dict() if result.summary else None,
            "batch_id": result.batch_id,
            "total_records": len(result.records),
            "approved": sum(1 for r in result.records if r.status.value == "approved"),
            "rejected": sum(1 for r in result.records if r.status.value == "rejected"),
        }
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        logger.error(f"Ingestion upload failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/ingestion/progress")
async def ingestion_progress():
    """Get real-time ingestion progress."""
    if not _ingestion_pipeline or not _ingestion_pipeline._progress_tracker:
        return {
            "available": False,
            "current_batch": None,
            "progress": 0,
        }

    tracker = _ingestion_pipeline._progress_tracker
    return {
        "available": True,
        "current_batch": tracker.current_batch_id,
        "progress": tracker.progress_percent,
        "records_processed": tracker.records_processed,
        "records_total": tracker.records_total,
        "current_stage": tracker.current_stage,
        "errors": tracker.error_count,
    }


@app.get("/api/ingestion/config")
async def ingestion_config():
    """Get ingestion pipeline configuration."""
    if not _ingestion_pipeline:
        return {"available": False}

    config = _ingestion_pipeline._config
    return {
        "available": True,
        "enable_ai_schema_mapping": config.enable_ai_schema_mapping,
        "max_concurrent": config.max_concurrent,
        "batch_size": config.batch_size,
        "progress_interval": config.progress_interval,
        "schema_cache_path": str(config.schema_cache_path) if config.schema_cache_path else None,
        "enforcers": {
            "enabled": config.enforcers.enabled,
            "fail_fast": config.enforcers.fail_fast,
            "required_fields": config.enforcers.required_fields,
            "similarity_threshold": config.enforcers.similarity_threshold,
            "min_quality_score": config.enforcers.min_quality_score,
        },
        "embeddings": {
            "target_dimension": config.embeddings.target_dimension,
            "enable_normalization": config.embeddings.enable_normalization,
            "fallback_order": config.embeddings.fallback_order,
        },
    }


# ==================== Frontend ====================

frontend_dist = BASE_DIR / "frontend" / "dist"
frontend_index = frontend_dist / "index.html"

# Mount static assets BEFORE catch-all routes
if frontend_dist.exists() and (frontend_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

# Serve site.webmanifest from dist if present, otherwise fall back to frontend/public for development
@app.get("/site.webmanifest")
async def site_manifest():
    candidates = [
        frontend_dist / "site.webmanifest",
        BASE_DIR / "frontend" / "public" / "site.webmanifest",
    ]
    for p in candidates:
        if p.exists():
            return FileResponse(p)
    return JSONResponse(status_code=404, content={"error": "site.webmanifest not found"})


@app.get("/", response_class=HTMLResponse)
async def root():
    if frontend_index.exists():
        return frontend_index.read_text()
    return (BASE_DIR / "templates/index.html").read_text()


# SPA catch-all route - serves index.html for client-side routing (admin, etc.)
async def _serve_spa():
    if frontend_index.exists():
        return HTMLResponse(frontend_index.read_text())
    return HTMLResponse((BASE_DIR / "templates/index.html").read_text())


@app.get("/admin", response_class=HTMLResponse)
async def spa_admin_root():
    return await _serve_spa()


@app.get("/admin/{path:path}", response_class=HTMLResponse)
async def spa_admin_path(path: str):
    return await _serve_spa()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
