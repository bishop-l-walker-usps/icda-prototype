"""Configuration with sensible defaults for LITE MODE (no external services)."""

from dataclasses import dataclass, field
from os import getenv


def _parse_bool(value: str, default: bool = True) -> bool:
    """Parse boolean from environment variable."""
    if not value:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _parse_float(value: str, default: float) -> float:
    """Parse float from environment variable."""
    try:
        return float(value) if value else default
    except ValueError:
        return default


def _parse_int(value: str, default: int) -> int:
    """Parse int from environment variable."""
    try:
        return int(value) if value else default
    except ValueError:
        return default


@dataclass(slots=True, frozen=True)
class Config:
    # ==================== AWS (optional - empty = LITE MODE) ====================
    aws_region: str = field(default_factory=lambda: getenv("AWS_REGION", "us-east-1"))
    # Default to Nova Micro for base model (cheapest, fastest for simple queries)
    # Model router will escalate to Lite/Pro based on complexity
    nova_model: str = field(default_factory=lambda: getenv("NOVA_MODEL", "us.amazon.nova-micro-v1:0"))
    titan_embed_model: str = field(default_factory=lambda: getenv("TITAN_EMBED_MODEL", "amazon.titan-embed-text-v2:0"))
    embed_dimensions: int = 1024

    # ==================== Nova Model Routing (NEW) ====================
    # Nova Lite - default for all queries (Micro has content filter issues)
    nova_lite_model: str = field(default_factory=lambda: getenv("NOVA_LITE_MODEL", "us.amazon.nova-lite-v1:0"))
    # Nova Pro - for complex queries, low confidence, multi-part queries
    nova_pro_model: str = field(default_factory=lambda: getenv("NOVA_PRO_MODEL", "us.amazon.nova-pro-v1:0"))
    # Confidence threshold - below this triggers escalation to Pro
    model_routing_threshold: float = field(
        default_factory=lambda: _parse_float(getenv("MODEL_ROUTING_THRESHOLD", ""), 0.6)
    )
    # Pagination threshold - above this suggests download
    pagination_threshold: int = field(
        default_factory=lambda: _parse_int(getenv("PAGINATION_THRESHOLD", ""), 50)
    )
    # Preview size for paginated results
    pagination_preview_size: int = field(
        default_factory=lambda: _parse_int(getenv("PAGINATION_PREVIEW_SIZE", ""), 15)
    )

    # ==================== Cache (optional - empty = in-memory fallback) ====================
    cache_ttl: int = 43200  # 12 hours
    redis_url: str = field(default_factory=lambda: getenv("REDIS_URL", ""))

    # ==================== OpenSearch (optional - empty = keyword fallback) ====================
    opensearch_host: str = field(default_factory=lambda: getenv("OPENSEARCH_HOST", ""))
    opensearch_index: str = field(default_factory=lambda: getenv("OPENSEARCH_INDEX", "icda-vectors"))

    # ==================== Index Hierarchy (NEW) ====================
    index_master: str = field(default_factory=lambda: getenv("INDEX_MASTER", "icda-master"))
    index_code: str = field(default_factory=lambda: getenv("INDEX_CODE", "icda-code"))
    index_knowledge: str = field(default_factory=lambda: getenv("INDEX_KNOWLEDGE", "icda-knowledge"))
    index_customers: str = field(default_factory=lambda: getenv("INDEX_CUSTOMERS", "icda-customers"))

    # ==================== Secondary LLM Enforcer ====================
    # Provider: auto, gemini, openai, claude, openrouter
    secondary_llm_provider: str = field(
        default_factory=lambda: getenv("SECONDARY_LLM_PROVIDER", "auto")
    )
    # Provider-specific API keys (auto-detection checks all)
    gemini_api_key: str = field(default_factory=lambda: getenv("GEMINI_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: getenv("ANTHROPIC_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: getenv("OPENROUTER_API_KEY", ""))
    # Model override (empty = use provider default)
    secondary_llm_model: str = field(
        default_factory=lambda: getenv("SECONDARY_LLM_MODEL", "")
    )
    # Enforcement thresholds
    enforcer_chunk_threshold: float = field(
        default_factory=lambda: _parse_float(getenv("ENFORCER_CHUNK_THRESHOLD", ""), 0.6)
    )
    enforcer_query_sample_rate: float = field(
        default_factory=lambda: _parse_float(getenv("ENFORCER_QUERY_SAMPLE_RATE", ""), 0.1)
    )
    enforcer_validation_interval: int = field(
        default_factory=lambda: _parse_int(getenv("ENFORCER_VALIDATION_INTERVAL", ""), 6)
    )

    # ==================== AgentCore Memory (optional - empty = local fallback) ====================
    agentcore_memory_id: str = field(default_factory=lambda: getenv("AGENTCORE_MEMORY_ID", ""))
    agentcore_region: str = field(default_factory=lambda: getenv("AGENTCORE_REGION", "us-east-1"))
    agentcore_enabled: bool = field(
        default_factory=lambda: _parse_bool(getenv("AGENTCORE_ENABLED", ""), True)
    )
    agentcore_use_ltm: bool = field(
        default_factory=lambda: _parse_bool(getenv("AGENTCORE_USE_LTM", ""), True)
    )
    agentcore_stm_retention_days: int = field(
        default_factory=lambda: _parse_int(getenv("AGENTCORE_STM_RETENTION_DAYS", ""), 7)
    )
    agentcore_ltm_retention_days: int = field(
        default_factory=lambda: _parse_int(getenv("AGENTCORE_LTM_RETENTION_DAYS", ""), 30)
    )

    # ==================== C Library Data Source ====================
    # Path to C library export file (weekly metadata export)
    c_library_export_path: str = field(
        default_factory=lambda: getenv("C_LIBRARY_EXPORT_PATH", "")
    )
    # C library REST API URL for direct fetches
    c_library_api_url: str = field(
        default_factory=lambda: getenv("C_LIBRARY_API_URL", "")
    )
    # Timeout for C library API requests (seconds)
    c_library_timeout: float = field(
        default_factory=lambda: _parse_float(getenv("C_LIBRARY_TIMEOUT", ""), 30.0)
    )

    # ==================== Address Validation Thresholds (UNIFIED) ====================
    # Exact match - address verified with high certainty
    address_threshold_exact: float = field(
        default_factory=lambda: _parse_float(getenv("ADDRESS_THRESHOLD_EXACT", ""), 0.95)
    )
    # High confidence - deliverable, minor corrections OK
    address_threshold_deliverable: float = field(
        default_factory=lambda: _parse_float(getenv("ADDRESS_THRESHOLD_DELIVERABLE", ""), 0.85)
    )
    # Medium confidence - valid but needs review
    address_threshold_valid: float = field(
        default_factory=lambda: _parse_float(getenv("ADDRESS_THRESHOLD_VALID", ""), 0.70)
    )
    # Minimum to return any result
    address_threshold_minimum: float = field(
        default_factory=lambda: _parse_float(getenv("ADDRESS_THRESHOLD_MINIMUM", ""), 0.50)
    )

    # ==================== Address Completion Pipeline ====================
    # Vector confidence threshold - skip Nova reranking if vector score >= this
    completion_vector_threshold: float = field(
        default_factory=lambda: _parse_float(getenv("COMPLETION_VECTOR_THRESHOLD", ""), 0.85)
    )
    # Minimum confidence to return a completion result
    completion_min_confidence: float = field(
        default_factory=lambda: _parse_float(getenv("COMPLETION_MIN_CONFIDENCE", ""), 0.5)
    )
    # Enable Redis vector search (requires Redis Stack with RediSearch module)
    enable_redis_vector: bool = field(
        default_factory=lambda: _parse_bool(getenv("ENABLE_REDIS_VECTOR", ""), True)
    )

    # ==================== Circuit Breaker Settings ====================
    # Number of failures before circuit opens
    circuit_breaker_threshold: int = field(
        default_factory=lambda: _parse_int(getenv("CIRCUIT_BREAKER_THRESHOLD", ""), 3)
    )
    # Seconds before attempting to close circuit
    circuit_breaker_reset_timeout: int = field(
        default_factory=lambda: _parse_int(getenv("CIRCUIT_BREAKER_RESET_TIMEOUT", ""), 300)
    )
    # Max retries for transient failures
    max_retries: int = field(
        default_factory=lambda: _parse_int(getenv("MAX_RETRIES", ""), 2)
    )
    # Retry backoff base (seconds)
    retry_backoff_base: float = field(
        default_factory=lambda: _parse_float(getenv("RETRY_BACKOFF_BASE", ""), 0.5)
    )

    # ==================== Input Validation Limits ====================
    max_address_length: int = field(
        default_factory=lambda: _parse_int(getenv("MAX_ADDRESS_LENGTH", ""), 500)
    )
    max_typo_corrections: int = field(
        default_factory=lambda: _parse_int(getenv("MAX_TYPO_CORRECTIONS", ""), 5)
    )

    # ==================== Feature Flags ====================
    enable_federation: bool = field(
        default_factory=lambda: _parse_bool(getenv("ENABLE_FEDERATION", ""), True)
    )
    enable_llm_enforcer: bool = field(
        default_factory=lambda: _parse_bool(getenv("ENABLE_LLM_ENFORCER", ""), True)
    )
    enable_code_index: bool = field(
        default_factory=lambda: _parse_bool(getenv("ENABLE_CODE_INDEX", ""), False)
    )
    admin_enabled: bool = field(
        default_factory=lambda: _parse_bool(getenv("ADMIN_ENABLED", ""), True)
    )
    enable_agentcore_memory: bool = field(
        default_factory=lambda: _parse_bool(getenv("ENABLE_AGENTCORE_MEMORY", ""), True)
    )

    def get_index_config(self) -> dict[str, str]:
        """Get index name configuration for IndexFederation."""
        return {
            "master": self.index_master,
            "code": self.index_code,
            "knowledge": self.index_knowledge,
            "customers": self.index_customers,
        }

    def is_enforcer_available(self) -> bool:
        """Check if LLM enforcer can be enabled (any provider)."""
        if not self.enable_llm_enforcer:
            return False
        # Check if any provider API key is available
        return bool(
            self.gemini_api_key or
            self.openai_api_key or
            self.anthropic_api_key or
            self.openrouter_api_key
        )


cfg = Config()
