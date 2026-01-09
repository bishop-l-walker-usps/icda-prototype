"""Ingestion system configuration.

Comprehensive configuration for all ingestion components including
adapters, embedding providers, and enforcer pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IngestionMode(str, Enum):
    """Ingestion operation modes."""

    BATCH = "batch"        # Daily NCOA batch processing
    REALTIME = "realtime"  # REST webhook real-time updates
    HYBRID = "hybrid"      # Both modes active


class ProviderType(str, Enum):
    """Supported embedding providers."""

    PRECOMPUTED = "precomputed"  # Pass-through for pre-computed vectors
    TITAN = "titan"              # AWS Titan Embeddings
    OPENAI = "openai"            # OpenAI text-embedding models
    COHERE = "cohere"            # Cohere embed models
    VOYAGE = "voyage"            # Voyage AI embeddings
    GOOGLE = "google"            # Google text-embedding
    MISTRAL = "mistral"          # Mistral embeddings
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # Local models


@dataclass(slots=True)
class AdapterConfig:
    """Configuration for data source adapters."""

    # NCOA Batch Adapter (primary for C library)
    ncoa_input_path: str = ""
    ncoa_embedding_path: str | None = None
    ncoa_embedding_dim: int = 1024
    ncoa_checkpoint_path: str | None = None
    ncoa_file_format: str = "json"  # json, csv, binary

    # REST Webhook Adapter
    webhook_enabled: bool = False
    webhook_path: str = "/api/address/webhook"
    webhook_api_key: str = ""
    webhook_rate_limit: int = 1000  # per minute

    # File Watcher Adapter
    file_watcher_enabled: bool = False
    file_watcher_path: str = "./data/address-drops"
    file_watcher_patterns: list[str] = field(
        default_factory=lambda: ["*.csv", "*.json"]
    )
    file_watcher_poll_interval: int = 60  # seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ncoa_input_path": self.ncoa_input_path,
            "ncoa_embedding_path": self.ncoa_embedding_path,
            "ncoa_embedding_dim": self.ncoa_embedding_dim,
            "ncoa_checkpoint_path": self.ncoa_checkpoint_path,
            "ncoa_file_format": self.ncoa_file_format,
            "webhook_enabled": self.webhook_enabled,
            "webhook_path": self.webhook_path,
            "webhook_rate_limit": self.webhook_rate_limit,
            "file_watcher_enabled": self.file_watcher_enabled,
            "file_watcher_path": self.file_watcher_path,
            "file_watcher_patterns": self.file_watcher_patterns,
            "file_watcher_poll_interval": self.file_watcher_poll_interval,
        }


@dataclass(slots=True)
class EmbeddingProviderConfig:
    """Configuration for embedding providers."""

    # Primary provider (precomputed for NCOA with C library embeddings)
    primary_provider: str = "precomputed"

    # Fallback chain order when primary unavailable or no precomputed
    fallback_order: list[str] = field(
        default_factory=lambda: [
            "titan", "openai", "cohere", "voyage", "sentence_transformers"
        ]
    )

    # Target dimension for all embeddings (normalize to this)
    target_dimension: int = 1024

    # Enable dimension normalization (PCA/padding)
    enable_normalization: bool = True

    # AWS Titan configuration
    titan_model: str = "amazon.titan-embed-text-v2:0"
    titan_region: str = "us-east-1"

    # OpenAI configuration
    openai_model: str = "text-embedding-3-small"
    openai_api_key: str = ""
    openai_dimensions: int = 1536  # Default for text-embedding-3-small

    # Cohere configuration
    cohere_model: str = "embed-english-v3.0"
    cohere_api_key: str = ""
    cohere_dimensions: int = 1024

    # Voyage configuration
    voyage_model: str = "voyage-2"
    voyage_api_key: str = ""
    voyage_dimensions: int = 1024

    # Google configuration
    google_model: str = "text-embedding-004"
    google_api_key: str = ""
    google_dimensions: int = 768

    # Mistral configuration
    mistral_model: str = "mistral-embed"
    mistral_api_key: str = ""
    mistral_dimensions: int = 1024

    # Sentence Transformers (local) configuration
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    sentence_transformer_device: str = "cpu"  # cpu, cuda, mps
    sentence_transformer_dimensions: int = 384

    # Circuit breaker settings
    circuit_breaker_threshold: int = 5  # failures before open
    circuit_breaker_timeout: int = 300  # seconds before retry

    # Batch settings
    batch_size: int = 100  # embeddings per batch call

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes API keys)."""
        return {
            "primary_provider": self.primary_provider,
            "fallback_order": self.fallback_order,
            "target_dimension": self.target_dimension,
            "enable_normalization": self.enable_normalization,
            "titan_model": self.titan_model,
            "titan_region": self.titan_region,
            "openai_model": self.openai_model,
            "cohere_model": self.cohere_model,
            "voyage_model": self.voyage_model,
            "google_model": self.google_model,
            "mistral_model": self.mistral_model,
            "sentence_transformer_model": self.sentence_transformer_model,
            "sentence_transformer_device": self.sentence_transformer_device,
            "batch_size": self.batch_size,
        }


@dataclass(slots=True)
class EnforcerConfig:
    """Configuration for ingestion enforcer pipeline."""

    # Global enforcer settings
    enabled: bool = True
    fail_fast: bool = False  # Stop at first enforcer failure
    strict_mode: bool = False  # Any gate failure = overall failure

    # Schema enforcer settings
    required_fields: list[str] = field(
        default_factory=lambda: ["street_name", "zip_code"]
    )
    optional_fields: list[str] = field(
        default_factory=lambda: ["street_number", "city", "state", "unit"]
    )

    # Normalization enforcer settings
    require_valid_state: bool = True
    require_valid_zip: bool = True

    # Duplicate enforcer settings
    similarity_threshold: float = 0.95  # Above this = duplicate
    check_existing_index: bool = True
    check_batch_duplicates: bool = True

    # Quality enforcer settings
    min_completeness_score: float = 0.6  # Minimum component completeness
    min_confidence_threshold: float = 0.7  # Minimum parsing confidence
    require_pr_urbanization: bool = True  # Require URB for Puerto Rico

    # Approval enforcer settings
    require_embedding: bool = True
    min_quality_score: float = 0.5  # Minimum to approve
    allow_manual_approval: bool = False  # Queue for manual review

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "fail_fast": self.fail_fast,
            "strict_mode": self.strict_mode,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "similarity_threshold": self.similarity_threshold,
            "min_completeness_score": self.min_completeness_score,
            "min_confidence_threshold": self.min_confidence_threshold,
            "require_pr_urbanization": self.require_pr_urbanization,
            "require_embedding": self.require_embedding,
            "min_quality_score": self.min_quality_score,
        }


@dataclass(slots=True)
class IngestionConfig:
    """Main ingestion system configuration.

    Combines all sub-configurations for adapters, embeddings, and enforcers.
    Can be loaded from environment variables.
    """

    # Operation mode
    mode: IngestionMode = IngestionMode.BATCH

    # Batch processing settings
    batch_size: int = 1000  # Records per batch
    max_concurrent: int = 10  # Concurrent record processing

    # Progress tracking
    emit_progress_events: bool = True
    progress_interval: int = 100  # Emit progress every N records

    # Sub-configurations
    adapters: AdapterConfig = field(default_factory=AdapterConfig)
    embeddings: EmbeddingProviderConfig = field(default_factory=EmbeddingProviderConfig)
    enforcers: EnforcerConfig = field(default_factory=EnforcerConfig)

    # Integration settings
    update_address_index: bool = True
    update_vector_index: bool = True

    # Schema mapper settings
    enable_ai_schema_mapping: bool = True
    schema_cache_path: str = "./data/schema_cache"

    # Logging and debugging
    debug_mode: bool = False
    log_rejected_records: bool = True

    @classmethod
    def from_env(cls) -> IngestionConfig:
        """Load configuration from environment variables.

        Environment variables:
            INGESTION_MODE: batch, realtime, or hybrid
            INGESTION_BATCH_SIZE: Records per batch
            INGESTION_MAX_CONCURRENT: Concurrent processing

            # NCOA Adapter
            NCOA_INPUT_PATH: Path to NCOA data file
            NCOA_EMBEDDING_PATH: Path to pre-computed embeddings
            NCOA_EMBEDDING_DIM: Embedding dimension (default 1024)

            # Embedding Providers
            EMBEDDING_PRIMARY_PROVIDER: Primary provider type
            EMBEDDING_TARGET_DIM: Target embedding dimension
            TITAN_MODEL: AWS Titan model name
            AWS_REGION: AWS region for Titan
            OPENAI_API_KEY: OpenAI API key
            COHERE_API_KEY: Cohere API key
            VOYAGE_API_KEY: Voyage API key
            GOOGLE_API_KEY: Google API key
            MISTRAL_API_KEY: Mistral API key

            # Enforcers
            ENFORCER_ENABLED: Enable enforcer pipeline
            ENFORCER_FAIL_FAST: Stop at first failure
            ENFORCER_MIN_QUALITY: Minimum quality score
        """
        mode_str = os.getenv("INGESTION_MODE", "batch").lower()
        mode = IngestionMode(mode_str) if mode_str in [m.value for m in IngestionMode] else IngestionMode.BATCH

        adapters = AdapterConfig(
            ncoa_input_path=os.getenv("NCOA_INPUT_PATH", ""),
            ncoa_embedding_path=os.getenv("NCOA_EMBEDDING_PATH"),
            ncoa_embedding_dim=int(os.getenv("NCOA_EMBEDDING_DIM", "1024")),
            ncoa_checkpoint_path=os.getenv("NCOA_CHECKPOINT_PATH"),
            ncoa_file_format=os.getenv("NCOA_FILE_FORMAT", "json"),
            webhook_enabled=os.getenv("WEBHOOK_ENABLED", "").lower() == "true",
            webhook_api_key=os.getenv("WEBHOOK_API_KEY", ""),
            file_watcher_enabled=os.getenv("FILE_WATCHER_ENABLED", "").lower() == "true",
            file_watcher_path=os.getenv("FILE_WATCHER_PATH", "./data/address-drops"),
        )

        embeddings = EmbeddingProviderConfig(
            primary_provider=os.getenv("EMBEDDING_PRIMARY_PROVIDER", "precomputed"),
            target_dimension=int(os.getenv("EMBEDDING_TARGET_DIM", "1024")),
            titan_model=os.getenv("TITAN_MODEL", "amazon.titan-embed-text-v2:0"),
            titan_region=os.getenv("AWS_REGION", "us-east-1"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            cohere_api_key=os.getenv("COHERE_API_KEY", ""),
            voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
        )

        enforcers = EnforcerConfig(
            enabled=os.getenv("ENFORCER_ENABLED", "true").lower() == "true",
            fail_fast=os.getenv("ENFORCER_FAIL_FAST", "false").lower() == "true",
            min_quality_score=float(os.getenv("ENFORCER_MIN_QUALITY", "0.5")),
            similarity_threshold=float(os.getenv("ENFORCER_SIMILARITY_THRESHOLD", "0.95")),
        )

        return cls(
            mode=mode,
            batch_size=int(os.getenv("INGESTION_BATCH_SIZE", "1000")),
            max_concurrent=int(os.getenv("INGESTION_MAX_CONCURRENT", "10")),
            emit_progress_events=os.getenv("INGESTION_EMIT_PROGRESS", "true").lower() == "true",
            adapters=adapters,
            embeddings=embeddings,
            enforcers=enforcers,
            update_address_index=os.getenv("UPDATE_ADDRESS_INDEX", "true").lower() == "true",
            update_vector_index=os.getenv("UPDATE_VECTOR_INDEX", "true").lower() == "true",
            enable_ai_schema_mapping=os.getenv("ENABLE_AI_SCHEMA_MAPPING", "true").lower() == "true",
            debug_mode=os.getenv("INGESTION_DEBUG", "false").lower() == "true",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "mode": self.mode.value,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "emit_progress_events": self.emit_progress_events,
            "progress_interval": self.progress_interval,
            "adapters": self.adapters.to_dict(),
            "embeddings": self.embeddings.to_dict(),
            "enforcers": self.enforcers.to_dict(),
            "update_address_index": self.update_address_index,
            "update_vector_index": self.update_vector_index,
            "enable_ai_schema_mapping": self.enable_ai_schema_mapping,
            "debug_mode": self.debug_mode,
        }
