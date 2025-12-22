"""
Gemini Package - DEPRECATED.

This package is deprecated. Use icda.llm instead for provider-agnostic LLM support.

All exports are re-exported from icda.llm for backward compatibility.

Migration:
    # Old (deprecated)
    from icda.gemini import GeminiEnforcer, GeminiClient

    # New (recommended)
    from icda.llm import LLMEnforcer, create_llm_client
    from icda.llm.providers import GeminiClient
"""

import warnings

# Re-export from new location for backward compatibility
from icda.llm import (
    LLMEnforcer as GeminiEnforcer,  # Alias for backward compat
    LLMConfig as GeminiConfig,       # Alias for backward compat
    ChunkQualityScore,
    ChunkGateResult,
    IndexHealthReport,
    QueryReviewResult,
    EnforcerMetrics,
)
from icda.llm.providers import GeminiClient

# Emit deprecation warning on import
warnings.warn(
    "icda.gemini is deprecated. Use icda.llm instead for provider-agnostic support.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Client (backward compat)
    "GeminiClient",
    "GeminiConfig",
    # Main enforcer (backward compat alias)
    "GeminiEnforcer",
    # Models
    "ChunkQualityScore",
    "ChunkGateResult",
    "IndexHealthReport",
    "QueryReviewResult",
    "EnforcerMetrics",
]
