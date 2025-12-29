"""Memory Enforcer Agents Package.

This package provides 5 specialized enforcer agents to ensure
no functionality reduction when using Bedrock AgentCore memory.

Enforcers:
1. MemoryIntegrityEnforcer - Validates memory read/write consistency
2. SearchContextEnforcer - Ensures memory enhances search operations
3. NovaContextEnforcer - Validates memory improves LLM responses
4. ResponseQualityEnforcer - Final quality validation
5. FunctionalityPreservationEnforcer - Central meta-enforcer

The EnforcerCoordinator orchestrates all enforcers in sequence.
"""

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate
from .memory_integrity_enforcer import MemoryIntegrityEnforcer
from .search_context_enforcer import SearchContextEnforcer
from .nova_context_enforcer import NovaContextEnforcer
from .response_quality_enforcer import ResponseQualityEnforcer
from .functionality_preservation_enforcer import FunctionalityPreservationEnforcer
from .enforcer_coordinator import EnforcerCoordinator

__all__ = [
    "BaseEnforcer",
    "EnforcerResult",
    "EnforcerGate",
    "MemoryIntegrityEnforcer",
    "SearchContextEnforcer",
    "NovaContextEnforcer",
    "ResponseQualityEnforcer",
    "FunctionalityPreservationEnforcer",
    "EnforcerCoordinator",
]
