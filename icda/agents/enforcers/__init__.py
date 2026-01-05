"""Memory Enforcer Agents Package.

This package provides 6 specialized enforcer agents to ensure
no functionality reduction when using Bedrock AgentCore memory.

Enforcers:
1. MemoryIntegrityEnforcer - Validates memory read/write consistency
2. SearchContextEnforcer - Ensures memory enhances search operations
3. NovaContextEnforcer - Validates memory improves LLM responses
4. ResponseQualityEnforcer - Final quality validation
5. PRAddressPreservationEnforcer - Puerto Rico address quality gates
6. FunctionalityPreservationEnforcer - Central meta-enforcer

The EnforcerCoordinator orchestrates all enforcers in sequence.
"""

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate, GateResult
from .memory_integrity_enforcer import MemoryIntegrityEnforcer
from .search_context_enforcer import SearchContextEnforcer
from .nova_context_enforcer import NovaContextEnforcer
from .response_quality_enforcer import ResponseQualityEnforcer
from .functionality_preservation_enforcer import FunctionalityPreservationEnforcer
from .pr_address_enforcer import PRAddressPreservationEnforcer, PRAddressMetrics
from .enforcer_coordinator import EnforcerCoordinator

__all__ = [
    "BaseEnforcer",
    "EnforcerResult",
    "EnforcerGate",
    "GateResult",
    "MemoryIntegrityEnforcer",
    "SearchContextEnforcer",
    "NovaContextEnforcer",
    "ResponseQualityEnforcer",
    "FunctionalityPreservationEnforcer",
    "PRAddressPreservationEnforcer",
    "PRAddressMetrics",
    "EnforcerCoordinator",
]
