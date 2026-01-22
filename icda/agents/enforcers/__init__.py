"""Memory Enforcer Agents Package.

This package provides specialized enforcer agents to ensure
no functionality reduction when using Bedrock AgentCore memory.

Memory Enforcers:
1. MemoryIntegrityEnforcer - Validates memory read/write consistency
2. SearchContextEnforcer - Ensures memory enhances search operations
3. NovaContextEnforcer - Validates memory improves LLM responses
4. ResponseQualityEnforcer - Final quality validation
5. PRAddressPreservationEnforcer - Puerto Rico address quality gates
6. FunctionalityPreservationEnforcer - Central meta-enforcer

Address Validation Enforcers (in address subpackage):
1. NormalizationEnforcerAgent - Validates address normalization
2. CompletionEnforcerAgent - Validates address completion
3. CorrectionEnforcerAgent - Validates address corrections
4. MatchConfidenceEnforcerAgent - Validates match confidence
5. ValidationReportAgent - Generates validation reports
6. BatchOrchestratorAgent - Coordinates batch processing

The EnforcerCoordinator orchestrates memory enforcers in sequence.
The AddressEnforcerCoordinator orchestrates address enforcers.
"""

from .base_enforcer import BaseEnforcer, EnforcerResult, EnforcerGate, GateResult
from .memory_integrity_enforcer import MemoryIntegrityEnforcer
from .search_context_enforcer import SearchContextEnforcer
from .nova_context_enforcer import NovaContextEnforcer
from .response_quality_enforcer import ResponseQualityEnforcer
from .functionality_preservation_enforcer import FunctionalityPreservationEnforcer
from .pr_address_enforcer import PRAddressPreservationEnforcer, PRAddressMetrics
from .enforcer_coordinator import EnforcerCoordinator

# Address validation enforcers
from .address import (
    AddressEnforcerCoordinator,
    NormalizationEnforcerAgent,
    CompletionEnforcerAgent,
    CorrectionEnforcerAgent,
    MatchConfidenceEnforcerAgent,
    ValidationReportAgent,
    BatchOrchestratorAgent,
    AddressEnforcerGate,
    parse_addresses_from_file,
)

__all__ = [
    # Base
    "BaseEnforcer",
    "EnforcerResult",
    "EnforcerGate",
    "GateResult",
    # Memory Enforcers
    "MemoryIntegrityEnforcer",
    "SearchContextEnforcer",
    "NovaContextEnforcer",
    "ResponseQualityEnforcer",
    "FunctionalityPreservationEnforcer",
    "PRAddressPreservationEnforcer",
    "PRAddressMetrics",
    "EnforcerCoordinator",
    # Address Enforcers
    "AddressEnforcerCoordinator",
    "NormalizationEnforcerAgent",
    "CompletionEnforcerAgent",
    "CorrectionEnforcerAgent",
    "MatchConfidenceEnforcerAgent",
    "ValidationReportAgent",
    "BatchOrchestratorAgent",
    "AddressEnforcerGate",
    "parse_addresses_from_file",
]
