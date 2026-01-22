"""Enforcer Agent - Quality gates and response validation with LLM integration.

This agent:
1. Validates response quality using 7 quality gates
2. Applies guardrails (PII filtering)
3. Checks response relevance
4. Ensures completeness
5. Uses secondary LLM for enhanced AI-powered validation (when available)
6. Returns final approved/modified response
"""

import logging
import re
from typing import Any

from .models import (
    IntentResult,
    ParsedQuery,
    NovaResponse,
    EnforcedResponse,
    QualityGate,
    QualityGateResult,
    ResponseStatus,
)

# Import LLM enforcer (optional) - supports any provider
try:
    from icda.llm import LLMEnforcer
    LLM_ENFORCER_AVAILABLE = True
except ImportError:
    LLM_ENFORCER_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnforcerAgent:
    """Validates and enforces response quality with secondary LLM integration.

    Follows the enforcer pattern with 10 quality gates:
    1. RESPONSIVE - Response addresses the query
    2. FACTUAL - Response matches tool results
    3. PII_SAFE - No leaked sensitive data
    4. COMPLETE - All requested info included
    5. COHERENT - Response is well-formed
    6. ON_TOPIC - No off-topic content
    7. CONFIDENCE_MET - Above threshold
    8. FILTER_MATCH - Results match requested filters
    9. PR_ADDRESS_QUALITY - Puerto Rico addresses have urbanization
    10. DOMAIN_RELEVANT - Response is within customer data domain

    When LLM enforcer is available (Gemini, OpenAI, Claude, etc.),
    adds AI-powered validation for hallucination detection.
    """
    __slots__ = ("_guardrails", "_llm_enforcer", "_available")

    # PII patterns for detection and redaction
    PII_PATTERNS = {
        "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "bank_account": r"\b\d{8,17}\b",
        "phone": r"\b(?:\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    }

    # Unsafe content patterns
    UNSAFE_PATTERNS = [
        r"social\s+security",
        r"ssn",
        r"credit\s+card",
        r"bank\s+account",
        r"routing\s+number",
        r"password",
        r"pin\s+number",
    ]

    def __init__(self, guardrails=None, llm_enforcer: "LLMEnforcer | None" = None):
        """Initialize EnforcerAgent with optional LLM validation.

        Args:
            guardrails: Optional Guardrails module for PII filtering.
            llm_enforcer: Optional LLMEnforcer for AI-powered validation.
        """
        self._guardrails = guardrails
        self._llm_enforcer = llm_enforcer
        self._available = True

        if self._llm_enforcer and self._llm_enforcer.available:
            provider = self._llm_enforcer.client.provider if self._llm_enforcer.client else "unknown"
            logger.info(f"EnforcerAgent: LLM validation enabled ({provider})")
        else:
            logger.info("EnforcerAgent: Using rule-based validation only")

    @property
    def available(self) -> bool:
        """Check if agent is available."""
        return self._available

    async def enforce(
        self,
        nova_response: NovaResponse,
        query: str,
        intent: IntentResult,
        parsed: ParsedQuery,
    ) -> EnforcedResponse:
        """Enforce quality gates on the response.

        Args:
            nova_response: Response from NovaAgent.
            intent: Intent classification.
            parsed: Parsed query.

        Returns:
            EnforcedResponse with validation results.
        """
        original = nova_response.response_text
        modified = original
        modifications = []
        gates_passed = []
        gates_failed = []

        # Gate 1: Responsive - does response address the query?
        responsive_result = self._check_responsive(modified, query, intent)
        if responsive_result.passed:
            gates_passed.append(responsive_result)
        else:
            gates_failed.append(responsive_result)

        # Gate 2: Factual - is response consistent with tool results?
        factual_result = self._check_factual(modified, nova_response.tool_results)
        if factual_result.passed:
            gates_passed.append(factual_result)
        else:
            gates_failed.append(factual_result)

        # Gate 3: PII Safe - no sensitive data leaked?
        pii_result, modified, pii_mods = self._check_pii_safe(modified)
        modifications.extend(pii_mods)
        if pii_result.passed:
            gates_passed.append(pii_result)
        else:
            gates_failed.append(pii_result)

        # Gate 4: Complete - all requested info included?
        complete_result = self._check_complete(modified, parsed, nova_response)
        if complete_result.passed:
            gates_passed.append(complete_result)
        else:
            gates_failed.append(complete_result)

        # Gate 5: Coherent - is response well-formed?
        coherent_result = self._check_coherent(modified)
        if coherent_result.passed:
            gates_passed.append(coherent_result)
        else:
            gates_failed.append(coherent_result)

        # Gate 6: On Topic - no off-topic content?
        on_topic_result = self._check_on_topic(modified, intent)
        if on_topic_result.passed:
            gates_passed.append(on_topic_result)
        else:
            gates_failed.append(on_topic_result)

        # Gate 7: Confidence Met - above threshold?
        confidence_result = self._check_confidence(nova_response.ai_confidence)
        if confidence_result.passed:
            gates_passed.append(confidence_result)
        else:
            gates_failed.append(confidence_result)

        # Gate 8: Filter Match - do results match requested filters (state, city)?
        filter_result = self._check_filter_match(parsed, nova_response.tool_results, modified)
        if filter_result.passed:
            gates_passed.append(filter_result)
        else:
            gates_failed.append(filter_result)

        # Gate 9: PR Address Quality - validate Puerto Rico addresses have urbanization
        pr_address_result, pr_penalty = self._check_pr_address_quality(
            modified, nova_response.tool_results
        )
        if pr_address_result.passed:
            gates_passed.append(pr_address_result)
        else:
            gates_failed.append(pr_address_result)

        # Gate 10: Domain Relevant - is response within customer data domain?
        domain_result = self._check_domain_relevant(modified, intent, parsed)
        if domain_result.passed:
            gates_passed.append(domain_result)
        else:
            gates_failed.append(domain_result)

        # Optional: LLM AI-powered validation for hallucination detection
        llm_quality_boost = 0.0
        if self._llm_enforcer and self._llm_enforcer.available:
            try:
                # Convert tool_results to chunks format for LLM review
                chunks = []
                for result in nova_response.tool_results:
                    if isinstance(result, dict):
                        for customer in result.get("data", result.get("results", [])):
                            if isinstance(customer, dict):
                                chunks.append({
                                    "chunk_id": customer.get("crid", "unknown"),
                                    "text": f"{customer.get('name', 'N/A')} - {customer.get('city', 'N/A')}, {customer.get('state', 'N/A')}",
                                })

                # Review query with LLM (non-blocking, uses sample rate)
                review = await self._llm_enforcer.review_query(
                    query_id=f"enf_{id(nova_response)}",
                    query_text=query,
                    retrieved_chunks=chunks,
                    response_text=modified,
                    force=False,  # Respect sample rate
                )

                if review:
                    # Apply LLM validation results
                    if review.hallucination_detected:
                        gates_failed.append(QualityGateResult(
                            gate=QualityGate.FACTUAL,
                            passed=False,
                            message=f"LLM detected hallucination: {review.hallucination_details}",
                            details={"llm_review": True},
                        ))
                        modifications.append("LLM enforcer flagged potential hallucination")
                    else:
                        # Boost quality score if LLM validates
                        llm_quality_boost = review.overall_quality * 0.1
                        logger.debug(f"LLM validation passed: {review.overall_quality:.2f}")

            except Exception as e:
                logger.warning(f"LLM validation failed: {e}")
                # Continue without LLM - graceful degradation

        # Calculate quality score (with optional LLM boost and PR penalty)
        total_gates = len(gates_passed) + len(gates_failed)
        quality_score = len(gates_passed) / total_gates if total_gates > 0 else 0.0
        quality_score = min(1.0, quality_score + llm_quality_boost)
        # Apply PR address penalty (caps penalty at 0.5 to avoid zeroing out score)
        quality_score = max(0.0, quality_score - min(pr_penalty, 0.5))

        # Determine status
        status = self._determine_status(
            gates_passed, gates_failed, modifications, nova_response
        )

        return EnforcedResponse(
            final_response=modified,
            original_response=original,
            quality_score=quality_score,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            modifications=modifications,
            status=status,
        )

    def _check_responsive(
        self,
        response: str,
        query: str,
        intent: IntentResult,
    ) -> QualityGateResult:
        """Check if response addresses the query.

        Args:
            response: Response text.
            query: Original query.
            intent: Intent classification.

        Returns:
            QualityGateResult.
        """
        response_lower = response.lower()
        query_lower = query.lower()

        # Check for refusal phrases that are legitimate
        legitimate_refusals = [
            "no customers found",
            "no results",
            "could not find",
            "0 matches",
        ]
        if any(phrase in response_lower for phrase in legitimate_refusals):
            return QualityGateResult(
                gate=QualityGate.RESPONSIVE,
                passed=True,
                message="Response appropriately indicates no results",
            )

        # Check for generic refusals
        refusal_phrases = [
            "i cannot help",
            "i'm not able",
            "i don't have access",
            "unable to assist",
        ]
        if any(phrase in response_lower for phrase in refusal_phrases):
            return QualityGateResult(
                gate=QualityGate.RESPONSIVE,
                passed=False,
                message="Response appears to refuse without attempting",
            )

        # Check for query keywords in response
        query_words = set(query_lower.split()) - {"the", "a", "an", "is", "are", "in", "to", "for"}
        response_words = set(response_lower.split())
        overlap = query_words & response_words

        if len(overlap) >= min(2, len(query_words)):
            return QualityGateResult(
                gate=QualityGate.RESPONSIVE,
                passed=True,
                message="Response addresses query content",
            )

        # Check if response contains customer data (relevant for most queries)
        if re.search(r"CRID[-\s]?\d+", response, re.IGNORECASE):
            return QualityGateResult(
                gate=QualityGate.RESPONSIVE,
                passed=True,
                message="Response contains relevant customer data",
            )

        return QualityGateResult(
            gate=QualityGate.RESPONSIVE,
            passed=True,  # Default to pass if no clear issues
            message="Response appears to address query",
        )

    def _check_factual(
        self,
        response: str,
        tool_results: list[dict[str, Any]],
    ) -> QualityGateResult:
        """Check if response is consistent with tool results.

        Args:
            response: Response text.
            tool_results: Results from tool calls.

        Returns:
            QualityGateResult.
        """
        if not tool_results:
            return QualityGateResult(
                gate=QualityGate.FACTUAL,
                passed=True,
                message="No tool results to validate against",
            )

        # Extract CRIDs from response
        response_crids = set(re.findall(r"CRID[-\s]?\d+", response, re.IGNORECASE))

        # Extract CRIDs from tool results
        tool_crids = set()
        for result in tool_results:
            if isinstance(result, dict):
                # Check results list
                for item in result.get("results", []):
                    if isinstance(item, dict) and "crid" in item:
                        tool_crids.add(item["crid"])
                # Check single customer
                if "customer" in result and isinstance(result["customer"], dict):
                    if "crid" in result["customer"]:
                        tool_crids.add(result["customer"]["crid"])

        # If response mentions CRIDs, verify they're from results
        if response_crids and tool_crids:
            # Normalize for comparison
            response_crids_norm = {c.upper().replace(" ", "-") for c in response_crids}
            tool_crids_norm = {c.upper() for c in tool_crids}

            if not response_crids_norm.issubset(tool_crids_norm):
                extra = response_crids_norm - tool_crids_norm
                return QualityGateResult(
                    gate=QualityGate.FACTUAL,
                    passed=False,
                    message=f"Response mentions CRIDs not in results: {extra}",
                )

        return QualityGateResult(
            gate=QualityGate.FACTUAL,
            passed=True,
            message="Response consistent with tool results",
        )

    def _check_pii_safe(
        self,
        response: str,
    ) -> tuple[QualityGateResult, str, list[str]]:
        """Check and redact PII from response.

        Args:
            response: Response text.

        Returns:
            Tuple of (result, modified_response, modifications).
        """
        modified = response
        modifications = []
        found_pii = []

        # Check each PII pattern
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, modified)
            if matches:
                found_pii.append(pii_type)
                # Redact
                modified = re.sub(pattern, f"[REDACTED-{pii_type.upper()}]", modified)
                modifications.append(f"Redacted {len(matches)} {pii_type} pattern(s)")

        # Check for unsafe content mentions
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, modified, re.IGNORECASE):
                found_pii.append(f"mention:{pattern}")

        if found_pii:
            return (
                QualityGateResult(
                    gate=QualityGate.PII_SAFE,
                    passed=False,
                    message=f"PII detected and redacted: {found_pii}",
                    details={"pii_types": found_pii},
                ),
                modified,
                modifications,
            )

        return (
            QualityGateResult(
                gate=QualityGate.PII_SAFE,
                passed=True,
                message="No PII detected",
            ),
            modified,
            modifications,
        )

    def _check_complete(
        self,
        response: str,
        parsed: ParsedQuery,
        nova_response: NovaResponse,
    ) -> QualityGateResult:
        """Check if response includes all requested information.

        Args:
            response: Response text.
            parsed: Parsed query.
            nova_response: Full Nova response.

        Returns:
            QualityGateResult.
        """
        # If query asked for specific count and we have results
        if parsed.limit and nova_response.tool_results:
            # Check if we mentioned a count
            count_pattern = r"\b(\d+)\s*(?:customer|result|match|record)"
            if re.search(count_pattern, response, re.IGNORECASE):
                return QualityGateResult(
                    gate=QualityGate.COMPLETE,
                    passed=True,
                    message="Response includes result count",
                )

        # Basic completeness - response has substance
        if len(response.split()) < 5:
            return QualityGateResult(
                gate=QualityGate.COMPLETE,
                passed=False,
                message="Response seems incomplete (too short)",
            )

        return QualityGateResult(
            gate=QualityGate.COMPLETE,
            passed=True,
            message="Response appears complete",
        )

    def _check_coherent(self, response: str) -> QualityGateResult:
        """Check if response is well-formed.

        Args:
            response: Response text.

        Returns:
            QualityGateResult.
        """
        # Check for garbled text
        if re.search(r"[^\x00-\x7F]{5,}", response):
            return QualityGateResult(
                gate=QualityGate.COHERENT,
                passed=False,
                message="Response contains garbled text",
            )

        # Check for incomplete sentences (ending in...)
        if response.rstrip().endswith("...") and len(response) < 50:
            return QualityGateResult(
                gate=QualityGate.COHERENT,
                passed=False,
                message="Response appears truncated",
            )

        # Check for repeated content
        sentences = response.split(". ")
        if len(sentences) > 2:
            unique = set(s.strip().lower() for s in sentences)
            if len(unique) < len(sentences) * 0.7:
                return QualityGateResult(
                    gate=QualityGate.COHERENT,
                    passed=False,
                    message="Response contains repeated content",
                )

        return QualityGateResult(
            gate=QualityGate.COHERENT,
            passed=True,
            message="Response is well-formed",
        )

    def _check_on_topic(
        self,
        response: str,
        intent: IntentResult,
    ) -> QualityGateResult:
        """Check if response stays on topic.

        Args:
            response: Response text.
            intent: Intent classification.

        Returns:
            QualityGateResult.
        """
        response_lower = response.lower()

        # Check for off-topic indicators (expanded list)
        off_topic_phrases = [
            # AI self-awareness
            "i'm an ai",
            "as an ai",
            "i was created",
            "my training",
            "i don't have personal",
            "as a language model",
            "my knowledge cutoff",
            # General topics
            "weather",
            "recipe",
            "joke",
            "stock market",
            "stock price",
            "cryptocurrency",
            "bitcoin",
            "sports score",
            "election",
            "president of",
            "capital of",
            "who won the",
            "movie review",
            "music recommendation",
        ]

        for phrase in off_topic_phrases:
            if phrase in response_lower:
                return QualityGateResult(
                    gate=QualityGate.ON_TOPIC,
                    passed=False,
                    message=f"Response contains off-topic content: '{phrase}'",
                )

        return QualityGateResult(
            gate=QualityGate.ON_TOPIC,
            passed=True,
            message="Response stays on topic",
        )

    def _check_confidence(self, ai_confidence: float) -> QualityGateResult:
        """Check if AI confidence meets threshold.

        Args:
            ai_confidence: Confidence score from NovaAgent.

        Returns:
            QualityGateResult.
        """
        threshold = 0.4

        if ai_confidence >= threshold:
            return QualityGateResult(
                gate=QualityGate.CONFIDENCE_MET,
                passed=True,
                message=f"AI confidence {ai_confidence:.2f} >= {threshold}",
            )

        return QualityGateResult(
            gate=QualityGate.CONFIDENCE_MET,
            passed=False,
            message=f"AI confidence {ai_confidence:.2f} < {threshold}",
        )

    def _check_domain_relevant(
        self,
        response: str,
        intent: IntentResult,
        parsed: ParsedQuery,
    ) -> QualityGateResult:
        """Check if response is relevant to customer data domain.

        This gate validates that:
        1. Out-of-scope queries get appropriate redirect responses
        2. In-scope queries don't drift into off-topic content
        3. Responses stay focused on customer data

        Args:
            response: Response text.
            intent: Intent classification.
            parsed: Parsed query with scope assessment.

        Returns:
            QualityGateResult.
        """
        from icda.classifier import QueryIntent

        response_lower = response.lower()
        scope = parsed.query_scope

        # If query was classified as out-of-scope, response should acknowledge it
        if intent.primary_intent == QueryIntent.OUT_OF_SCOPE:
            redirect_indicators = [
                "customer data",
                "specialize in",
                "can help you with",
                "i'm icda",
                "customer data assistant",
                "outside my",
                "area of expertise",
            ]
            if any(ind in response_lower for ind in redirect_indicators):
                return QualityGateResult(
                    gate=QualityGate.DOMAIN_RELEVANT,
                    passed=True,
                    message="Response appropriately handles out-of-scope query",
                )
            # Response didn't redirect - may have hallucinated
            return QualityGateResult(
                gate=QualityGate.DOMAIN_RELEVANT,
                passed=False,
                message="Out-of-scope query not redirected properly",
            )

        # If parser marked as out_of_scope
        if scope.get("assessment") == "out_of_scope":
            if any(ind in response_lower for ind in ["customer data", "specialize"]):
                return QualityGateResult(
                    gate=QualityGate.DOMAIN_RELEVANT,
                    passed=True,
                    message="Response acknowledges scope limitation",
                )

        # For in-scope queries, check response doesn't contain off-topic hallucinations
        hallucination_indicators = [
            "as an ai language model",
            "i cannot provide real-time",
            "my knowledge cutoff",
            "i don't have access to current",
            "i cannot browse the internet",
        ]

        for indicator in hallucination_indicators:
            if indicator in response_lower:
                # Exception: if response also mentions customer data, it's OK
                if "customer" in response_lower or "crid" in response_lower:
                    continue
                return QualityGateResult(
                    gate=QualityGate.DOMAIN_RELEVANT,
                    passed=False,
                    message=f"Response contains off-domain content: '{indicator}'",
                )

        return QualityGateResult(
            gate=QualityGate.DOMAIN_RELEVANT,
            passed=True,
            message="Response is relevant to customer data domain",
        )

    def _check_filter_match(
        self,
        parsed: ParsedQuery,
        tool_results: list[dict[str, Any]],
        response: str,
    ) -> QualityGateResult:
        """Check if results match the requested filters (state, city, etc.).

        This gate catches cases where:
        - User asks for TX but gets AZ results
        - User asks for Miami but gets Las Vegas
        - User asks for filters that weren't applied

        Args:
            parsed: Parsed query with filters.
            tool_results: Results from search.
            response: Response text (to check for admissions of filter issues).

        Returns:
            QualityGateResult.
        """
        response_lower = response.lower()
        issues = []

        # Check if response admits it couldn't apply filters
        filter_failure_phrases = [
            "cannot filter",
            "no information about",
            "don't have data for",
            "does not provide",
            "however, the context does not",
            "cannot filter based on",
        ]
        for phrase in filter_failure_phrases:
            if phrase in response_lower:
                issues.append(f"Response admits filter limitation: '{phrase}'")

        # Extract requested state from parsed query
        requested_state = parsed.filters.get("state", "").upper() if parsed.filters else ""

        # Extract states from tool results
        result_states = set()
        for result in tool_results:
            if isinstance(result, dict):
                # Check data list (common format)
                for customer in result.get("data", result.get("results", [])):
                    if isinstance(customer, dict) and customer.get("state"):
                        result_states.add(customer["state"].upper())

        # State mismatch detection
        if requested_state and result_states:
            if requested_state not in result_states:
                # Results don't contain requested state at all
                issues.append(
                    f"State mismatch: requested {requested_state}, "
                    f"got {', '.join(sorted(result_states))}"
                )
            elif len(result_states) > 1:
                # Multiple states in results when one was requested
                other_states = result_states - {requested_state}
                if other_states:
                    issues.append(
                        f"Results include unexpected states: {', '.join(sorted(other_states))}"
                    )

        # Check for city mismatch
        requested_city = parsed.filters.get("city", "").lower() if parsed.filters else ""
        if requested_city:
            result_cities = set()
            for result in tool_results:
                if isinstance(result, dict):
                    for customer in result.get("data", result.get("results", [])):
                        if isinstance(customer, dict) and customer.get("city"):
                            result_cities.add(customer["city"].lower())

            if result_cities and requested_city not in result_cities:
                issues.append(
                    f"City mismatch: requested {requested_city}, "
                    f"got {', '.join(sorted(result_cities)[:3])}"
                )

        # Determine pass/fail
        if issues:
            return QualityGateResult(
                gate=QualityGate.FILTER_MATCH,
                passed=False,
                message="; ".join(issues),
                details={
                    "requested_state": requested_state,
                    "result_states": list(result_states),
                    "issues": issues,
                },
            )

        return QualityGateResult(
            gate=QualityGate.FILTER_MATCH,
            passed=True,
            message="Results match requested filters",
        )

    def _check_pr_address_quality(
        self,
        response: str,
        tool_results: list[dict[str, Any]],
        knowledge_chunks: list[dict[str, Any]] | None = None,
    ) -> tuple[QualityGateResult, float]:
        """Check Puerto Rico address quality including urbanization validation.

        Returns:
            Tuple of (QualityGateResult, confidence_penalty).
        """
        # Extract PR addresses from tool results
        pr_addresses = []
        for result in tool_results:
            if isinstance(result, dict):
                # Check for is_puerto_rico flag or ZIP 006-009
                # Handle both flat results and nested data/results structures
                items = result.get("data", result.get("results", []))
                if not items and "zip_code" in result:
                    # Single result format
                    items = [result]

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    zip_code = item.get("zip_code", item.get("zip", ""))
                    if zip_code and len(str(zip_code)) >= 3:
                        prefix = str(zip_code)[:3]
                        if prefix in ("006", "007", "008", "009"):
                            pr_addresses.append(item)

        if not pr_addresses:
            return QualityGateResult(
                gate=QualityGate.PR_ADDRESS_QUALITY,
                passed=True,
                message="No Puerto Rico addresses in results",
            ), 0.0

        # Validate each PR address
        issues = []
        total_penalty = 0.0

        for addr in pr_addresses:
            urbanization = addr.get("urbanization", addr.get("urb"))
            if not urbanization:
                zip_display = addr.get("zip_code", addr.get("zip", "unknown"))
                issues.append(f"PR address missing urbanization: ZIP {zip_display}")
                total_penalty += 0.25

        if issues:
            return QualityGateResult(
                gate=QualityGate.PR_ADDRESS_QUALITY,
                passed=False,
                message=f"PR address issues: {'; '.join(issues)}",
                details={
                    "pr_addresses_found": len(pr_addresses),
                    "issues": issues,
                    "confidence_penalty": total_penalty,
                },
            ), total_penalty

        return QualityGateResult(
            gate=QualityGate.PR_ADDRESS_QUALITY,
            passed=True,
            message=f"All {len(pr_addresses)} PR addresses validated with urbanization",
        ), 0.0

    def _check_domain_relevant(
        self,
        response: str,
        intent: IntentResult,
        parsed: ParsedQuery,
    ) -> QualityGateResult:
        """Check if the response is relevant to the customer data domain.

        This gate catches responses that drift into unrelated topics or
        attempt to answer questions outside the system's scope.

        Args:
            response: Response text.
            intent: Intent classification.
            parsed: Parsed query with scope assessment.

        Returns:
            QualityGateResult.
        """
        from icda.classifier import QueryIntent

        # Check query scope from parser
        scope = parsed.query_scope
        assessment = scope.get("assessment", "in_scope")

        # If query was out of scope, check that response acknowledges it properly
        if assessment == "out_of_scope":
            response_lower = response.lower()
            # Response should redirect user to customer data capabilities
            redirect_phrases = [
                "customer data",
                "can help with",
                "specialize in",
                "customer search",
                "address verification",
                "i'm icda",
            ]
            if any(phrase in response_lower for phrase in redirect_phrases):
                return QualityGateResult(
                    gate=QualityGate.DOMAIN_RELEVANT,
                    passed=True,
                    message="Response appropriately redirects out-of-scope query",
                )
            return QualityGateResult(
                gate=QualityGate.DOMAIN_RELEVANT,
                passed=False,
                message="Response to out-of-scope query doesn't redirect properly",
            )

        # For in-scope queries, check response doesn't contain irrelevant content
        response_lower = response.lower()
        off_domain_indicators = [
            "i cannot provide information about",
            "i don't have access to",
            "my knowledge cutoff",
            "as an ai language model",
            "i'm not able to help with",
            "that's outside my capabilities",
        ]

        for phrase in off_domain_indicators:
            if phrase in response_lower:
                # Check if it's a legitimate limitation message about customer data
                if "customer" in response_lower or "data" in response_lower:
                    return QualityGateResult(
                        gate=QualityGate.DOMAIN_RELEVANT,
                        passed=True,
                        message="Response notes limitation while staying on domain",
                    )
                return QualityGateResult(
                    gate=QualityGate.DOMAIN_RELEVANT,
                    passed=False,
                    message=f"Response contains off-domain indicator: '{phrase}'",
                )

        return QualityGateResult(
            gate=QualityGate.DOMAIN_RELEVANT,
            passed=True,
            message="Response is relevant to customer data domain",
        )

    def _determine_status(
        self,
        gates_passed: list[QualityGateResult],
        gates_failed: list[QualityGateResult],
        modifications: list[str],
        nova_response: NovaResponse,
    ) -> ResponseStatus:
        """Determine final response status.

        Args:
            gates_passed: Passed quality gates.
            gates_failed: Failed quality gates.
            modifications: Modifications made.
            nova_response: Original Nova response.

        Returns:
            ResponseStatus.
        """
        # Check for fallback model
        if nova_response.model_used == "fallback":
            return ResponseStatus.FALLBACK

        # All gates passed
        if not gates_failed:
            if modifications:
                return ResponseStatus.MODIFIED
            return ResponseStatus.APPROVED

        # Critical gates failed (FILTER_MATCH added - wrong state = critical)
        critical_gates = {QualityGate.PII_SAFE, QualityGate.RESPONSIVE, QualityGate.FACTUAL, QualityGate.FILTER_MATCH}
        critical_failed = [g for g in gates_failed if g.gate in critical_gates]

        if critical_failed:
            # PII was redacted - modified but ok
            if any(g.gate == QualityGate.PII_SAFE for g in critical_failed) and modifications:
                return ResponseStatus.MODIFIED
            return ResponseStatus.REJECTED

        # Non-critical failures - modified
        if modifications:
            return ResponseStatus.MODIFIED

        return ResponseStatus.APPROVED

    def should_escalate(self, enforced: EnforcedResponse) -> bool:
        """Check if quality warrants strategy escalation.

        Used by the wrong answer tracker to determine if a query
        should be retried with different search strategies.

        Args:
            enforced: The enforced response to evaluate.

        Returns:
            True if escalation is warranted.
        """
        # Low quality score indicates poor response
        if enforced.quality_score < 0.5:
            return True

        # Critical gate failures warrant escalation (FILTER_MATCH = wrong state/city)
        critical_gates = {QualityGate.FACTUAL, QualityGate.RESPONSIVE, QualityGate.FILTER_MATCH}
        for gate_result in enforced.gates_failed:
            if gate_result.gate in critical_gates:
                return True

        # Rejection status always warrants escalation
        if enforced.status == ResponseStatus.REJECTED:
            return True

        return False
