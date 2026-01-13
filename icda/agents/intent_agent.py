"""Intent Agent - Classifies query intent and determines domains.

This agent is the first in the pipeline. It analyzes the user's query to:
1. Classify the primary intent (LOOKUP, SEARCH, STATS, etc.)
2. Detect secondary intents
3. Determine relevant domains (customer, address, knowledge, etc.)
4. Assess query complexity WITH MULTI-CONDITION DETECTION
5. Suggest tools that might be useful

ROUTING VISIBILITY:
- Multi-condition queries (state + status + move_from) = COMPLEX
- Complexity reasons logged for debugging
- Tool suggestions based on detected patterns
"""

import logging
import re
from typing import Any

from icda.classifier import QueryComplexity, QueryIntent

from .models import IntentResult, QueryDomain

logger = logging.getLogger(__name__)


class IntentAgent:
    """Classifies query intent and determines relevant domains.

    Uses pattern matching with fallback to semantic classification if available.
    Follows the enforcer pattern - receives only the context it needs.
    
    ENHANCED: Multi-condition detection for proper routing.
    """
    __slots__ = ("_vector_index", "_available")

    # Pattern definitions for intent detection
    LOOKUP_PATTERNS = (
        r"\bcrid[-\s]?\d+",
        r"\bcustomer\s+id",
        r"\blook\s*up\b",
        r"\bfind\s+customer\b",
        r"\bget\s+customer\b",
        r"\bshow\s+me\s+customer\b",
        r"\bpull\s+up\b",
        r"\bcustomer\s+record\b",
        r"\bcustomer\s+details\b",
    )

    STATS_PATTERNS = (
        r"\bhow\s+many\b",
        r"\bcount\b",
        r"\bstatistics?\b",
        r"\btotals?\b",
        r"\bper\s+state\b",
        r"\bby\s+state\b",
        r"\bbreakdown\b",
        r"\bnumbers?\b",
        r"\bsummary\b",
        r"\baggregate\b",
    )

    SEARCH_PATTERNS = (
        r"\bsearch\b",
        r"\bfind\b",
        r"\bshow\b",
        r"\blist\b",
        r"\bgive\s+me\b",
        r"\bcustomers?\s+in\b",
        r"\bpeople\s+in\b",
        r"\bwho\s+lives?\b",
        r"\bresidents?\b",
        r"\bliving\s+in\b",
        r"\bfrom\b",
        r"\bmoved\b",
        r"\bmovers?\b",
        r"\brelocated\b",
        r"\bhigh\s+movers?\b",
        r"\bfrequent\b",
        # Status-qualified customer queries
        r"\b(?:active|inactive|pending|dormant)\s+customers?\b",
    )

    ANALYSIS_PATTERNS = (
        r"\btrends?\b",
        r"\bpatterns?\b",
        r"\banalyze\b",
        r"\banalysis\b",
        r"\binsights?\b",
        r"\bwhy\b",
        r"\bmigration\b",
        r"\bmovement\b",
        r"\bbehavior\b",
        r"\bdo\s+you\s+see\b",
    )

    COMPARISON_PATTERNS = (
        r"\bcompare\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bdifference\b",
        r"\bbetween\b",
        r"\bcomparison\b",
    )

    RECOMMENDATION_PATTERNS = (
        r"\brecommend\b",
        r"\bsuggest\b",
        r"\bshould\b",
        r"\bpredict\b",
        r"\bforecast\b",
        r"\bwhich\s+customers?\b",
    )

    ADDRESS_PATTERNS = (
        r"\baddress\b",
        r"\bverify\b",
        r"\bvalidate\b",
        r"\bnormalize\b",
        r"\bstreet\b",
        r"\bzip\s*code\b",
        r"\bpostal\b",
    )

    KNOWLEDGE_PATTERNS = (
        r"\bpolicy\b",
        r"\bprocedure\b",
        r"\bdocumentation\b",
        r"\bhow\s+do\s+i\b",
        r"\bwhat\s+is\s+the\s+process\b",
        r"\brules?\b",
        r"\bguidelines?\b",
    )

    # =========================================================================
    # COMPLEXITY INDICATORS - CRITICAL FOR MODEL ROUTING
    # =========================================================================
    
    COMPLEX_INDICATORS = (
        r"\btrends?\b",
        r"\bpatterns?\b",
        r"\banalyze\b",
        r"\banalysis\b",
        r"\brecommend\b",
        r"\bpredict\b",
        r"\binsights?\b",
        r"\bwhy\b",
        r"\bforecast\b",
        r"\bmigration\b",
        r"\bbehavior\b",
        r"\bdo\s+you\s+see\b",
        # CRITICAL: Multi-condition queries requiring move history
        r"moved\s+from\s+\w+",              # "moved from California"
        r"relocated\s+from\s+\w+",          # "relocated from Texas"
        r"came\s+from\s+\w+",               # "came from Nevada"
        r"originally\s+from\s+\w+",         # "originally from New York"
        r"previously\s+in\s+\w+",           # "previously in Florida"
    )

    MEDIUM_INDICATORS = (
        r"\bcompare\b",
        r"\bfilter\b",
        r"\bbetween\b",
        r"\bsummary\b",
        r"\bper\s+state\b",
        r"\bwho\s+moved\b",
        r"\bwhich\b",
        r"\bmultiple\b",
        r"\bseveral\b",
        r"\ball\b",
        r"\bmost\b",
        r"\bleast\b",
        r"\btop\b",
        r"\bbottom\b",
        r"\bhigh\s+movers?\b",
        r"\bfrequent\s+movers?\b",
        r"\bapartment\b",
        r"\brenters?\b",
        r"\binactive\b",
        r"\bactive\b",
        r"\bmoved\s+from\b",
        r"\brelocated\b",
        r"\b\d+\+?\s*moves?\b",
    )

    # Status-related patterns for tool suggestion
    STATUS_TOOL_PATTERNS = (
        r"\binactive\b",
        r"\bactive\b",
        r"\bpending\b",
        r"\bdormant\b",
        r"\bstatus\b",
    )

    # Move origin patterns for tool suggestion
    MOVE_FROM_PATTERNS = (
        r"moved\s+from",
        r"relocated\s+from",
        r"came\s+from",
        r"originally\s+from",
        r"previously\s+in",
    )

    # State patterns for multi-condition detection
    STATE_PATTERNS = (
        r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b",
        r"\b(alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new\s+hampshire|new\s+jersey|new\s+mexico|new\s+york|north\s+carolina|north\s+dakota|ohio|oklahoma|oregon|pennsylvania|rhode\s+island|south\s+carolina|south\s+dakota|tennessee|texas|utah|vermont|virginia|washington|west\s+virginia|wisconsin|wyoming)\b",
    )

    def __init__(self, vector_index=None):
        """Initialize IntentAgent."""
        self._vector_index = vector_index
        self._available = True

    @property
    def available(self) -> bool:
        """Check if agent is available."""
        return self._available

    async def classify(self, query: str, session_id: str | None = None) -> IntentResult:
        """Classify query intent and determine domains."""
        q = query.lower().strip()
        signals: dict[str, Any] = {"patterns_matched": [], "original_query": query}

        # Detect primary intent
        primary_intent, intent_confidence = self._detect_intent(q, signals)

        # Detect secondary intents
        secondary_intents = self._detect_secondary_intents(q, primary_intent)

        # Determine domains
        domains = self._detect_domains(q, primary_intent)

        # Assess complexity WITH MULTI-CONDITION DETECTION
        complexity, complexity_reasons = self._assess_complexity_with_reasons(q)
        signals["complexity_reasons"] = complexity_reasons
        signals["complexity_value"] = complexity.value

        # Suggest tools based on intent, domains, and query patterns
        suggested_tools = self._suggest_tools(primary_intent, domains, complexity, query)
        signals["suggested_tools"] = suggested_tools

        # Calculate overall confidence
        confidence = self._calculate_confidence(q, primary_intent, intent_confidence, signals)

        # Log for debugging
        logger.info(
            f"IntentAgent: query='{query[:50]}' intent={primary_intent.value} "
            f"complexity={complexity.value} reasons={complexity_reasons}"
        )

        return IntentResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            domains=domains,
            complexity=complexity,
            suggested_tools=suggested_tools,
            raw_signals=signals,
        )

    def _detect_intent(self, query: str, signals: dict) -> tuple[QueryIntent, float]:
        """Detect primary query intent using pattern matching."""
        if self._match_patterns(query, self.LOOKUP_PATTERNS):
            signals["patterns_matched"].append("lookup")
            if re.search(r"crid[-\s]?\d+", query):
                return QueryIntent.LOOKUP, 0.95
            return QueryIntent.LOOKUP, 0.8

        if self._match_patterns(query, self.STATS_PATTERNS):
            signals["patterns_matched"].append("stats")
            return QueryIntent.STATS, 0.85

        if self._match_patterns(query, self.COMPARISON_PATTERNS):
            signals["patterns_matched"].append("comparison")
            return QueryIntent.COMPARISON, 0.8

        if self._match_patterns(query, self.ANALYSIS_PATTERNS):
            signals["patterns_matched"].append("analysis")
            return QueryIntent.ANALYSIS, 0.8

        if self._match_patterns(query, self.RECOMMENDATION_PATTERNS):
            signals["patterns_matched"].append("recommendation")
            return QueryIntent.RECOMMENDATION, 0.75

        if self._match_patterns(query, self.SEARCH_PATTERNS):
            signals["patterns_matched"].append("search")
            return QueryIntent.SEARCH, 0.85

        signals["patterns_matched"].append("default_search")
        return QueryIntent.SEARCH, 0.6

    def _detect_secondary_intents(self, query: str, primary: QueryIntent) -> list[QueryIntent]:
        """Detect secondary intents that might also be relevant."""
        secondary = []
        intent_patterns = [
            (QueryIntent.LOOKUP, self.LOOKUP_PATTERNS),
            (QueryIntent.STATS, self.STATS_PATTERNS),
            (QueryIntent.SEARCH, self.SEARCH_PATTERNS),
            (QueryIntent.ANALYSIS, self.ANALYSIS_PATTERNS),
            (QueryIntent.COMPARISON, self.COMPARISON_PATTERNS),
            (QueryIntent.RECOMMENDATION, self.RECOMMENDATION_PATTERNS),
        ]
        for intent, patterns in intent_patterns:
            if intent != primary and self._match_patterns(query, patterns):
                secondary.append(intent)
        return secondary[:2]

    def _detect_domains(self, query: str, primary_intent: QueryIntent) -> list[QueryDomain]:
        """Detect relevant query domains."""
        domains = []
        if primary_intent in (QueryIntent.LOOKUP, QueryIntent.SEARCH, QueryIntent.STATS):
            domains.append(QueryDomain.CUSTOMER)
        if self._match_patterns(query, self.ADDRESS_PATTERNS):
            domains.append(QueryDomain.ADDRESS)
        if self._match_patterns(query, self.KNOWLEDGE_PATTERNS):
            domains.append(QueryDomain.KNOWLEDGE)
        if primary_intent == QueryIntent.STATS or self._match_patterns(query, self.STATS_PATTERNS):
            if QueryDomain.STATS not in domains:
                domains.append(QueryDomain.STATS)
        if not domains:
            domains.append(QueryDomain.CUSTOMER)
        return domains

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity level (simplified)."""
        complexity, _ = self._assess_complexity_with_reasons(query)
        return complexity

    def _assess_complexity_with_reasons(self, query: str) -> tuple[QueryComplexity, list[str]]:
        """Assess query complexity WITH reasoning for visibility.

        CRITICAL: Multi-condition queries (state + status + move_from) = COMPLEX.
        This ensures routing to Nova Pro for better reasoning.

        Args:
            query: Lowercased query.

        Returns:
            Tuple of (QueryComplexity, list of reasons).
        """
        reasons = []

        # =====================================================================
        # MULTI-CONDITION CHECK: 2+ filter conditions = COMPLEX
        # This is THE KEY FIX for queries like:
        # "Texas customers who are inactive and moved from California"
        # =====================================================================
        condition_count = 0
        
        # Check for status filter (active/inactive/pending)
        if self._match_patterns(query, self.STATUS_TOOL_PATTERNS):
            condition_count += 1
            reasons.append("has_status_filter")
        
        # Check for move-from filter
        if self._match_patterns(query, self.MOVE_FROM_PATTERNS):
            condition_count += 1
            reasons.append("has_move_from_filter")
        
        # Check for state filter
        if self._match_patterns(query, self.STATE_PATTERNS):
            condition_count += 1
            reasons.append("has_state_filter")
        
        # 2+ conditions = COMPLEX (multi-part query needs Pro model)
        if condition_count >= 2:
            reasons.append(f"COMPLEX:multi_condition({condition_count})")
            logger.info(f"IntentAgent: COMPLEX due to {condition_count} filter conditions")
            return QueryComplexity.COMPLEX, reasons

        # 1 condition = MEDIUM (single filter requires some reasoning)
        if condition_count == 1:
            reasons.append("MEDIUM:single_filter")
            logger.info(f"IntentAgent: MEDIUM due to single filter condition")
            return QueryComplexity.MEDIUM, reasons

        # Check for complex pattern indicators
        if self._match_patterns(query, self.COMPLEX_INDICATORS):
            reasons.append("COMPLEX:pattern_match")
            return QueryComplexity.COMPLEX, reasons

        # Check for medium indicators
        if self._match_patterns(query, self.MEDIUM_INDICATORS):
            reasons.append("MEDIUM:pattern_match")
            return QueryComplexity.MEDIUM, reasons

        # Word count heuristic
        word_count = len(query.split())
        if word_count > 15:
            reasons.append(f"COMPLEX:long_query({word_count})")
            return QueryComplexity.COMPLEX, reasons
        if word_count > 8:
            reasons.append(f"MEDIUM:medium_query({word_count})")
            return QueryComplexity.MEDIUM, reasons

        reasons.append("SIMPLE:default")
        return QueryComplexity.SIMPLE, reasons

    def _suggest_tools(
        self,
        primary_intent: QueryIntent,
        domains: list[QueryDomain],
        complexity: QueryComplexity,
        query: str = "",
    ) -> list[str]:
        """Suggest tools based on intent, domains, complexity, and query patterns.

        CRITICAL: Specialized tools are suggested FIRST for status/move queries.
        """
        tools = []
        query_lower = query.lower()

        # =====================================================================
        # CRITICAL: Check for specialized patterns FIRST
        # =====================================================================
        has_status = any(re.search(p, query_lower) for p in self.STATUS_TOOL_PATTERNS)
        has_move_from = any(re.search(p, query_lower) for p in self.MOVE_FROM_PATTERNS)

        if has_move_from:
            tools.append("customers_moved_from")
            tools.append("get_move_timeline")
        
        if has_status:
            tools.append("filter_by_status")
            if "inactive" in query_lower:
                tools.append("get_inactive_customers")
            elif "active" in query_lower:
                tools.append("get_active_customers")

        # Intent-based suggestions
        match primary_intent:
            case QueryIntent.LOOKUP:
                tools.append("lookup_crid")
            case QueryIntent.SEARCH:
                tools.append("search_customers")
                if "how many" in query_lower or "count" in query_lower:
                    tools.append("count_by_criteria")
                if complexity != QueryComplexity.SIMPLE:
                    tools.append("fuzzy_search")
                    tools.append("semantic_search")
            case QueryIntent.STATS:
                tools.append("get_stats")
                tools.append("group_by_field")
                tools.append("count_by_criteria")
            case QueryIntent.ANALYSIS | QueryIntent.COMPARISON:
                tools.extend(["get_stats", "group_by_field", "search_customers", "semantic_search"])
            case QueryIntent.RECOMMENDATION:
                tools.extend(["search_customers", "semantic_search", "get_stats", "count_by_criteria"])

        # Domain-based additions
        if QueryDomain.ADDRESS in domains:
            tools.append("verify_address")
        if QueryDomain.KNOWLEDGE in domains:
            tools.append("search_knowledge")

        # Complexity-based additions
        if complexity == QueryComplexity.COMPLEX:
            if "hybrid_search" not in tools:
                tools.append("hybrid_search")
            tools.append("multi_criteria_search")

        return list(dict.fromkeys(tools))

    def _calculate_confidence(
        self,
        query: str,
        primary_intent: QueryIntent,
        intent_confidence: float,
        signals: dict,
    ) -> float:
        """Calculate overall classification confidence."""
        confidence = intent_confidence
        if len(signals.get("patterns_matched", [])) > 1:
            confidence = min(confidence + 0.1, 1.0)
        if len(query.split()) < 3:
            confidence *= 0.9
        if len(query.split()) > 20:
            confidence *= 0.85
        return round(confidence, 3)

    def _match_patterns(self, text: str, patterns: tuple) -> bool:
        """Check if any pattern matches the text."""
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)
