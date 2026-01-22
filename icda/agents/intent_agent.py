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
    # OUT_OF_SCOPE PATTERNS - Questions clearly outside customer data domain
    OUT_OF_SCOPE_PATTERNS = (
        # Weather
        r"\bweather\b",
        r"\bforecast\b.*\b(rain|snow|temperature|sunny|cloudy)\b",
        # Sports
        r"\b(super\s*bowl|world\s*series|nba|nfl|mlb|nhl|soccer|football\s+game)\b",
        r"\bwho\s+won\b.*\b(game|match|championship|tournament)\b",
        r"\bsports?\s+(score|news|update)\b",
        # Politics/News
        r"\belection\b",
        r"\bpresident\s+of\b",
        r"\bpolitics?\b",
        r"\bgovernment\b",
        # Entertainment
        r"\bmovie\s+(review|recommendation)\b",
        r"\bmusic\s+(recommendation|playlist)\b",
        r"\btv\s+show\b",
        r"\bcelebrit(y|ies)\b",
        # General knowledge
        r"\bcapital\s+of\b",
        r"\bwhat\s+year\s+did\b(?!.*\bcustomer\b)",
        r"\bhistory\s+of\b(?!.*\bcustomer\b)",
        r"\bwho\s+invented\b",
        r"\bwhat\s+is\s+the\s+meaning\s+of\b",
        r"\bdefine\b(?!.*\bcustomer\b)",
        # Coding/Technical
        r"\bwrite\s+(me\s+)?(a\s+)?(code|program|script|function)\b",
        r"\bwrite\s+me\s+a?\s*(python|javascript|java)\b",
        r"\bprogram\s+in\s+(python|javascript|java|c\+\+)\b",
        r"\bdebug\s+(this|my)\s+(code|program)\b",
        r"\bhow\s+to\s+code\b",
        r"\b(python|javascript|java)\s+(function|code|script)\b",
        # Food/Recipes
        r"\brecipe\s+for\b",
        r"\bhow\s+(do\s+i\s+|to\s+)(cook|bake|make)\b(?!.*\breport\b)",
        r"\bingredients?\s+for\b",
        # Health/Medical
        r"\bsymptoms?\s+of\b",
        r"\bmedical\s+advice\b",
        r"\btreatment\s+for\b",
        # Financial (non-customer)
        r"\bstock\s+(price|market)\b",
        r"\bcryptocurrency\b",
        r"\bbitcoin\b",
        # Travel (non-customer)
        r"\bflight\s+(to|from)\b",
        r"\bhotel\s+(in|near)\b",
        r"\btourist\s+attractions?\b",
        # Jokes/Fun
        r"\btell\s+me\s+a\s+joke\b",
        r"\bfunny\s+story\b",
    )

    # CONVERSATIONAL PATTERNS - Check these FIRST to avoid RAG for simple chat
    CONVERSATIONAL_PATTERNS = (
        r"^(hi|hello|hey|yo|sup|howdy|greetings)[\s!.,]*$",
        r"^(good\s+)?(morning|afternoon|evening|night)[\s!.,]*$",
        r"^my\s+name\s+is\s+",
        r"^i\s+am\s+\w+$",
        r"^i'm\s+\w+$",
        r"^(thanks|thank\s+you|thx|ty)[\s!.,]*$",
        r"^(yes|no|ok|okay|sure|yep|nope|yeah|nah)[\s!.,]*$",
        r"^(bye|goodbye|see\s+you|later|cya)[\s!.,]*$",
        r"^how\s+are\s+you",
        r"^what('s|\s+is)\s+up",
        r"^who\s+are\s+you",
        r"^what\s+can\s+you\s+do",
        r"^help\s*$",
        r"^\?+$",
    )

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
        # CONVERSATIONAL CHECK FIRST - avoid RAG for simple chat
        if self._match_patterns(query, self.CONVERSATIONAL_PATTERNS):
            signals["patterns_matched"].append("conversational")
            return QueryIntent.CONVERSATIONAL, 0.95

        # OUT_OF_SCOPE CHECK - detect questions outside customer data domain
        if self._match_patterns(query, self.OUT_OF_SCOPE_PATTERNS):
            signals["patterns_matched"].append("out_of_scope")
            logger.info(f"IntentAgent: OUT_OF_SCOPE detected for query: '{query[:50]}...'")
            return QueryIntent.OUT_OF_SCOPE, 0.9

        # Check domain relevance for longer queries without data keywords
        if len(query.split()) > 5 and not self._has_data_keywords(query):
            domain_relevance = self._calculate_domain_relevance(query)
            if domain_relevance < 0.2:
                signals["patterns_matched"].append("out_of_scope_low_relevance")
                signals["domain_relevance"] = domain_relevance
                logger.info(f"IntentAgent: OUT_OF_SCOPE (low relevance={domain_relevance:.2f})")
                return QueryIntent.OUT_OF_SCOPE, 0.75

        # Short queries without data keywords = likely conversational
        if len(query.split()) < 5 and not self._has_data_keywords(query):
            signals["patterns_matched"].append("conversational_short")
            return QueryIntent.CONVERSATIONAL, 0.8

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

    def _has_data_keywords(self, query: str) -> bool:
        """Check if query contains customer/data-related keywords."""
        data_keywords = (
            "customer", "crid", "state", "city", "address", "moved",
            "search", "find", "list", "show", "count", "how many",
            "stats", "nevada", "california", "texas", "active", "inactive"
        )
        q = query.lower()
        return any(kw in q for kw in data_keywords)

    def _match_patterns(self, text: str, patterns: tuple) -> bool:
        """Check if any pattern matches the text."""
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _calculate_domain_relevance(self, query: str) -> float:
        """Calculate how relevant the query is to customer data domain.

        Returns:
            Score from 0.0 (irrelevant) to 1.0 (highly relevant).
        """
        score = 0.0
        q = query.lower()

        # Strong relevance signals (customer data specific)
        strong_signals = [
            "customer", "crid", "address", "zip", "moved", "move count",
            "status", "inactive", "active", "pending", "search", "find",
            "lookup", "verify", "normalize"
        ]
        for signal in strong_signals:
            if signal in q:
                score += 0.3

        # Medium relevance signals (could be customer-related)
        medium_signals = [
            "name", "how many", "count", "list", "show", "people",
            "who", "where", "state", "city"
        ]
        for signal in medium_signals:
            if signal in q:
                score += 0.15

        # State names add strong relevance (likely asking about customers in a state)
        state_names = [
            "alabama", "alaska", "arizona", "arkansas", "california",
            "colorado", "connecticut", "delaware", "florida", "georgia",
            "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
            "kentucky", "louisiana", "maine", "maryland", "massachusetts",
            "michigan", "minnesota", "mississippi", "missouri", "montana",
            "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
            "new york", "north carolina", "north dakota", "ohio", "oklahoma",
            "oregon", "pennsylvania", "rhode island", "south carolina",
            "south dakota", "tennessee", "texas", "utah", "vermont",
            "virginia", "washington", "west virginia", "wisconsin", "wyoming"
        ]
        for state in state_names:
            if state in q:
                score += 0.25
                break

        return min(1.0, score)
