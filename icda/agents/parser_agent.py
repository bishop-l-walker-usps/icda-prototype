"""Parser Agent - Normalizes queries and extracts entities.

This agent processes the raw query to:
1. Rewrite ambiguous queries for clarity (NEW)
2. Normalize the query text
3. Extract entities (CRIDs, names, locations)
4. Parse filter criteria
5. Extract date ranges
6. Determine result limits and sorting

IMPORTANT: The query rewriter transforms demographic-sounding queries like
"how many people live in virginia?" into explicit customer data queries
like "how many customers live in virginia?" to prevent Nova from
misinterpreting them as census/population questions.
"""

import logging
import re
from difflib import get_close_matches
from typing import Any

from icda.classifier import QueryIntent
from .models import IntentResult, QueryContext, ParsedQuery
from .city_state_validator import CityStateValidator
from .query_rewriter import QueryRewriter, RewriteResult

logger = logging.getLogger(__name__)


class ParserAgent:
    """Normalizes queries and extracts structured information.

    Follows the enforcer pattern - receives only the context it needs.

    ENHANCED: Now uses QueryRewriter to transform ambiguous queries into
    explicit customer data queries before processing.
    """
    __slots__ = ("_db", "_available", "_city_state_validator", "_query_rewriter")

    # Common state misspellings -> correct name
    STATE_TYPOS = {
        # Kansas variations
        "kanas": "kansas", "kanses": "kansas", "kanzas": "kansas", "kanss": "kansas",
        # California variations
        "californa": "california", "californai": "california", "califronia": "california",
        # Florida variations
        "flordia": "florida", "florda": "florida", "floirda": "florida",
        # Texas variations
        "texs": "texas", "texsa": "texas", "teaxs": "texas",
        # Arizona variations
        "arizon": "arizona", "arizonia": "arizona", "arizonza": "arizona",
        "argintina": "arizona",  # Common confusion
        # New York variations
        "newyork": "new york", "new yor": "new york", "newy ork": "new york",
        # Pennsylvania variations
        "pennsylvnia": "pennsylvania", "pensilvania": "pennsylvania", "pensylvania": "pennsylvania",
        # Other common typos
        "gorgia": "georgia", "virgina": "virginia", "michagan": "michigan",
        "colordo": "colorado", "nevda": "nevada", "washingon": "washington",
        "illnois": "illinois", "ohoi": "ohio", "noth carolina": "north carolina",
    }

    # State name to code mapping (full names only - no abbreviations that conflict with words)
    STATE_NAMES = {
        "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
        "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
        "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
        "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
        "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
        "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
        "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
        "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
        "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
        "oregon": "OR", "pennsylvania": "PA", "puerto rico": "PR", "rhode island": "RI",
        "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
        "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
        "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
        "district of columbia": "DC", "d.c.": "DC",
        # Common variations that don't conflict with English words
        "cali": "CA", "vegas": "NV",
    }

    # State codes that are also common English words - require special context
    AMBIGUOUS_STATE_CODES = {"IN", "OR", "ME", "OK", "HI", "OH", "LA", "PA", "MA", "DC"}

    # All valid state codes
    VALID_STATE_CODES = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "PR", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
    }

    # All state names for fuzzy matching
    ALL_STATE_NAMES = [
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada",
        "new hampshire", "new jersey", "new mexico", "new york",
        "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
        "pennsylvania", "puerto rico", "rhode island", "south carolina",
        "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
        "washington", "west virginia", "wisconsin", "wyoming",
        "district of columbia",
    ]

    # Move count interpretations
    MOVE_PATTERNS = {
        r"high\s+movers?": 5,
        r"frequent\s+movers?": 3,
        r"multiple\s+moves?": 2,
        r"moved\s+(\d+)\+?\s+times?": None,  # Extract number
        r"(\d+)\+?\s+moves?": None,  # Extract number
        r"moved\s+(?:at\s+least\s+)?(\d+)": None,  # Extract number
    }

    # Status filter patterns - THE FIX for "inactive customers" queries
    STATUS_PATTERNS = {
        "INACTIVE": [
            r"\binactive\b",
            r"\bdormant\b",
            r"\bcanceled\b",
            r"\bcancelled\b",
            r"\bnot\s+active\b",
            r"\bno\s+longer\s+active\b",
            r"\blapsed\b",
            r"\bsuspended\b",
            r"\bdeactivated\b",
        ],
        "ACTIVE": [
            r"\bactive\b",
            r"\bcurrent\b",
            r"\bstill\s+active\b",
            r"\benabled\b",
        ],
        "PENDING": [
            r"\bpending\b",
            r"\bawaiting\b",
            r"\bunconfirmed\b",
        ],
    }

    # Move origin patterns - THE FIX for "moved from California" queries
    MOVE_ORIGIN_PATTERNS = [
        # "moved from California", "relocated from Texas"
        r"(?:moved|relocated|transferred|came)\s+from\s+([a-zA-Z\s]+?)(?:\s+to\s+|\s*$|,|\s+and\s+|\s+who)",
        # "originally from California"
        r"originally\s+from\s+([a-zA-Z\s]+?)(?:\s+to\s+|\s*$|,)",
        # "previously in California", "formerly in Texas"
        r"(?:previously|formerly)\s+(?:in|from)\s+([a-zA-Z\s]+?)(?:\s+to\s+|\s*$|,)",
        # "who were in California" (past tense implies moved)
        r"who\s+were\s+in\s+([a-zA-Z\s]+?)(?:\s+and\s+|\s*$|,)",
    ]

    def _correct_state_typos(self, text: str, notes: list[str]) -> str:
        """Correct common state name misspellings.

        Args:
            text: Text to check for typos.
            notes: List to append resolution notes.

        Returns:
            Text with typos corrected.
        """
        text_lower = text.lower()
        for typo, correct in self.STATE_TYPOS.items():
            if typo in text_lower:
                # Replace the typo with the correct spelling
                text = re.sub(
                    rf"\b{re.escape(typo)}\b",
                    correct,
                    text,
                    flags=re.IGNORECASE
                )
                notes.append(f"Corrected typo: '{typo}' → '{correct}'")
        return text

    def __init__(self, db=None):
        """Initialize ParserAgent.

        Args:
            db: Optional CustomerDB for validation.
        """
        self._db = db
        self._available = True
        self._city_state_validator = CityStateValidator()
        self._query_rewriter = QueryRewriter()

    @property
    def available(self) -> bool:
        """Check if agent is available."""
        return self._available

    def _fuzzy_match_state(self, word: str, notes: list[str]) -> str | None:
        """Try to fuzzy match a word to a state name.

        Args:
            word: Word to match.
            notes: Resolution notes to append to.

        Returns:
            State code if matched, None otherwise.
        """
        word_lower = word.lower().strip()

        # Skip very short words or common English words
        if len(word_lower) < 4:
            return None

        # Try exact match first
        if word_lower in self.STATE_NAMES:
            return self.STATE_NAMES[word_lower]

        # Try fuzzy matching with high threshold (0.8 = must be very similar)
        matches = get_close_matches(word_lower, self.ALL_STATE_NAMES, n=1, cutoff=0.8)
        if matches:
            matched_name = matches[0]
            code = self.STATE_NAMES[matched_name]
            notes.append(f"Fuzzy matched '{word}' → {matched_name} ({code})")
            return code

        return None

    def _is_likely_state_code_context(self, query: str, code: str, position: int) -> bool:
        """Check if a 2-letter code is likely meant as a state code based on context.

        Args:
            query: The full query.
            code: The 2-letter code found.
            position: Position of the code in the query.

        Returns:
            True if likely a state code, False if likely a common word.
        """
        # Check for patterns that indicate it's a state code:
        # - Preceded by city name pattern: "City, ST" or "City ST"
        # - Followed by ZIP code
        # - Part of address-like patterns

        # Look for city-state patterns (e.g., "New York, NY" or "Denver CO")
        before = query[:position].strip()
        after = query[position + 2:].strip()

        # Check for comma before (strong indicator of city, state)
        if before.endswith(","):
            return True

        # Check for ZIP code after
        if re.match(r"^\s*\d{5}", after):
            return True

        # Check if preceded by a capitalized word (likely city name)
        # Pattern: "CityName ST" where ST is preceded by a word starting with capital
        city_pattern = re.search(r"([A-Z][a-z]+)\s*$", before)
        if city_pattern:
            potential_city = city_pattern.group(1).lower()
            # Make sure it's not a common word
            common_words = {"how", "many", "users", "are", "the", "for", "from", "with"}
            if potential_city not in common_words and potential_city not in self.STATE_NAMES:
                return True

        # For ambiguous codes, be conservative - require clear context
        return False

    async def parse(
        self,
        query: str,
        intent: IntentResult,
        context: QueryContext,
    ) -> ParsedQuery:
        """Parse and normalize the query.

        Args:
            query: Raw user query.
            intent: Classification from IntentAgent.
            context: Context from ContextAgent.

        Returns:
            ParsedQuery with extracted information.
        """
        resolution_notes = []

        # =====================================================================
        # NEW: Rewrite ambiguous queries FIRST
        # This transforms "how many people live in virginia?" into
        # "how many customers live in virginia?" to prevent Nova from
        # misinterpreting as a demographic/census question.
        # =====================================================================
        rewrite_result = self._query_rewriter.rewrite(query)

        if rewrite_result.was_rewritten:
            resolution_notes.append(
                f"Query rewritten: '{query}' → '{rewrite_result.rewritten_query}'"
            )
            resolution_notes.extend(rewrite_result.rewrites_applied)
            logger.info(
                f"ParserAgent: Query rewritten for clarity: "
                f"'{query}' → '{rewrite_result.rewritten_query}'"
            )

        # Use the rewritten query for all subsequent processing
        working_query = rewrite_result.rewritten_query

        # Track detected state from rewriter for consistency
        if rewrite_result.detected_state_code and not context.geographic_context.get("state"):
            resolution_notes.append(
                f"State detected: {rewrite_result.detected_state} ({rewrite_result.detected_state_code})"
            )

        # Track ambiguous city if detected
        if rewrite_result.is_ambiguous_city:
            resolution_notes.append(
                f"Ambiguous city: {rewrite_result.detected_city} "
                f"(could be {', '.join(rewrite_result.ambiguous_city_states or [])})"
            )

        # Normalize query (using rewritten version)
        normalized = self._normalize_query(working_query, resolution_notes)

        # Extract entities (using rewritten version)
        entities = self._extract_entities(working_query, context, resolution_notes)

        # If rewriter detected state, ensure it's in entities
        if rewrite_result.detected_state_code:
            if rewrite_result.detected_state_code not in entities.get("states", []):
                entities.setdefault("states", []).append(rewrite_result.detected_state_code)

        # Extract filters (using rewritten version)
        filters = self._extract_filters(working_query, context, resolution_notes)

        # If rewriter detected state and filters don't have it, add it
        if rewrite_result.detected_state_code and "state" not in filters:
            filters["state"] = rewrite_result.detected_state_code
            resolution_notes.append(
                f"State filter from rewriter: {rewrite_result.detected_state_code}"
            )

        # Extract date range
        date_range = self._extract_date_range(working_query, resolution_notes)

        # Determine limit
        limit = self._extract_limit(working_query, context)

        # Determine sort preference
        sort_preference = self._extract_sort(working_query)

        # Check if follow-up
        is_follow_up = context.is_follow_up

        # Assess query scope (in_scope, out_of_scope, conversational)
        query_scope = self._assess_query_scope(query, intent, entities, filters)
        resolution_notes.append(f"Query scope: {query_scope['assessment']}")

        return ParsedQuery(
            original_query=query,  # Keep original for reference
            normalized_query=normalized,
            entities=entities,
            filters=filters,
            date_range=date_range,
            sort_preference=sort_preference,
            limit=limit,
            is_follow_up=is_follow_up,
            resolution_notes=resolution_notes,
            query_scope=query_scope,
        )

    def _normalize_query(self, query: str, notes: list[str]) -> str:
        """Normalize the query text.

        Args:
            query: Raw query.
            notes: List to append resolution notes.

        Returns:
            Normalized query.
        """
        normalized = query.strip()

        # Expand common abbreviations
        expansions = [
            (r"\bcust\b", "customer"),
            (r"\bcusts\b", "customers"),
            (r"\binfo\b", "information"),
            (r"\bnum\b", "number"),
            (r"\baddr\b", "address"),
        ]

        for pattern, replacement in expansions:
            if re.search(pattern, normalized, re.IGNORECASE):
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
                notes.append(f"Expanded '{pattern}' to '{replacement}'")

        # Correct state typos BEFORE normalizing state names
        normalized = self._correct_state_typos(normalized, notes)

        # Normalize state names to codes
        for name, code in self.STATE_NAMES.items():
            if name in normalized.lower():
                normalized = re.sub(
                    rf"\b{re.escape(name)}\b",
                    code,
                    normalized,
                    flags=re.IGNORECASE
                )
                notes.append(f"'{name}' → {code}")

        return normalized

    def _extract_entities(
        self,
        query: str,
        context: QueryContext,
        notes: list[str],
    ) -> dict[str, list[str]]:
        """Extract entities from query.

        Args:
            query: User query.
            context: Session context.
            notes: Resolution notes.

        Returns:
            Dict of entity type to list of values.
        """
        entities: dict[str, list[str]] = {
            "crids": [],
            "names": [],
            "states": [],
            "cities": [],
            "zips": [],
        }

        # Extract CRIDs
        crids = re.findall(r"CRID[-\s]?(\d+)", query, re.IGNORECASE)
        entities["crids"] = [f"CRID-{c.zfill(5)}" for c in crids]

        # Extract state codes with context-aware filtering
        # Find all 2-letter uppercase sequences with their positions
        for match in re.finditer(r"\b([A-Z]{2})\b", query):
            code = match.group(1)
            position = match.start()

            # Skip if not a valid state code
            if code not in self.VALID_STATE_CODES:
                continue

            # For ambiguous codes (IN, OR, ME, etc.), require clear context
            if code in self.AMBIGUOUS_STATE_CODES:
                if not self._is_likely_state_code_context(query, code, position):
                    logger.debug(f"Skipping ambiguous state code '{code}' - not in state context")
                    continue

            if code not in entities["states"]:
                entities["states"].append(code)

        # Correct state typos before looking up state names
        corrected_query = self._correct_state_typos(query, notes)

        # Extract state names (exact match) and convert to codes
        for name, code in self.STATE_NAMES.items():
            # Use word boundary matching to avoid partial matches
            pattern = rf"\b{re.escape(name)}\b"
            if re.search(pattern, corrected_query.lower()) and code not in entities["states"]:
                entities["states"].append(code)
                notes.append(f"State: '{name}' → {code}")

        # Fuzzy match for potential misspelled state names
        # Look for capitalized words that might be misspelled states
        words = re.findall(r"\b([A-Z][a-z]+)\b", query)
        for word in words:
            # Skip if too short or already matched
            if len(word) < 4:
                continue

            # Try fuzzy matching
            fuzzy_code = self._fuzzy_match_state(word, notes)
            if fuzzy_code and fuzzy_code not in entities["states"]:
                entities["states"].append(fuzzy_code)

        # Extract ZIP codes
        zips = re.findall(r"\b(\d{5})\b", query)
        entities["zips"] = zips

        # Add referenced entities from context
        if context.referenced_entities:
            for entity in context.referenced_entities:
                if entity.startswith("CRID") and entity not in entities["crids"]:
                    entities["crids"].append(entity)

        # Add geographic context
        if context.geographic_context.get("state"):
            state = context.geographic_context["state"]
            if state not in entities["states"]:
                entities["states"].append(state)
                notes.append(f"State from context: {state}")

        return entities

    def _extract_filters(
        self,
        query: str,
        context: QueryContext,
        notes: list[str],
    ) -> dict[str, Any]:
        """Extract filter criteria from query.

        Args:
            query: User query.
            context: Session context.
            notes: Resolution notes.

        Returns:
            Dict of filter criteria.
        """
        filters: dict[str, Any] = {}

        # Correct state typos before processing
        corrected_query = self._correct_state_typos(query, notes)
        query_lower = corrected_query.lower()
        logger.debug(f"ParserAgent._extract_filters: query_lower={query_lower}")

        # Extract state filter - use word boundaries to avoid partial matches
        for name, code in self.STATE_NAMES.items():
            pattern = rf"\b{re.escape(name)}\b"
            if re.search(pattern, query_lower):
                filters["state"] = code
                logger.debug(f"ParserAgent._extract_filters: Found state name '{name}' -> {code}")
                break

        # Try fuzzy matching for misspelled state names if no exact match
        if "state" not in filters:
            words = re.findall(r"\b([A-Z][a-z]+)\b", query)
            for word in words:
                fuzzy_code = self._fuzzy_match_state(word, notes)
                if fuzzy_code:
                    filters["state"] = fuzzy_code
                    logger.debug(f"ParserAgent._extract_filters: Fuzzy matched '{word}' -> {fuzzy_code}")
                    break

        # Check for state codes (but filter out ambiguous ones without context)
        if "state" not in filters:
            for match in re.finditer(r"\b([A-Z]{2})\b", query):
                code = match.group(1)
                position = match.start()

                if code not in self.VALID_STATE_CODES:
                    continue

                # For ambiguous codes, require clear context
                if code in self.AMBIGUOUS_STATE_CODES:
                    if not self._is_likely_state_code_context(query, code, position):
                        logger.debug(f"Skipping ambiguous state code '{code}' in filters")
                        continue

                filters["state"] = code
                break

        # Use context state if not in query
        if "state" not in filters and context.geographic_context.get("state"):
            filters["state"] = context.geographic_context["state"]
            notes.append(f"Using state from context: {filters['state']}")

        # Extract move count filter
        for pattern, default_count in self.MOVE_PATTERNS.items():
            match = re.search(pattern, query_lower)
            if match:
                if default_count is not None:
                    filters["min_move_count"] = default_count
                    notes.append(f"'{pattern}' → min_move_count: {default_count}")
                else:
                    # Extract number from match
                    num = int(match.group(1))
                    filters["min_move_count"] = num
                    notes.append(f"Extracted move count: {num}")
                break

        # Extract city filter (basic heuristic)
        city_patterns = [
            r"(?:in|from|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*,?\s*[A-Z]{2}",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+customers?",
        ]
        for pattern in city_patterns:
            match = re.search(pattern, query)
            if match:
                city = match.group(1)
                # Exclude state names
                if city.lower() not in self.STATE_NAMES:
                    filters["city"] = city
                    break

        # Validate city/state combination (if both are present)
        if "city" in filters and "state" in filters:
            mismatch = self._city_state_validator.validate(
                filters["city"], filters["state"]
            )
            if mismatch.has_mismatch:
                # Store mismatch for suggestion generation
                filters["_city_state_mismatch"] = mismatch
                notes.append(
                    f"City/state mismatch detected: {mismatch.city}, "
                    f"{mismatch.stated_state} - expected {mismatch.expected_state}"
                )
                logger.info(
                    f"City/state mismatch: {mismatch.city}, {mismatch.stated_state} "
                    f"(expected: {mismatch.expected_state})"
                )

        # Extract customer type filter
        if any(word in query_lower for word in ["business", "company", "companies", "corporate"]):
            filters["customer_type"] = "BUSINESS"
            notes.append("Customer type: BUSINESS")
        elif any(word in query_lower for word in ["residential", "homeowner", "renter"]):
            filters["customer_type"] = "RESIDENTIAL"
            notes.append("Customer type: RESIDENTIAL")
        elif "po box" in query_lower or "pobox" in query_lower:
            filters["customer_type"] = "PO_BOX"
            notes.append("Customer type: PO_BOX")

        # Extract apartment filter
        if any(word in query_lower for word in ["apartment", "apt", "unit", "condo"]):
            filters["has_apartment"] = True
            notes.append("Filter: apartment/unit addresses only")

        # =====================================================================
        # NEW: Extract status filter (ACTIVE, INACTIVE, PENDING)
        # This enables queries like "show me inactive customers"
        # =====================================================================
        for status, patterns in self.STATUS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    filters["status"] = status
                    notes.append(f"Status filter: {status}")
                    logger.debug(f"ParserAgent: Extracted status={status} from pattern '{pattern}'")
                    break
            if "status" in filters:
                break

        # =====================================================================
        # NEW: Extract origin state (for "moved from X" queries)
        # IMPORTANT: Do this BEFORE state filter so we can exclude origin from main state
        # This enables queries like "customers who moved from California"
        # =====================================================================
        for pattern in self.MOVE_ORIGIN_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                origin_text = match.group(1).strip()
                # Try to resolve to a state code
                origin_code = None

                # Check exact state name match
                if origin_text.lower() in self.STATE_NAMES:
                    origin_code = self.STATE_NAMES[origin_text.lower()]
                else:
                    # Try fuzzy matching
                    origin_code = self._fuzzy_match_state(origin_text.title(), notes)

                if origin_code:
                    filters["origin_state"] = origin_code
                    notes.append(f"Origin state filter: {origin_text} -> {origin_code}")
                    logger.debug(f"ParserAgent: Extracted origin_state={origin_code} from '{origin_text}'")

                    # FIX: If origin_state matches the current "state" filter, we need to
                    # re-extract the main state by finding ANOTHER state in the query
                    if filters.get("state") == origin_code:
                        logger.debug(f"ParserAgent: state={origin_code} same as origin, re-extracting...")
                        del filters["state"]
                        # Find another state in the query that's NOT the origin
                        for name, code in self.STATE_NAMES.items():
                            if code != origin_code:
                                state_pattern = rf"\b{re.escape(name)}\b"
                                if re.search(state_pattern, query_lower):
                                    filters["state"] = code
                                    notes.append(f"Destination state: {name} -> {code}")
                                    logger.debug(f"ParserAgent: Re-extracted state={code} (destination)")
                                    break
                    break
                else:
                    logger.warning(f"ParserAgent: Found 'moved from' pattern but couldn't resolve '{origin_text}' to state code")

        logger.debug(f"ParserAgent._extract_filters: FINAL filters={filters}")
        return filters

    def _extract_date_range(
        self,
        query: str,
        notes: list[str],
    ) -> tuple[str, str] | None:
        """Extract date range from query.

        Args:
            query: User query.
            notes: Resolution notes.

        Returns:
            Tuple of (start_date, end_date) or None.
        """
        query_lower = query.lower()

        # Look for relative date patterns
        if "last month" in query_lower:
            notes.append("Date range: last month")
            return ("last_month_start", "last_month_end")
        if "last year" in query_lower:
            notes.append("Date range: last year")
            return ("last_year_start", "last_year_end")
        if "this year" in query_lower:
            notes.append("Date range: this year")
            return ("this_year_start", "today")
        if "recently" in query_lower or "recent" in query_lower:
            notes.append("Date range: recent (last 30 days)")
            return ("30_days_ago", "today")

        # Look for specific date patterns (YYYY-MM-DD or MM/DD/YYYY)
        date_pattern = r"(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})"
        dates = re.findall(date_pattern, query)
        if len(dates) >= 2:
            return (dates[0], dates[1])
        if len(dates) == 1:
            return (dates[0], "today")

        return None

    def _extract_limit(self, query: str, context: QueryContext) -> int:
        """Extract result limit from query.

        Args:
            query: User query.
            context: Session context.

        Returns:
            Result limit.
        """
        # Check for explicit limit
        limit_patterns = [
            r"(?:top|first|show|give\s+me)\s+(\d+)",
            r"(\d+)\s+(?:customers?|results?|records?)",
            r"limit\s+(\d+)",
        ]

        for pattern in limit_patterns:
            match = re.search(pattern, query.lower())
            if match:
                return min(int(match.group(1)), 100)

        # Use context preference
        return context.user_preferences.get("preferred_limit", 10)

    def _extract_sort(self, query: str) -> str | None:
        """Extract sort preference from query.

        Args:
            query: User query.

        Returns:
            Sort field or None.
        """
        query_lower = query.lower()

        if "most moves" in query_lower or "highest movers" in query_lower:
            return "move_count_desc"
        if "least moves" in query_lower or "fewest moves" in query_lower:
            return "move_count_asc"
        if "recent" in query_lower or "newest" in query_lower:
            return "created_date_desc"
        if "oldest" in query_lower:
            return "created_date_asc"
        if "alphabetical" in query_lower or "by name" in query_lower:
            return "name_asc"

        return None

    def _assess_query_scope(
        self,
        query: str,
        intent: IntentResult,
        entities: dict[str, list[str]],
        filters: dict[str, Any],
    ) -> dict[str, Any]:
        """Assess if the query is within scope of the customer data system.

        Args:
            query: Original user query.
            intent: Intent classification result.
            entities: Extracted entities.
            filters: Extracted filters.

        Returns:
            Dict with assessment, confidence, and reason.
        """
        # Check if intent was classified as OUT_OF_SCOPE
        if intent.primary_intent == QueryIntent.OUT_OF_SCOPE:
            return {
                "assessment": "out_of_scope",
                "confidence": intent.confidence,
                "reason": "Query classified as outside customer data domain",
            }

        # Check if intent is conversational
        if intent.primary_intent == QueryIntent.CONVERSATIONAL:
            return {
                "assessment": "conversational",
                "confidence": intent.confidence,
                "reason": "Conversational query - no data lookup needed",
            }

        # Check if query has any customer data indicators
        has_entities = any(
            entities.get(key) for key in ["crids", "states", "cities", "zips"]
        )
        has_filters = bool(filters)
        has_data_keywords = self._has_data_keywords(query)

        if has_entities or has_filters or has_data_keywords:
            return {
                "assessment": "in_scope",
                "confidence": intent.confidence,
                "reason": "Query contains customer data indicators",
            }

        # Query has no clear customer data indicators - might be ambiguous
        return {
            "assessment": "in_scope",
            "confidence": max(0.5, intent.confidence - 0.2),
            "reason": "Query may be ambiguous but defaulting to in-scope",
        }

    def _has_data_keywords(self, query: str) -> bool:
        """Check if query contains customer/data-related keywords."""
        data_keywords = (
            "customer", "crid", "state", "city", "address", "moved",
            "search", "find", "list", "show", "count", "how many",
            "stats", "nevada", "california", "texas", "active", "inactive",
            "zip", "status", "movers", "relocated"
        )
        q = query.lower()
        return any(kw in q for kw in data_keywords)
