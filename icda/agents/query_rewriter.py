"""Query Rewriter - Smart query normalization and disambiguation.

This module transforms ambiguous user queries into explicit customer data queries.
It handles:
1. Ambiguous phrasing ("people in virginia" → "customers in virginia")
2. State misspellings with fuzzy matching ("vaginia" → "Virginia")
3. City/state disambiguation ("Kansas City" → clarify MO vs KS)
4. Slang and informal references ("cali folks" → "California customers")

The goal is to ensure Nova Pro interprets queries as customer data queries,
not demographic/census questions.
"""

import logging
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# State Normalization Data
# =============================================================================

# State name to code (canonical)
STATE_NAME_TO_CODE: dict[str, str] = {
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
    "district of columbia": "DC", "d.c.": "DC", "dc": "DC",
}

# Code to full name (for display)
STATE_CODE_TO_NAME: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "PR": "Puerto Rico", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

# Valid 2-letter state codes
VALID_STATE_CODES: set[str] = set(STATE_CODE_TO_NAME.keys())

# All state names for fuzzy matching
ALL_STATE_NAMES: list[str] = list(STATE_NAME_TO_CODE.keys())

# =============================================================================
# State Misspelling Database (comprehensive)
# =============================================================================

STATE_MISSPELLINGS: dict[str, str] = {
    # Virginia variations (the original problem!)
    "vaginia": "virginia", "virgina": "virginia", "virgnia": "virginia",
    "virignia": "virginia", "virgina": "virginia", "virgiinia": "virginia",
    "virginai": "virginia", "verginia": "virginia", "virgenia": "virginia",
    "va": "virginia",  # Common abbreviation used as word

    # California variations
    "californa": "california", "californai": "california", "califronia": "california",
    "calfornia": "california", "califonia": "california", "californnia": "california",
    "cali": "california", "cal": "california",

    # Kansas variations
    "kanas": "kansas", "kanses": "kansas", "kanzas": "kansas", "kanss": "kansas",
    "kansass": "kansas",
    # Note: "kansas city" is NOT a misspelling - it's a real city in MO and KS

    # Florida variations
    "flordia": "florida", "florda": "florida", "floirda": "florida",
    "floridia": "florida", "flordia": "florida", "fla": "florida",

    # Texas variations
    "texs": "texas", "texsa": "texas", "teaxs": "texas", "texass": "texas",
    "tx": "texas",

    # Arizona variations
    "arizon": "arizona", "arizonia": "arizona", "arizonza": "arizona",
    "arizna": "arizona", "az": "arizona",

    # New York variations
    "newyork": "new york", "new yor": "new york", "newy ork": "new york",
    "nyc": "new york", "ny": "new york",

    # Pennsylvania variations
    "pennsylvnia": "pennsylvania", "pensilvania": "pennsylvania",
    "pensylvania": "pennsylvania", "pennslvania": "pennsylvania",
    "penn": "pennsylvania", "pa": "pennsylvania",

    # Georgia variations
    "gorgia": "georgia", "gerogia": "georgia", "georga": "georgia",
    "goergia": "georgia", "ga": "georgia",

    # Michigan variations
    "michagan": "michigan", "michgan": "michigan", "michighan": "michigan",
    "mich": "michigan", "mi": "michigan",

    # Colorado variations
    "colordo": "colorado", "colorodo": "colorado", "colrado": "colorado",
    "colo": "colorado", "co": "colorado",

    # Nevada variations
    "nevda": "nevada", "navada": "nevada", "neveda": "nevada",
    "nv": "nevada", "vegas": "nevada",

    # Washington variations
    "washingon": "washington", "washigton": "washington", "washinton": "washington",
    "wa": "washington",

    # Illinois variations
    "illnois": "illinois", "illinios": "illinois", "ilinois": "illinois",
    "ill": "illinois", "il": "illinois",

    # Ohio variations
    "ohoi": "ohio", "oho": "ohio", "oh": "ohio",

    # North Carolina variations
    "noth carolina": "north carolina", "north carolia": "north carolina",
    "northcarolina": "north carolina", "nc": "north carolina",

    # South Carolina variations
    "south carolia": "south carolina", "southcarolina": "south carolina",
    "sc": "south carolina",

    # Tennessee variations
    "tenessee": "tennessee", "tennesse": "tennessee", "tn": "tennessee",

    # Massachusetts variations
    "massachusets": "massachusetts", "massachusettes": "massachusetts",
    "mass": "massachusetts", "ma": "massachusetts",

    # Connecticut variations
    "connecticuit": "connecticut", "conneticut": "connecticut",
    "conn": "connecticut", "ct": "connecticut",

    # New Jersey variations
    "newjersey": "new jersey", "nj": "new jersey",

    # Maryland variations
    "marylnd": "maryland", "md": "maryland",

    # Indiana variations
    "indana": "indiana", "indianna": "indiana",

    # Missouri variations
    "misouri": "missouri", "missori": "missouri", "mo": "missouri",

    # Wisconsin variations
    "wisconson": "wisconsin", "wisconsion": "wisconsin", "wi": "wisconsin",

    # Minnesota variations
    "minesota": "minnesota", "minessota": "minnesota", "mn": "minnesota",

    # Oregon variations
    "oregan": "oregon", "oregn": "oregon",

    # Kentucky variations
    "kentuckey": "kentucky", "kentucty": "kentucky", "ky": "kentucky",

    # Louisiana variations
    "louisianna": "louisiana", "louisana": "louisiana", "la": "louisiana",

    # Alabama variations
    "alabma": "alabama", "al": "alabama",

    # Mississippi variations
    "missisippi": "mississippi", "mississipi": "mississippi", "ms": "mississippi",

    # Arkansas variations
    "arkasas": "arkansas", "ar": "arkansas",

    # Oklahoma variations
    "oklahma": "oklahoma", "ok": "oklahoma",

    # Iowa variations
    "iowaa": "iowa", "ia": "iowa",

    # Utah variations
    "uta": "utah", "ut": "utah",

    # New Mexico variations
    "newmexico": "new mexico", "nm": "new mexico",

    # Hawaii variations
    "hawai": "hawaii", "hawii": "hawaii", "hi": "hawaii",

    # Idaho variations
    "idahoe": "idaho", "id": "idaho",

    # Montana variations
    "montanna": "montana", "mt": "montana",

    # Wyoming variations
    "wyomming": "wyoming", "wy": "wyoming",

    # Nebraska variations
    "nebraska": "nebraska", "ne": "nebraska",

    # North Dakota variations
    "northdakota": "north dakota", "nd": "north dakota",

    # South Dakota variations
    "southdakota": "south dakota", "sd": "south dakota",

    # Vermont variations
    "vermon": "vermont", "vt": "vermont",

    # New Hampshire variations
    "newhampshire": "new hampshire", "nh": "new hampshire",

    # Maine variations
    "main": "maine",  # Note: "me" is ambiguous with pronoun

    # Rhode Island variations
    "rhodeisland": "rhode island", "ri": "rhode island",

    # Delaware variations
    "deleware": "delaware", "de": "delaware",

    # West Virginia variations
    "westvirginia": "west virginia", "wv": "west virginia",
    "w virginia": "west virginia", "w. virginia": "west virginia",
}

# =============================================================================
# Slang and Informal References
# =============================================================================

INFORMAL_STATE_REFS: dict[str, str] = {
    # Nicknames
    "the golden state": "california",
    "golden state": "california",
    "the sunshine state": "florida",
    "sunshine state": "florida",
    "the lone star state": "texas",
    "lone star state": "texas",
    "lone star": "texas",
    "the empire state": "new york",
    "empire state": "new york",
    "the garden state": "new jersey",
    "garden state": "new jersey",
    "the buckeye state": "ohio",
    "buckeye state": "ohio",
    "the peach state": "georgia",
    "peach state": "georgia",
    "the grand canyon state": "arizona",
    "sin city": "nevada",  # Las Vegas reference
    "motor city": "michigan",  # Detroit reference
    "windy city": "illinois",  # Chicago reference

    # Slang
    "cali": "california",
    "socal": "california",
    "norcal": "california",
    "the bay": "california",  # Bay Area
    "bay area": "california",
    "silicon valley": "california",
    "vegas": "nevada",
    "the district": "district of columbia",
    "the dmv": "district of columbia",  # DC/Maryland/Virginia area
    "nyc": "new york",
    "the city": "new york",  # NYC context
    "chi-town": "illinois",
    "chiraq": "illinois",  # Chicago slang
    "philly": "pennsylvania",
    "the keys": "florida",  # Florida Keys
    "south beach": "florida",
    "miami beach": "florida",
    "bourbon street": "louisiana",  # New Orleans reference
    "music city": "tennessee",  # Nashville
    "mile high city": "colorado",  # Denver
}

# =============================================================================
# Ambiguous City Database
# =============================================================================

AMBIGUOUS_CITIES: dict[str, list[str]] = {
    "kansas city": ["MO", "KS"],  # Major city spanning both states
    "springfield": ["IL", "MA", "MO", "OH", "OR"],
    "portland": ["OR", "ME"],
    "columbus": ["OH", "GA", "IN"],
    "richmond": ["VA", "CA", "IN"],
    "jackson": ["MS", "TN", "MI", "WY"],
    "columbia": ["SC", "MO", "MD"],
    "aurora": ["CO", "IL"],
    "arlington": ["TX", "VA"],
    "hollywood": ["FL", "CA"],
    "salem": ["OR", "MA"],
    "greenville": ["SC", "NC", "MS"],
    "manchester": ["NH", "CT"],
    "fayetteville": ["NC", "AR"],
    "wilmington": ["NC", "DE"],
    "peoria": ["IL", "AZ"],
    "glendale": ["AZ", "CA"],
    "pasadena": ["CA", "TX"],
    "athens": ["GA", "OH"],
    "vancouver": ["WA", "BC"],  # BC is Canada but often confused
    "st. louis": ["MO", "IL"],  # East St. Louis is IL
    "saint louis": ["MO", "IL"],
}

# Primary state for ambiguous cities (most populous/famous)
AMBIGUOUS_CITY_PRIMARY: dict[str, str] = {
    "kansas city": "MO",  # KC proper is in MO
    "portland": "OR",     # Oregon Portland is larger
    "springfield": "IL",  # Illinois capital
    "columbus": "OH",     # Ohio is largest Columbus
    "vancouver": "WA",    # In US context
}


# =============================================================================
# Ambiguous People/Population Words
# =============================================================================

# Words that sound like demographic queries but should mean "customers"
PEOPLE_WORDS: list[str] = [
    "people",
    "folks",
    "residents",
    "inhabitants",
    "population",
    "citizens",
    "individuals",
    "persons",
    "humans",
    "men",
    "women",
    "families",
    "households",
    "occupants",
    "locals",
    "natives",
    "dwellers",
]


@dataclass
class RewriteResult:
    """Result of query rewriting."""

    original_query: str
    rewritten_query: str
    was_rewritten: bool
    rewrites_applied: list[str] = field(default_factory=list)
    detected_state: str | None = None
    detected_state_code: str | None = None
    detected_city: str | None = None
    is_ambiguous_city: bool = False
    ambiguous_city_states: list[str] | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "was_rewritten": self.was_rewritten,
            "rewrites_applied": self.rewrites_applied,
            "detected_state": self.detected_state,
            "detected_state_code": self.detected_state_code,
            "detected_city": self.detected_city,
            "is_ambiguous_city": self.is_ambiguous_city,
            "ambiguous_city_states": self.ambiguous_city_states,
            "confidence": self.confidence,
        }


class QueryRewriter:
    """Smart query rewriter for customer data queries.

    Transforms ambiguous queries into explicit customer data queries
    to prevent Nova Pro from misinterpreting them as demographic questions.
    """

    __slots__ = ("_misspellings", "_informal_refs", "_ambiguous_cities")

    def __init__(self) -> None:
        """Initialize the query rewriter."""
        self._misspellings = STATE_MISSPELLINGS
        self._informal_refs = INFORMAL_STATE_REFS
        self._ambiguous_cities = AMBIGUOUS_CITIES

    def rewrite(self, query: str) -> RewriteResult:
        """Rewrite a query to be explicit about customer data.

        Args:
            query: Original user query.

        Returns:
            RewriteResult with rewritten query and metadata.
        """
        original = query
        rewrites: list[str] = []
        rewritten = query
        detected_state: str | None = None
        detected_state_code: str | None = None
        detected_city: str | None = None
        is_ambiguous = False
        ambiguous_states: list[str] | None = None
        confidence = 1.0

        # Step 1: Detect and handle ambiguous cities FIRST
        # This must come before state misspelling fixes to prevent
        # "kansas city" from being treated as state "kansas"
        rewritten, city_info = self._handle_ambiguous_cities(rewritten)
        if city_info:
            detected_city = city_info.get("city")
            is_ambiguous = city_info.get("is_ambiguous", False)
            ambiguous_states = city_info.get("possible_states")
            if city_info.get("rewrite"):
                rewrites.append(city_info["rewrite"])

        # Step 2: Fix state misspellings (after city detection)
        rewritten, state_fixes = self._fix_state_misspellings(rewritten)
        rewrites.extend(state_fixes)

        # Step 3: Expand informal state references
        rewritten, informal_fixes = self._expand_informal_refs(rewritten)
        rewrites.extend(informal_fixes)

        # Step 4: Replace people words with "customers"
        rewritten, people_fixes = self._replace_people_words(rewritten)
        rewrites.extend(people_fixes)

        # Step 5: Detect state in the query
        detected_state, detected_state_code = self._detect_state(rewritten)

        # Step 6: If query mentions a state but doesn't clarify "customers",
        # and uses demographic-sounding language, add clarification
        if detected_state_code and not self._has_customer_word(rewritten):
            rewritten, demo_fix = self._add_customer_context(rewritten, detected_state)
            if demo_fix:
                rewrites.append(demo_fix)

        # Calculate confidence
        if is_ambiguous:
            confidence = 0.7
        elif len(rewrites) > 3:
            confidence = 0.8

        was_rewritten = rewritten.lower().strip() != original.lower().strip()

        if was_rewritten:
            logger.info(
                f"QueryRewriter: '{original}' → '{rewritten}' "
                f"(rewrites: {rewrites})"
            )

        return RewriteResult(
            original_query=original,
            rewritten_query=rewritten,
            was_rewritten=was_rewritten,
            rewrites_applied=rewrites,
            detected_state=detected_state,
            detected_state_code=detected_state_code,
            detected_city=detected_city,
            is_ambiguous_city=is_ambiguous,
            ambiguous_city_states=ambiguous_states,
            confidence=confidence,
        )

    def _fix_state_misspellings(self, query: str) -> tuple[str, list[str]]:
        """Fix state name misspellings.

        Args:
            query: Query to fix.

        Returns:
            Tuple of (fixed query, list of fixes applied).
        """
        fixes: list[str] = []
        result = query
        query_lower = query.lower()

        # Sort by length descending to match longer phrases first
        sorted_misspellings = sorted(
            self._misspellings.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for misspelling, correct in sorted_misspellings:
            # Skip very short abbreviations that could be false positives
            if len(misspelling) <= 2:
                # Only match 2-letter codes if they appear as standalone words
                # AND are clearly meant as state codes (e.g., after a city)
                pattern = rf"\b{re.escape(misspelling)}\b"
                if re.search(pattern, query_lower):
                    # Check context - is it likely a state code?
                    if self._is_likely_state_context(query, misspelling):
                        result = re.sub(
                            pattern,
                            correct,
                            result,
                            flags=re.IGNORECASE
                        )
                        fixes.append(f"'{misspelling}' → '{correct}'")
            else:
                # For longer misspellings, just match word boundaries
                pattern = rf"\b{re.escape(misspelling)}\b"
                if re.search(pattern, query_lower):
                    result = re.sub(
                        pattern,
                        correct,
                        result,
                        flags=re.IGNORECASE
                    )
                    fixes.append(f"'{misspelling}' → '{correct}'")

        return result, fixes

    def _is_likely_state_context(self, query: str, code: str) -> bool:
        """Check if a 2-letter code is likely meant as a state code.

        Args:
            query: Full query.
            code: 2-letter code found.

        Returns:
            True if likely a state code.
        """
        code_lower = code.lower()
        query_lower = query.lower()

        # Ambiguous codes that are common English words
        ambiguous = {"in", "or", "me", "ok", "hi", "oh", "la", "pa", "ma", "id", "al"}

        if code_lower in ambiguous:
            # Require stronger context for these
            # Look for patterns like "City, XX" or "in XX" where XX follows "in"
            patterns = [
                rf"[a-z]+,?\s*{re.escape(code_lower)}\b",  # City, XX
                rf"from\s+{re.escape(code_lower)}\b",      # from XX
                rf"to\s+{re.escape(code_lower)}\b",        # to XX
                rf"\d{{5}}\s*{re.escape(code_lower)}\b",   # ZIP XX
            ]
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return True
            return False

        # Non-ambiguous codes are always treated as state codes
        return True

    def _expand_informal_refs(self, query: str) -> tuple[str, list[str]]:
        """Expand informal state references (nicknames, slang).

        Args:
            query: Query to expand.

        Returns:
            Tuple of (expanded query, list of expansions).
        """
        fixes: list[str] = []
        result = query
        query_lower = query.lower()

        # Sort by length descending
        sorted_refs = sorted(
            self._informal_refs.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for informal, formal in sorted_refs:
            pattern = rf"\b{re.escape(informal)}\b"
            if re.search(pattern, query_lower):
                result = re.sub(
                    pattern,
                    formal,
                    result,
                    flags=re.IGNORECASE
                )
                fixes.append(f"'{informal}' → '{formal}'")

        return result, fixes

    # State codes that are common English words - require explicit context
    AMBIGUOUS_STATE_CODES: set[str] = {
        "IN", "OR", "ME", "OK", "HI", "OH", "LA", "PA", "MA", "ID", "AL", "CO", "DE"
    }

    def _handle_ambiguous_cities(self, query: str) -> tuple[str, dict[str, Any] | None]:
        """Handle cities that exist in multiple states.

        Args:
            query: Query to check.

        Returns:
            Tuple of (potentially modified query, city info dict or None).
        """
        query_lower = query.lower()

        # Sort by length descending to match longer city names first
        # (e.g., "kansas city" before just partial matching)
        sorted_cities = sorted(
            self._ambiguous_cities.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for city, states in sorted_cities:
            pattern = rf"\b{re.escape(city)}\b"
            if re.search(pattern, query_lower):
                # Check if a state is already EXPLICITLY specified
                # Don't count the city name as a state (e.g., "kansas" in "kansas city")
                has_state = False

                # Create a version of query without the city name for state detection
                query_without_city = re.sub(pattern, " ", query_lower, flags=re.IGNORECASE)

                # First check for full state names (unambiguous)
                for state_name in ALL_STATE_NAMES:
                    state_pattern = rf"\b{re.escape(state_name)}\b"
                    if re.search(state_pattern, query_without_city):
                        has_state = True
                        break

                # Then check for state codes, but filter out ambiguous ones
                # that could be English words (e.g., "in", "or", "me")
                if not has_state:
                    for code in VALID_STATE_CODES:
                        # Skip ambiguous codes that could be prepositions/words
                        if code in self.AMBIGUOUS_STATE_CODES:
                            # For ambiguous codes, require stronger context:
                            # - After a comma (e.g., "Springfield, IL")
                            # - After a city that's in that state's possible list
                            comma_pattern = rf",\s*{re.escape(code)}\b"
                            if re.search(comma_pattern, query_without_city, re.IGNORECASE):
                                has_state = True
                                break
                        else:
                            # Non-ambiguous codes are fine
                            if re.search(rf"\b{re.escape(code)}\b", query_without_city, re.IGNORECASE):
                                has_state = True
                                break

                if has_state:
                    # State is specified, no ambiguity
                    return query, {
                        "city": city.title(),
                        "is_ambiguous": False,
                        "possible_states": states,
                    }

                # No state specified - note ambiguity
                # Use the primary state as default
                primary = AMBIGUOUS_CITY_PRIMARY.get(city, states[0])

                return query, {
                    "city": city.title(),
                    "is_ambiguous": True,
                    "possible_states": states,
                    "default_state": primary,
                    "rewrite": f"'{city}' is ambiguous (could be {', '.join(states)}), assuming {primary}",
                }

        return query, None

    def _replace_people_words(self, query: str) -> tuple[str, list[str]]:
        """Replace demographic words with 'customers'.

        Transforms "people in virginia" to "customers in virginia".

        Args:
            query: Query to transform.

        Returns:
            Tuple of (transformed query, list of replacements).
        """
        fixes: list[str] = []
        result = query
        query_lower = query.lower()

        for word in PEOPLE_WORDS:
            # Match word with various patterns
            patterns = [
                # "how many people" → "how many customers"
                (rf"\bhow\s+many\s+{re.escape(word)}\b", f"how many customers"),
                # "people in/from/who" → "customers in/from/who"
                (rf"\b{re.escape(word)}\s+(in|from|who|that|living)\b", r"customers \1"),
                # "X people" → "X customers" (where X is a number)
                (rf"\b(\d+)\s+{re.escape(word)}\b", r"\1 customers"),
                # Just the word alone in certain contexts
                (rf"\bthe\s+{re.escape(word)}\s+(in|from|of)\b", r"the customers \1"),
            ]

            for pattern, replacement in patterns:
                if re.search(pattern, query_lower):
                    old_result = result
                    result = re.sub(
                        pattern,
                        replacement,
                        result,
                        flags=re.IGNORECASE
                    )
                    if result != old_result:
                        fixes.append(f"'{word}' → 'customers'")

        return result, fixes

    def _detect_state(self, query: str) -> tuple[str | None, str | None]:
        """Detect state mentioned in query.

        Args:
            query: Query to analyze.

        Returns:
            Tuple of (state name, state code) or (None, None).
        """
        query_lower = query.lower()

        # First try full state names (sorted by length descending)
        sorted_names = sorted(
            STATE_NAME_TO_CODE.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for name, code in sorted_names:
            pattern = rf"\b{re.escape(name)}\b"
            if re.search(pattern, query_lower):
                return name.title(), code

        # Then try state codes
        for code in VALID_STATE_CODES:
            # Skip ambiguous codes without context check
            if code.lower() in {"in", "or", "me", "ok", "hi", "oh", "la", "pa", "ma"}:
                continue

            pattern = rf"\b{re.escape(code)}\b"
            if re.search(pattern, query, re.IGNORECASE):
                return STATE_CODE_TO_NAME.get(code, code), code

        # Try fuzzy matching for potential misspellings
        words = re.findall(r"\b[A-Za-z]{4,}\b", query)
        for word in words:
            matches = get_close_matches(
                word.lower(),
                ALL_STATE_NAMES,
                n=1,
                cutoff=0.8
            )
            if matches:
                matched_name = matches[0]
                code = STATE_NAME_TO_CODE[matched_name]
                return matched_name.title(), code

        return None, None

    def _has_customer_word(self, query: str) -> bool:
        """Check if query already has a customer-related word.

        Args:
            query: Query to check.

        Returns:
            True if query mentions customers.
        """
        customer_words = [
            "customer", "customers", "client", "clients",
            "user", "users", "account", "accounts",
            "subscriber", "subscribers", "member", "members",
            "crid", "record", "records", "data",
        ]
        query_lower = query.lower()
        return any(word in query_lower for word in customer_words)

    def _add_customer_context(
        self,
        query: str,
        state_name: str | None,
    ) -> tuple[str, str | None]:
        """Add customer context to demographic-sounding queries.

        Transforms "how many in virginia?" to "how many customers in virginia?"

        Args:
            query: Query to transform.
            state_name: Detected state name.

        Returns:
            Tuple of (transformed query, fix description or None).
        """
        query_lower = query.lower()

        # Patterns that suggest counting/demographic questions
        patterns = [
            # "how many in X" → "how many customers in X"
            (r"\bhow\s+many\s+(in|from)\b", r"how many customers \1"),
            # "count in X" → "customer count in X"
            (r"\bcount\s+(in|from)\b", r"customer count \1"),
            # "total in X" → "total customers in X"
            (r"\btotal\s+(in|from)\b", r"total customers \1"),
            # "number in X" → "number of customers in X"
            (r"\bnumber\s+(in|from)\b", r"number of customers \1"),
            # "live in X" → "customers who live in X"
            (r"\blive\s+in\b", r"customers who live in"),
            # "living in X" → "customers living in X"
            (r"\bliving\s+in\b", r"customers living in"),
        ]

        for pattern, replacement in patterns:
            if re.search(pattern, query_lower):
                old_query = query
                query = re.sub(
                    pattern,
                    replacement,
                    query,
                    flags=re.IGNORECASE
                )
                if query != old_query:
                    return query, f"Added 'customers' context for state query"

        return query, None

    def normalize_state(self, state_input: str) -> tuple[str | None, str | None]:
        """Normalize a state input to canonical form.

        Args:
            state_input: State name, code, or misspelling.

        Returns:
            Tuple of (full name, state code) or (None, None).
        """
        if not state_input:
            return None, None

        input_lower = state_input.lower().strip()

        # Check if it's already a valid code
        if input_lower.upper() in VALID_STATE_CODES:
            code = input_lower.upper()
            return STATE_CODE_TO_NAME.get(code), code

        # Check if it's a full name
        if input_lower in STATE_NAME_TO_CODE:
            code = STATE_NAME_TO_CODE[input_lower]
            return input_lower.title(), code

        # Check misspellings
        if input_lower in self._misspellings:
            correct = self._misspellings[input_lower]
            code = STATE_NAME_TO_CODE.get(correct)
            if code:
                return correct.title(), code

        # Check informal refs
        if input_lower in self._informal_refs:
            correct = self._informal_refs[input_lower]
            code = STATE_NAME_TO_CODE.get(correct)
            if code:
                return correct.title(), code

        # Try fuzzy matching
        matches = get_close_matches(input_lower, ALL_STATE_NAMES, n=1, cutoff=0.75)
        if matches:
            matched = matches[0]
            code = STATE_NAME_TO_CODE[matched]
            return matched.title(), code

        return None, None


# Module-level instance for convenience
_rewriter = QueryRewriter()


def rewrite_query(query: str) -> RewriteResult:
    """Convenience function to rewrite a query.

    Args:
        query: Original user query.

    Returns:
        RewriteResult with rewritten query.
    """
    return _rewriter.rewrite(query)


def normalize_state(state_input: str) -> tuple[str | None, str | None]:
    """Convenience function to normalize a state.

    Args:
        state_input: State name, code, or misspelling.

    Returns:
        Tuple of (full name, state code).
    """
    return _rewriter.normalize_state(state_input)
