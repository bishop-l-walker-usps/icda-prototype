"""RediSearch enhanced wrapper with autocomplete, facets, and highlighting.

Enhancements over basic vector search:
- Full-text search with fuzzy matching
- Autocomplete suggestions
- Faceted search (aggregations)
- Result highlighting
- Spell correction
- Phonetic matching (SOUNDEX)
"""

import asyncio
import logging
from typing import Any

from .models import SearchSuggestion, FacetResult

logger = logging.getLogger(__name__)


class RedisSearchEnhanced:
    """Enhanced RediSearch wrapper with advanced features.

    Key naming convention:
        idx:customers              - Customer search index
        suggest:address            - Address autocomplete dictionary
        suggest:city               - City autocomplete dictionary
        suggest:name               - Name autocomplete dictionary
    """

    # Autocomplete suggestion dictionaries
    SUGGEST_ADDRESS = "suggest:address"
    SUGGEST_CITY = "suggest:city"
    SUGGEST_NAME = "suggest:name"

    # Index names
    INDEX_CUSTOMERS = "idx:customers"

    def __init__(self, redis):
        self.redis = redis

    # =========================================================================
    # Autocomplete Suggestions
    # =========================================================================

    async def add_suggestion(
        self,
        dictionary: str,
        string: str,
        score: float = 1.0,
        payload: str | None = None,
        incr: bool = False,
    ) -> bool:
        """Add string to suggestion dictionary.

        Args:
            dictionary: Suggestion dictionary name
            string: String to add
            score: Initial score
            payload: Optional payload data
            incr: If True, increment existing score instead of replace

        Returns:
            True if added successfully
        """
        try:
            args = [dictionary, string, score]
            if incr:
                args.append("INCR")
            if payload:
                args.extend(["PAYLOAD", payload])

            await self.redis.execute_command("FT.SUGADD", *args)
            return True
        except Exception as e:
            logger.debug(f"Failed to add suggestion: {e}")
            return False

    async def get_suggestions(
        self,
        dictionary: str,
        prefix: str,
        fuzzy: bool = True,
        max_results: int = 10,
        with_scores: bool = False,
        with_payloads: bool = False,
    ) -> list[SearchSuggestion]:
        """Get autocomplete suggestions.

        Args:
            dictionary: Suggestion dictionary name
            prefix: Prefix to search
            fuzzy: Enable fuzzy matching
            max_results: Max suggestions to return
            with_scores: Include scores in results
            with_payloads: Include payloads in results

        Returns:
            List of SearchSuggestion objects
        """
        try:
            args = [dictionary, prefix, "MAX", max_results]
            if fuzzy:
                args.append("FUZZY")
            if with_scores:
                args.append("WITHSCORES")
            if with_payloads:
                args.append("WITHPAYLOADS")

            result = await self.redis.execute_command("FT.SUGGET", *args)

            if not result:
                return []

            suggestions = []
            i = 0
            while i < len(result):
                text = result[i]
                i += 1

                score = 1.0
                payload = None

                if with_scores and i < len(result):
                    try:
                        score = float(result[i])
                        i += 1
                    except (ValueError, TypeError):
                        pass

                if with_payloads and i < len(result):
                    payload = result[i]
                    i += 1

                suggestions.append(SearchSuggestion(
                    text=text,
                    score=score,
                    payload=payload,
                ))

            return suggestions

        except Exception as e:
            logger.debug(f"Failed to get suggestions: {e}")
            return []

    async def suggest_address(
        self,
        prefix: str,
        limit: int = 10,
        fuzzy: bool = True,
    ) -> list[SearchSuggestion]:
        """Get address autocomplete suggestions."""
        return await self.get_suggestions(
            self.SUGGEST_ADDRESS,
            prefix,
            fuzzy=fuzzy,
            max_results=limit,
            with_scores=True,
        )

    async def suggest_city(
        self,
        prefix: str,
        limit: int = 10,
        fuzzy: bool = True,
    ) -> list[SearchSuggestion]:
        """Get city autocomplete suggestions."""
        return await self.get_suggestions(
            self.SUGGEST_CITY,
            prefix,
            fuzzy=fuzzy,
            max_results=limit,
            with_scores=True,
        )

    async def suggest_name(
        self,
        prefix: str,
        limit: int = 10,
        fuzzy: bool = True,
    ) -> list[SearchSuggestion]:
        """Get name autocomplete suggestions."""
        return await self.get_suggestions(
            self.SUGGEST_NAME,
            prefix,
            fuzzy=fuzzy,
            max_results=limit,
            with_scores=True,
        )

    async def build_suggestions_from_customers(
        self,
        customers: list[dict],
        progress_callback=None,
    ) -> dict:
        """Build suggestion dictionaries from customer data.

        Args:
            customers: List of customer dicts
            progress_callback: Optional callback(processed, total)

        Returns:
            Dict with counts of suggestions added
        """
        stats = {"addresses": 0, "cities": 0, "names": 0}
        seen_cities = set()
        seen_names = set()

        total = len(customers)
        for i, customer in enumerate(customers):
            # Add address suggestion
            if address := customer.get("address"):
                if await self.add_suggestion(self.SUGGEST_ADDRESS, address):
                    stats["addresses"] += 1

            # Add city suggestion (dedupe)
            if city := customer.get("city"):
                city_lower = city.lower()
                if city_lower not in seen_cities:
                    seen_cities.add(city_lower)
                    payload = customer.get("state", "")
                    if await self.add_suggestion(self.SUGGEST_CITY, city, payload=payload):
                        stats["cities"] += 1

            # Add name suggestion (dedupe)
            if name := customer.get("name"):
                name_lower = name.lower()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    if await self.add_suggestion(self.SUGGEST_NAME, name):
                        stats["names"] += 1

            if progress_callback and (i + 1) % 1000 == 0:
                progress_callback(i + 1, total)

        return stats

    # =========================================================================
    # Faceted Search (Aggregations)
    # =========================================================================

    async def facet_search(
        self,
        index: str,
        query: str,
        facet_field: str,
        limit: int = 10,
    ) -> FacetResult:
        """Get faceted counts for a field.

        Args:
            index: Index name
            query: Search query (use * for all)
            facet_field: Field to facet on
            limit: Max facet values

        Returns:
            FacetResult with field and value counts
        """
        try:
            # FT.AGGREGATE idx:customers "*"
            #   GROUPBY 1 @state REDUCE COUNT 0 AS count
            #   SORTBY 2 @count DESC
            #   LIMIT 0 10
            result = await self.redis.execute_command(
                "FT.AGGREGATE", index, query,
                "GROUPBY", "1", f"@{facet_field}",
                "REDUCE", "COUNT", "0", "AS", "count",
                "SORTBY", "2", "@count", "DESC",
                "LIMIT", "0", str(limit),
            )

            if not result or len(result) < 2:
                return FacetResult(field=facet_field, values=[])

            # Parse result: [total, [field, value, count, count_value], ...]
            values = []
            for i in range(1, len(result)):
                entry = result[i]
                if isinstance(entry, list) and len(entry) >= 4:
                    field_value = entry[1]
                    count = int(entry[3]) if entry[3] else 0
                    values.append((field_value, count))

            return FacetResult(field=facet_field, values=values)

        except Exception as e:
            logger.warning(f"Facet search failed: {e}")
            return FacetResult(field=facet_field, values=[])

    async def get_state_facets(self, query: str = "*", limit: int = 20) -> FacetResult:
        """Get customer counts by state."""
        return await self.facet_search(self.INDEX_CUSTOMERS, query, "state", limit)

    async def get_city_facets(self, query: str = "*", limit: int = 20) -> FacetResult:
        """Get customer counts by city."""
        return await self.facet_search(self.INDEX_CUSTOMERS, query, "city", limit)

    async def get_type_facets(self, query: str = "*", limit: int = 10) -> FacetResult:
        """Get customer counts by type."""
        return await self.facet_search(self.INDEX_CUSTOMERS, query, "customer_type", limit)

    async def get_status_facets(self, query: str = "*", limit: int = 10) -> FacetResult:
        """Get customer counts by status."""
        return await self.facet_search(self.INDEX_CUSTOMERS, query, "status", limit)

    # =========================================================================
    # Full-text Search with Highlighting
    # =========================================================================

    async def search_with_highlight(
        self,
        index: str,
        query: str,
        fields: list[str] | None = None,
        highlight_fields: list[str] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        """Search with result highlighting.

        Args:
            index: Index name
            query: Search query
            fields: Fields to return
            highlight_fields: Fields to highlight matches in
            limit: Max results
            offset: Pagination offset

        Returns:
            Dict with results and highlighted snippets
        """
        try:
            args = ["FT.SEARCH", index, query]

            # Add highlighting
            if highlight_fields:
                args.extend(["HIGHLIGHT", "FIELDS", str(len(highlight_fields))])
                args.extend(highlight_fields)
                args.extend(["TAGS", "<mark>", "</mark>"])

            # Limit/offset
            args.extend(["LIMIT", str(offset), str(limit)])

            result = await self.redis.execute_command(*args)

            if not result or result[0] == 0:
                return {"total": 0, "results": []}

            total = result[0]
            results = []

            # Parse results: [total, doc_id, [field, value, ...], doc_id, ...]
            i = 1
            while i < len(result):
                doc_id = result[i]
                i += 1

                if i >= len(result):
                    break

                fields_data = result[i]
                i += 1

                # Parse field pairs
                doc = {"id": doc_id}
                if isinstance(fields_data, list):
                    for j in range(0, len(fields_data), 2):
                        if j + 1 < len(fields_data):
                            doc[fields_data[j]] = fields_data[j + 1]

                results.append(doc)

            return {"total": total, "results": results}

        except Exception as e:
            logger.warning(f"Highlighted search failed: {e}")
            return {"total": 0, "results": [], "error": str(e)}

    # =========================================================================
    # Spell Correction
    # =========================================================================

    async def spell_check(
        self,
        index: str,
        query: str,
        distance: int = 1,
    ) -> list[dict]:
        """Get spelling suggestions for query terms.

        Args:
            index: Index name with dictionary
            query: Query to check
            distance: Levenshtein distance for suggestions

        Returns:
            List of term corrections
        """
        try:
            result = await self.redis.execute_command(
                "FT.SPELLCHECK", index, query,
                "DISTANCE", str(distance),
            )

            corrections = []
            for entry in result or []:
                if isinstance(entry, list) and len(entry) >= 3:
                    term = entry[1]
                    suggestions = []
                    for suggestion_pair in entry[2]:
                        if isinstance(suggestion_pair, list) and len(suggestion_pair) >= 2:
                            suggestions.append({
                                "score": float(suggestion_pair[0]) if suggestion_pair[0] else 0,
                                "suggestion": suggestion_pair[1],
                            })
                    corrections.append({
                        "term": term,
                        "suggestions": suggestions,
                    })

            return corrections

        except Exception as e:
            logger.debug(f"Spell check failed: {e}")
            return []

    # =========================================================================
    # Phonetic Search
    # =========================================================================

    async def create_phonetic_index(
        self,
        index: str,
        prefix: str,
        schema: list[tuple[str, str]],
    ) -> bool:
        """Create index with phonetic matching.

        Args:
            index: Index name
            prefix: Key prefix
            schema: List of (field_name, field_type) tuples

        Returns:
            True if created successfully
        """
        try:
            # Check if exists
            try:
                await self.redis.execute_command("FT.INFO", index)
                return True  # Already exists
            except Exception:
                pass

            # Build schema with PHONETIC for text fields
            args = [
                "FT.CREATE", index,
                "ON", "HASH",
                "PREFIX", "1", prefix,
                "SCHEMA",
            ]

            for field_name, field_type in schema:
                args.append(field_name)
                args.append(field_type.upper())
                if field_type.upper() == "TEXT":
                    args.extend(["PHONETIC", "dm:en"])  # Double Metaphone English

            await self.redis.execute_command(*args)
            logger.info(f"Created phonetic index: {index}")
            return True

        except Exception as e:
            if "Index already exists" in str(e):
                return True
            logger.warning(f"Failed to create phonetic index {index}: {e}")
            return False

    # =========================================================================
    # Index Management
    # =========================================================================

    async def get_index_info(self, index: str) -> dict:
        """Get index statistics."""
        try:
            info = await self.redis.execute_command("FT.INFO", index)

            # Parse info response
            info_dict = {}
            for i in range(0, len(info), 2):
                if i + 1 < len(info):
                    info_dict[info[i]] = info[i + 1]

            return {
                "index_name": index,
                "num_docs": info_dict.get("num_docs", 0),
                "num_terms": info_dict.get("num_terms", 0),
                "num_records": info_dict.get("num_records", 0),
                "inverted_sz_mb": info_dict.get("inverted_sz_mb", 0),
                "indexing": info_dict.get("indexing", 0),
            }

        except Exception as e:
            return {"error": str(e)}

    async def delete_suggestion_dict(self, dictionary: str) -> bool:
        """Delete a suggestion dictionary."""
        try:
            await self.redis.delete(dictionary)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete dictionary {dictionary}: {e}")
            return False
