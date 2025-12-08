"""Nova Classifier - Uses OpenSearch RAG to classify query intent and complexity."""

from dataclasses import dataclass
from enum import Enum

import boto3

from .vector_index import VectorIndex


class QueryComplexity(Enum):
    SIMPLE = "simple"      # Nova Micro - direct lookups, yes/no, counts
    MEDIUM = "medium"      # Nova Lite - filtering, comparisons, summaries
    COMPLEX = "complex"    # Nova Pro - analysis, trends, recommendations


class QueryIntent(Enum):
    LOOKUP = "lookup"           # Direct CRID lookup
    SEARCH = "search"           # Filter/search customers
    STATS = "stats"             # Aggregations/counts
    ANALYSIS = "analysis"       # Trends, patterns, insights
    COMPARISON = "comparison"   # Compare customers/states
    RECOMMENDATION = "recommendation"  # Suggestions based on data


@dataclass(slots=True)
class Classification:
    intent: QueryIntent
    complexity: QueryComplexity
    confidence: float
    rag_context: list[str]


class NovaClassifier:
    """Classifies queries using Nova + OpenSearch RAG context."""
    __slots__ = ("client", "model", "vector_index", "available")

    SYSTEM_PROMPT = """You are a query classifier. Analyze the user query and classify it.

Return ONLY a JSON object with these fields:
- intent: one of [lookup, search, stats, analysis, comparison, recommendation]
- complexity: one of [simple, medium, complex]
- confidence: float 0.0-1.0

Classification rules:
SIMPLE (Nova Micro): Direct lookups by ID, yes/no questions, single counts
- "Look up CRID-001" → lookup, simple
- "How many customers?" → stats, simple
- "Is there a customer named John?" → search, simple

MEDIUM (Nova Lite): Filtering with conditions, basic comparisons, summaries
- "Nevada customers who moved twice" → search, medium
- "Customers per state" → stats, medium
- "Compare Nevada vs California counts" → comparison, medium

COMPLEX (Nova Pro): Trends, patterns, recommendations, multi-step analysis
- "Which states have growing customer bases?" → analysis, complex
- "Recommend which customers to contact" → recommendation, complex
- "Analyze migration patterns" → analysis, complex

Return only valid JSON, no explanation."""

    def __init__(self, region: str, model: str, vector_index: VectorIndex):
        try:
            self.client = boto3.client("bedrock-runtime", region_name=region)
            self.model = model
            self.vector_index = vector_index
            self.available = True
        except Exception:
            self.available = False

    async def classify(self, query: str) -> Classification:
        """Classify query intent and complexity using RAG context."""
        # Get RAG context from vector index
        rag_docs = await self._get_rag_context(query)

        if not self.available:
            return self._fallback_classify(query, rag_docs)

        # Build prompt with RAG context
        context_str = "\n".join(f"- {doc}" for doc in rag_docs) if rag_docs else "No similar queries found."

        user_msg = f"""Similar queries in our system:
{context_str}

Classify this query: "{query}"

Return JSON only."""

        try:
            response = self.client.converse(
                modelId=self.model,
                system=[{"text": self.SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": user_msg}]}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 200}
            )

            result_text = response["output"]["message"]["content"][0]["text"]
            return self._parse_classification(result_text, rag_docs)

        except Exception:
            return self._fallback_classify(query, rag_docs)

    async def _get_rag_context(self, query: str, k: int = 5) -> list[str]:
        """Retrieve relevant context from OpenSearch."""
        if not self.vector_index.available:
            return []

        try:
            results = await self.vector_index.search(query, k)
            return [r.get("text", "") for r in results if r.get("text")]
        except Exception:
            return []

    def _parse_classification(self, text: str, rag_docs: list[str]) -> Classification:
        """Parse Nova's JSON response."""
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', text)
        if not json_match:
            return self._fallback_classify("", rag_docs)

        try:
            data = json.loads(json_match.group())
            return Classification(
                intent=QueryIntent(data.get("intent", "search")),
                complexity=QueryComplexity(data.get("complexity", "medium")),
                confidence=float(data.get("confidence", 0.5)),
                rag_context=rag_docs
            )
        except (json.JSONDecodeError, ValueError):
            return self._fallback_classify("", rag_docs)

    def _fallback_classify(self, query: str, rag_docs: list[str]) -> Classification:
        """Keyword-based fallback classification."""
        q = query.lower()

        # Intent detection
        if "crid" in q or "look up" in q or "find customer" in q:
            intent = QueryIntent.LOOKUP
        elif "how many" in q or "count" in q or "stats" in q or "per state" in q:
            intent = QueryIntent.STATS
        elif "compare" in q or "vs" in q or "versus" in q:
            intent = QueryIntent.COMPARISON
        elif "trend" in q or "pattern" in q or "analyze" in q or "analysis" in q:
            intent = QueryIntent.ANALYSIS
        elif "recommend" in q or "suggest" in q or "should" in q:
            intent = QueryIntent.RECOMMENDATION
        else:
            intent = QueryIntent.SEARCH

        # Complexity detection
        complex_words = ["trend", "pattern", "analyze", "analysis", "recommend", "predict", "insight", "why"]
        medium_words = ["compare", "filter", "between", "summary", "per state", "who moved", "which"]

        if any(w in q for w in complex_words):
            complexity = QueryComplexity.COMPLEX
        elif any(w in q for w in medium_words) or len(q.split()) > 8:
            complexity = QueryComplexity.MEDIUM
        else:
            complexity = QueryComplexity.SIMPLE

        return Classification(
            intent=intent,
            complexity=complexity,
            confidence=0.6,
            rag_context=rag_docs
        )
