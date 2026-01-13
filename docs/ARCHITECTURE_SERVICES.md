# ICDA Service Architecture

## Overview

ICDA (Intelligent Customer Data Access) uses a multi-tier architecture combining caching, vector search, and AI to deliver fast, accurate results. Each service plays a specific role, and together they create a system that is both performant and intelligent.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ICDA Service Stack                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Query: "101 turkey 22222"                                            │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌──────────┐ │
│   │   Redis     │────▶│ Redis Stack │────▶│ OpenSearch  │────▶│  Nova    │ │
│   │   Cache     │     │   Vector    │     │   Search    │     │   LLM    │ │
│   │   (L1)      │     │   (L2)      │     │   (L3)      │     │  (L4)    │ │
│   └─────────────┘     └─────────────┘     └─────────────┘     └──────────┘ │
│        ~1ms               ~3-5ms             ~20-50ms          ~200-500ms   │
│                                                                              │
│   ◄─────────────────── Increasing Latency ──────────────────────────────►   │
│   ◄─────────────────── Increasing Intelligence ─────────────────────────►   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Service Breakdown

### 1. Redis Cache (L1 - Instant Response)

**What it does:**
- Stores frequently accessed query results
- Caches completed addresses for instant retrieval
- Manages session state and conversation history
- Stores download tokens for paginated results

**Why we need it:**
- Eliminates redundant processing for repeated queries
- Sub-millisecond response times for cached data
- Reduces load on downstream services (OpenSearch, Nova)
- Cost savings by avoiding repeated LLM calls

**In ICDA:**
```
Query: "customers in Nevada"
  │
  ├─► Cache HIT  → Return cached result (~1ms)
  │
  └─► Cache MISS → Process query → Cache result → Return
```

**Configuration:**
```bash
REDIS_URL=redis://localhost:6379
CACHE_TTL=43200  # 12 hours
```

---

### 2. Redis Stack Vector Search (L2 - Fast Similarity)

**What it does:**
- Stores address embeddings in-memory for ultra-fast vector search
- Uses HNSW (Hierarchical Navigable Small World) algorithm
- Supports filtered vector search (by ZIP, state, etc.)
- Provides sub-5ms similarity lookups

**Why we need it:**
- **Speed**: 10x faster than OpenSearch for simple vector queries
- **Filtering**: Combine vector similarity with exact filters (ZIP code)
- **Hot path optimization**: Addresses queried frequently stay in RAM
- **Real-time**: Instant autocomplete and address completion

**In ICDA:**
```
Input: "101 turkey 22222"
  │
  ├─► Extract ZIP filter: 22222
  │
  ├─► Vector search with filter:
  │   "Find addresses similar to 'turkey' WHERE zip=22222"
  │
  └─► Results in ~3ms:
      1. 101 Turkey Run, Springfield, VA 22222 (0.92)
      2. 101 Turkey Trot Ln, Springfield, VA 22222 (0.87)
```

**Key Capability - Filtered Vector Search:**
```
Without ZIP filter: Search 500,000 addresses → 50ms
With ZIP filter:    Search 200 addresses    → 3ms
```

**Configuration:**
```bash
ENABLE_REDIS_VECTOR=true
COMPLETION_VECTOR_THRESHOLD=0.85
```

**Requirements:**
- Redis Stack (not standard Redis)
- RediSearch module for vector capabilities

---

### 3. OpenSearch (L3 - Deep Search)

**What it does:**
- Full-text search with BM25 ranking
- Vector search (kNN) for semantic similarity
- Complex filtering (nested, range, geo, boolean)
- Aggregations and analytics
- Fuzzy matching for typos

**Why we need it:**
- **Scale**: Handles millions of records efficiently
- **Persistence**: Data survives restarts (disk-based)
- **Rich queries**: Complex boolean logic, fuzzy matching
- **Analytics**: Aggregations, histograms, statistics
- **Hybrid search**: Combine text relevance + vector similarity

**In ICDA:**
```
Query: "inactive customers who moved from California to Texas"
  │
  ├─► Parse intent: status=INACTIVE, from_state=CA, to_state=TX
  │
  ├─► OpenSearch query:
  │   {
  │     "bool": {
  │       "must": [
  │         {"term": {"status": "INACTIVE"}},
  │         {"term": {"move_history.from_state": "CA"}},
  │         {"term": {"state": "TX"}}
  │       ]
  │     }
  │   }
  │
  └─► Results with full customer data, move history, analytics
```

**Capabilities Redis Stack Cannot Do:**
| Feature | OpenSearch | Redis Stack |
|---------|------------|-------------|
| Fuzzy text search | ✅ "turkye" → "turkey" | ❌ |
| Nested object queries | ✅ move_history[].state | ❌ |
| Range queries | ✅ last_move > 2020 | Limited |
| Aggregations | ✅ count by state | ❌ |
| Highlighting | ✅ show matched terms | ❌ |
| 100M+ records | ✅ disk-based | ❌ RAM limited |

**Configuration:**
```bash
OPENSEARCH_HOST=http://localhost:9200
OPENSEARCH_INDEX=icda-vectors
```

---

### 4. Amazon Titan Embeddings (Semantic Understanding)

**What it does:**
- Converts text into 1024-dimensional vectors
- Captures semantic meaning, not just keywords
- Powers both Redis Stack and OpenSearch vector search
- Understands synonyms, abbreviations, context

**Why we need it:**
- **Semantic search**: "turkey" matches "Turkey Run" (not just exact text)
- **Typo tolerance**: "turky" still finds "Turkey" addresses
- **Abbreviation handling**: "St" and "Street" are similar vectors
- **Context awareness**: "spring" in address vs "spring" season

**In ICDA:**
```
Text: "123 Main Street, Springfield"
  │
  ├─► Preprocess: "123 main street springfield"
  │
  ├─► Expand abbreviations: "123 main street springfield"
  │
  └─► Titan Embedding: [0.12, -0.34, 0.56, ...] (1024 dims)
```

**Why Not Just Text Search?**
```
Text search:  "turkey run"  → Only matches "turkey run" exactly
Vector search: "turkey run" → Matches:
  - "Turkey Run"       (exact)
  - "Turkey Trot"      (similar)
  - "Wild Turkey Ln"   (related)
  - "Turky Run"        (typo)
```

**Configuration:**
```bash
TITAN_EMBED_MODEL=amazon.titan-embed-text-v2:0
EMBED_DIMENSIONS=1024
```

---

### 5. Amazon Nova LLM (L4 - Intelligence)

**What it does:**
- Natural language understanding
- Query intent classification
- Intelligent reranking of search results
- Address completion with reasoning
- Conversational responses

**Why we need it:**
- **Ambiguity resolution**: When vector results are uncertain
- **Complex reasoning**: "Who moved the most times last year?"
- **Natural language**: Understands "high movers" = frequent relocators
- **Explanation**: Can explain why it chose a particular match

**In ICDA - Address Completion:**
```
Input: "101 turkey 22222"
Vector candidates:
  1. 101 Turkey Run, Springfield, VA 22222 (0.82)
  2. 101 Turkey Trot Ln, Springfield, VA 22222 (0.80)
  3. 1010 Turkey Creek Rd, Richmond, VA 22222 (0.71)

Nova reasoning:
  "101 matches exactly with candidates 1 and 2.
   'turkey' is partial street name.
   Candidate 1 'Turkey Run' is more common.
   Candidate 3 has wrong street number (1010 vs 101).

   Best match: 101 Turkey Run, Springfield, VA 22222
   Confidence: 0.95"
```

**In ICDA - Query Understanding:**
```
User: "Show me frequent movers in Vegas who are inactive"
  │
  ├─► Nova parses intent:
  │   - "frequent movers" → min_move_count: 3
  │   - "Vegas" → city: "Las Vegas" OR "North Las Vegas"
  │   - "inactive" → status: "INACTIVE"
  │
  └─► Generates search parameters automatically
```

**Model Routing:**
```
Simple queries  → Nova Micro  (fastest, cheapest)
Medium queries  → Nova Lite   (balanced)
Complex queries → Nova Pro    (most capable)
```

**Configuration:**
```bash
NOVA_MODEL=us.amazon.nova-micro-v1:0
NOVA_LITE_MODEL=us.amazon.nova-lite-v1:0
NOVA_PRO_MODEL=us.amazon.nova-pro-v1:0
MODEL_ROUTING_THRESHOLD=0.6
```

---

## How Services Complement Each Other

### Address Completion Pipeline

```
User types: "101 turkey 22222"
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: Redis Cache                                                        │
│ ────────────────────                                                        │
│ Check: Have we completed this exact input before?                          │
│                                                                             │
│ HIT  → Return cached: "101 Turkey Run, Springfield, VA 22222" (~1ms)       │
│ MISS → Continue to Layer 2                                                  │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: Redis Stack Vector Search                                          │
│ ──────────────────────────────────                                          │
│ 1. Extract ZIP: 22222 (for filtering)                                       │
│ 2. Generate Titan embedding for "101 turkey"                                │
│ 3. Vector search: KNN with ZIP filter                                       │
│                                                                             │
│ Results (in 3ms):                                                           │
│   • 101 Turkey Run, Springfield, VA 22222 (score: 0.89)                    │
│   • 101 Turkey Trot Ln, Springfield, VA 22222 (score: 0.84)                │
│                                                                             │
│ Score >= 0.85? → Return top result, cache it (~5ms total)                  │
│ Score < 0.85?  → Continue to Layer 3                                        │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: OpenSearch (if needed)                                             │
│ ───────────────────────────────                                             │
│ Fallback for:                                                               │
│   • Redis Stack not available                                               │
│   • Complex queries needing fuzzy text match                                │
│   • Queries without clear ZIP filter                                        │
│                                                                             │
│ Hybrid search: Vector similarity + BM25 text relevance                      │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: Nova LLM Reranking                                                 │
│ ───────────────────────────                                                 │
│ When: Vector scores are close or ambiguous                                  │
│                                                                             │
│ Input to Nova:                                                              │
│   "User entered: 101 turkey 22222                                          │
│    Candidates:                                                              │
│    1. 101 Turkey Run, Springfield, VA 22222 (0.82)                         │
│    2. 101 Turkey Trot Ln, Springfield, VA 22222 (0.80)                     │
│    Select best match."                                                      │
│                                                                             │
│ Nova response:                                                              │
│   {"best_match": "101 Turkey Run...", "confidence": 0.94}                  │
│                                                                             │
│ Cache result for future queries                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Customer Query Pipeline

```
User asks: "How many inactive customers moved from California last year?"
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: Cache Check                                                        │
│ ────────────────────                                                        │
│ Check for identical recent query → HIT/MISS                                │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: Nova Intent Parsing                                                │
│ ────────────────────────────                                                │
│ Extract:                                                                    │
│   • status: INACTIVE                                                        │
│   • from_state: CA                                                          │
│   • date_range: 2025-01-01 to 2025-12-31                                   │
│   • aggregation: COUNT                                                      │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: OpenSearch Query Execution                                         │
│ ───────────────────────────────────                                         │
│ {                                                                           │
│   "query": {                                                                │
│     "bool": {                                                               │
│       "must": [                                                             │
│         {"term": {"status": "INACTIVE"}},                                  │
│         {"nested": {                                                        │
│           "path": "move_history",                                           │
│           "query": {                                                        │
│             "bool": {                                                       │
│               "must": [                                                     │
│                 {"term": {"move_history.from_state": "CA"}},               │
│                 {"range": {"move_history.move_date": {"gte": "2025-01"}}}  │
│               ]                                                             │
│             }                                                               │
│           }                                                                 │
│         }}                                                                  │
│       ]                                                                     │
│     }                                                                       │
│   },                                                                        │
│   "aggs": {"total": {"value_count": {"field": "crid"}}}                    │
│ }                                                                           │
└────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: Nova Response Generation                                           │
│ ─────────────────────────────────                                           │
│ "There are 1,247 inactive customers who moved from California in 2025.     │
│  Top destination states: Texas (412), Nevada (298), Arizona (187)."        │
│                                                                             │
│ Cache response for future identical queries                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Comparison

### Address Completion Latency

| Scenario | Without Optimization | With Full Stack |
|----------|---------------------|-----------------|
| Cached query | N/A | **~1ms** |
| Vector match (high confidence) | ~200ms (Nova only) | **~5ms** |
| Vector match (needs rerank) | ~500ms | **~250ms** |
| Cold start (no cache) | ~800ms | **~300ms** |

### Cost Reduction

| Component | Cost per 1K Requests | Notes |
|-----------|---------------------|-------|
| Redis Cache hit | ~$0.00 | Free (in-memory) |
| Redis Stack vector | ~$0.001 | Titan embedding only |
| OpenSearch query | ~$0.002 | Infrastructure cost |
| Nova Lite call | ~$0.02 | Per rerank request |

**With caching and vector threshold:**
- 70% of requests → Cache hit ($0)
- 25% of requests → Vector only ($0.001)
- 5% of requests → Nova rerank ($0.02)

**Average cost: ~$0.0015 per request** (vs $0.02 without optimization)

---

## Failure Modes & Graceful Degradation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Graceful Degradation                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Redis Down?                                                                 │
│  └─► In-memory cache fallback                                               │
│      └─► Continue to vector search                                          │
│                                                                              │
│  Redis Stack (vector) unavailable?                                          │
│  └─► Skip to OpenSearch                                                     │
│      └─► Full functionality, slightly slower                                │
│                                                                              │
│  OpenSearch Down?                                                            │
│  └─► FATAL - Required service                                               │
│      └─► Return error, alert ops                                            │
│                                                                              │
│  AWS/Nova unavailable?                                                       │
│  └─► LITE MODE                                                              │
│      └─► Vector search still works                                          │
│      └─► No intelligent reranking                                           │
│      └─► Return best vector match                                           │
│                                                                              │
│  Titan Embeddings unavailable?                                               │
│  └─► LITE MODE                                                              │
│      └─► Fall back to keyword search                                        │
│      └─► BM25 text matching in OpenSearch                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Service | Role | Speed | Capability | When Used |
|---------|------|-------|------------|-----------|
| **Redis Cache** | L1 Cache | ~1ms | Exact match | Every request (first check) |
| **Redis Stack** | L2 Vector | ~3-5ms | Fast similarity + filter | Address completion |
| **OpenSearch** | L3 Search | ~20-50ms | Full search + analytics | Complex queries |
| **Titan** | Embeddings | ~50ms | Semantic understanding | Vector indexing |
| **Nova** | L4 Intelligence | ~200-500ms | Reasoning + NLU | Ambiguous cases |

**The key insight:** Each layer handles what it does best. Redis is fastest but simplest. Nova is smartest but slowest. By layering them correctly, we get both speed AND intelligence.

```
Fast ◄──────────────────────────────────────────────► Smart
     Redis    Redis Stack    OpenSearch    Nova
     Cache    Vector         Search        LLM

     "Did we    "What's      "Search      "Think about
      see this   similar?"    everything"  this carefully"
      before?"
```
