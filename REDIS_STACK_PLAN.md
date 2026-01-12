# Redis Stack Comprehensive Integration Plan

## Executive Summary
This plan integrates 8 Redis Stack modules into ICDA to enhance capabilities without reducing existing functionality. Each module addresses specific use cases while maintaining graceful degradation.

## Current State Analysis

### Existing Redis Usage
1. **RedisCache** (`icda/cache.py`) - Basic caching with TTL
2. **RedisAddressIndex** (`icda/indexes/redis_vector_index.py`) - Vector search for addresses using RediSearch

### Existing Capabilities (MUST PRESERVE)
- Query pipeline (8-agent orchestration)
- Address verification pipeline
- Knowledge base RAG
- Customer search (semantic + keyword)
- Autocomplete (prefix + fuzzy)
- Session management
- Progress tracking
- Health checks

---

## Module Integration Plan

### 1. RediSearch Enhancements
**Status**: Partially implemented
**File**: `icda/indexes/redis_search_enhanced.py`

#### New Features
- Full-text search with fuzzy matching (SOUNDEX, phonetic)
- Autocomplete suggestions (FT.SUGADD/FT.SUGGET)
- Faceted search (state, city, customer_type aggregations)
- Highlighting matched terms
- Spell correction suggestions
- Synonym support

#### API Endpoints
```
GET /api/search/suggest?q=<prefix>&field=<field>
GET /api/search/facets?q=<query>
GET /api/search/spell?q=<query>
```

---

### 2. RedisJSON
**Status**: New
**File**: `icda/redis_stack/redis_json.py`

#### Use Cases
- Store customer records as JSON documents
- Nested objects for customer history
- Atomic partial updates with JSONPath
- Efficient memory with shared string tables

#### Implementation
```python
# Store customer with history
JSON.SET customer:CRID-001 $ '{"name":"...", "history":[...]}'

# Update single field atomically
JSON.SET customer:CRID-001 $.status '"INACTIVE"'

# Append to history array
JSON.ARRAPPEND customer:CRID-001 $.history '{"date":"2024-01-01", "action":"moved"}'
```

#### API Endpoints
```
GET  /api/customers/{crid}/json
PUT  /api/customers/{crid}/json
PATCH /api/customers/{crid}/json (partial update)
```

---

### 3. RedisTimeSeries
**Status**: New
**File**: `icda/redis_stack/redis_timeseries.py`

#### Metrics to Track
- Query volume (queries/min)
- Response time (p50, p95, p99)
- Cache hit rate
- Agent performance (per-agent latency)
- Error rates
- Indexing throughput

#### Key Naming Convention
```
ts:queries:volume          # Query count per minute
ts:queries:latency:p50     # Median latency
ts:queries:latency:p99     # P99 latency
ts:cache:hit_rate          # Cache hit percentage
ts:agent:{name}:latency    # Per-agent timing
ts:errors:count            # Error rate
ts:index:throughput        # Docs indexed per second
```

#### API Endpoints
```
GET /api/analytics/queries?range=1h|24h|7d
GET /api/analytics/performance?range=1h|24h|7d
GET /api/analytics/agents?range=1h|24h|7d
```

---

### 4. RedisBloom
**Status**: New
**File**: `icda/redis_stack/redis_bloom.py`

#### Data Structures
1. **Bloom Filter**: Query deduplication
   - `bf:seen_queries` - Skip duplicate processing

2. **Cuckoo Filter**: Document deduplication
   - `cf:doc_hashes` - Dedupe knowledge chunks

3. **Count-Min Sketch**: Query frequency
   - `cms:query_freq` - Track popular queries

4. **Top-K**: Most frequent queries
   - `topk:queries:1h` - Trending queries

5. **HyperLogLog**: Unique visitors
   - `hll:users:daily` - Unique session count

#### API Endpoints
```
GET /api/analytics/trending?k=10
GET /api/analytics/unique-users?range=24h
POST /api/bloom/check-seen?query=<query>
```

---

### 5. Redis Pub/Sub
**Status**: New
**File**: `icda/redis_stack/redis_pubsub.py`

#### Channels
```
icda:index:progress     # Indexing progress updates
icda:cache:invalidate   # Cache invalidation events
icda:health:status      # Health status broadcasts
icda:admin:alerts       # Admin notifications
```

#### SSE Endpoint
```
GET /api/events/stream  # Server-Sent Events endpoint
```

#### Frontend Integration
```typescript
// React hook for real-time updates
const { progress, status } = useRedisEvents('index:progress');
```

---

### 6. Redis Streams
**Status**: New
**File**: `icda/redis_stack/redis_streams.py`

#### Streams
```
stream:customer:changes     # Customer CRUD events
stream:query:audit          # Full query audit trail
stream:agent:execution      # Agent pipeline trace
stream:errors               # Error stream for alerting
```

#### Event Schema
```json
{
  "event_id": "auto",
  "timestamp": "ISO8601",
  "type": "customer_created|query_executed|agent_completed",
  "actor": "user_session_id",
  "data": { ... },
  "trace_id": "uuid"
}
```

#### Consumer Groups
```
XGROUP CREATE stream:query:audit analytics-group $ MKSTREAM
XGROUP CREATE stream:errors alerting-group $ MKSTREAM
```

#### API Endpoints
```
GET /api/audit/queries?from=<timestamp>&limit=100
GET /api/audit/changes?entity=customer&limit=100
GET /api/audit/trace/{trace_id}
```

---

## Architecture

### Unified Redis Stack Client
**File**: `icda/redis_stack/client.py`

```python
class RedisStackClient:
    """Unified client for all Redis Stack modules."""

    def __init__(self):
        self.redis: Redis = None

        # Module availability flags
        self.search_available = False
        self.json_available = False
        self.timeseries_available = False
        self.bloom_available = False
        self.graph_available = False  # Future

        # Module wrappers
        self.search: RedisSearchEnhanced = None
        self.json: RedisJSONWrapper = None
        self.timeseries: RedisTimeSeriesWrapper = None
        self.bloom: RedisBloomWrapper = None
        self.pubsub: RedisPubSubManager = None
        self.streams: RedisStreamsManager = None

    async def connect(self, url: str) -> dict[str, bool]:
        """Connect and detect available modules."""

    async def health_check(self) -> dict:
        """Check all module health."""
```

### Initialization Order
```
1. Redis connection
2. Module detection (FT, JSON, TS, BF, etc.)
3. Index creation (if missing)
4. Background task registration
5. Health check setup
```

### Graceful Degradation
```
If module unavailable:
- RediSearch → Fall back to keyword search
- RedisJSON → Fall back to in-memory dict
- RedisTimeSeries → Fall back to no metrics (log only)
- RedisBloom → Fall back to no deduplication
- Pub/Sub → Fall back to polling
- Streams → Fall back to log file audit
```

---

## Integration Points

### main.py Changes
```python
# Add to lifespan
from icda.redis_stack.client import RedisStackClient

_redis_stack: RedisStackClient = None

async def lifespan(app):
    global _redis_stack

    # Initialize unified client
    _redis_stack = RedisStackClient()
    modules = await _redis_stack.connect(cfg.REDIS_URL)
    logger.info(f"Redis Stack modules: {modules}")

    # Register background tasks
    if _redis_stack.timeseries_available:
        asyncio.create_task(_redis_stack.timeseries.aggregate_loop())

    # Start event publisher
    if _redis_stack.pubsub:
        asyncio.create_task(_redis_stack.pubsub.health_broadcast_loop())
```

### Query Pipeline Integration
```python
# In query_orchestrator.py
async def process_query(query: str):
    trace_id = str(uuid.uuid4())
    start = time.time()

    # Check bloom filter for duplicate
    if redis_stack.bloom_available:
        if await redis_stack.bloom.query_seen(query):
            # Return cached response faster
            pass

    # Process query...
    result = await pipeline.execute(query)

    # Record metrics
    if redis_stack.timeseries_available:
        latency = time.time() - start
        await redis_stack.timeseries.record_query(latency)

    # Audit to stream
    if redis_stack.streams_available:
        await redis_stack.streams.add_query_event(query, result, trace_id)

    return result
```

---

## File Structure

```
icda/redis_stack/
├── __init__.py
├── client.py              # Unified RedisStackClient
├── redis_search.py        # RediSearch enhancements
├── redis_json.py          # RedisJSON wrapper
├── redis_timeseries.py    # RedisTimeSeries metrics
├── redis_bloom.py         # RedisBloom filters
├── redis_pubsub.py        # Pub/Sub manager
├── redis_streams.py       # Streams event sourcing
└── models.py              # Shared Pydantic models

tests/redis_stack/
├── test_client.py
├── test_search.py
├── test_json.py
├── test_timeseries.py
├── test_bloom.py
├── test_pubsub.py
└── test_streams.py
```

---

## Enforcer Validation Checklist

### Capability Preservation
- [ ] Query pipeline still works in LITE mode
- [ ] Address verification unchanged
- [ ] Knowledge RAG unchanged
- [ ] Customer search unchanged
- [ ] Autocomplete unchanged
- [ ] Background indexing continues
- [ ] Health checks work
- [ ] Session management works

### Non-Duplication Rules
- [ ] No duplicate customer storage (JSON OR dict, not both)
- [ ] No duplicate metrics (TimeSeries OR log, not both active)
- [ ] No duplicate caching logic
- [ ] Single source of truth for each data type

### Performance Requirements
- [ ] Startup time < 30 seconds
- [ ] Query latency < 500ms p95
- [ ] Index operations don't block queries
- [ ] Memory usage reasonable

---

## Implementation Order

1. **Phase 1: Foundation** (PR 1)
   - Create `icda/redis_stack/` directory
   - Implement `RedisStackClient` with module detection
   - Add graceful degradation framework

2. **Phase 2: Analytics** (PR 2)
   - Implement RedisTimeSeries wrapper
   - Add metrics recording to query pipeline
   - Create analytics endpoints

3. **Phase 3: Search Enhancements** (PR 3)
   - Enhance RediSearch with suggestions/facets
   - Add highlighting and spell correction
   - Update search endpoints

4. **Phase 4: Bloom Filters** (PR 4)
   - Implement RedisBloom wrapper
   - Add query deduplication
   - Add trending queries tracking

5. **Phase 5: Real-time** (PR 5)
   - Implement Pub/Sub manager
   - Add SSE endpoint
   - Frontend integration

6. **Phase 6: Audit** (PR 6)
   - Implement Streams manager
   - Add audit endpoints
   - Consumer group setup

7. **Phase 7: JSON Storage** (PR 7)
   - Implement RedisJSON wrapper
   - Migrate customer storage (optional)
   - Add partial update endpoints

8. **Phase 8: Tests & Enforcer** (PR 8)
   - Comprehensive test suite
   - Enforcer validation
   - Documentation

---

## Background Indexing Preservation

**Critical**: Background indexing MUST continue working throughout implementation.

### Current Background Tasks
```python
# In main.py
_progress_tracker: ProgressTracker  # Tracks indexing progress
_knowledge_watcher: KnowledgeWatcher  # Watches /knowledge folder

# Background indexing triggered by:
# 1. POST /api/admin/index/reindex
# 2. File changes in /knowledge folder
```

### Integration with Redis Stack
```python
# Publish progress to Pub/Sub
async def index_with_progress(customers: list):
    total = len(customers)
    for i, batch in enumerate(batches):
        await index_batch(batch)

        # Publish progress
        await redis_stack.pubsub.publish('index:progress', {
            'indexed': i * batch_size,
            'total': total,
            'percent': round((i * batch_size / total) * 100)
        })

        # Record throughput
        await redis_stack.timeseries.record_index_throughput(len(batch))
```

---

## Notes for Implementation

1. **Module Detection**: Use `MODULE LIST` command to detect available modules
2. **Timeout Protection**: All Redis operations must have timeouts (5s default)
3. **Error Handling**: Catch all Redis errors, log, and degrade gracefully
4. **Testing**: Each module needs unit tests + integration tests
5. **Documentation**: Update API docs for new endpoints

---

*Generated by Redis Stack Integration Planning Agents*
*Date: 2026-01-08*
