# Memory Fix Reference Guide

## Quick Reference for Common Memory Issues

This document provides quick solutions for common AgentCore Memory issues.

---

## Issue: Memory Not Persisting Across Sessions

### Symptoms
- Agent forgets information between sessions
- Long-term memory queries return empty results

### Root Causes & Solutions

#### 1. No Long-Term Strategy Configured
```python
# WRONG: Short-term only (empty strategies)
memory = memory_manager.get_or_create_memory(
    name="MyMemory",
    strategies=[]  # No long-term extraction!
)

# CORRECT: Add semantic strategy for long-term storage
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SemanticStrategy

memory = memory_manager.get_or_create_memory(
    name="MyMemory",
    strategies=[
        SemanticStrategy(
            name="factExtraction",
            namespaces=['/facts/{actorId}']
        )
    ]
)
```

#### 2. Strategy Not Active
```python
# Check strategy status before using
strategies = memory_manager.get_memory_strategies(memoryId=memory_id)
for s in strategies:
    if s['status'] != 'ACTIVE':
        print(f"WARNING: Strategy {s['name']} is {s['status']}")
        # Wait or abort
```

#### 3. Not Waiting for Extraction
```python
# WRONG: Query immediately after adding turns
session.add_turns(messages=[...])
records = session.list_long_term_memory_records("/")  # Empty!

# CORRECT: Wait for async extraction (5-10 seconds)
session.add_turns(messages=[...])
import time
time.sleep(10)
records = session.list_long_term_memory_records("/")
```

---

## Issue: Semantic Search Returns No Results

### Symptoms
- `search_long_term_memories()` returns empty list
- Records exist but aren't found by query

### Root Causes & Solutions

#### 1. Wrong Namespace Prefix
```python
# WRONG: Searching wrong namespace
results = session.search_long_term_memories(
    query="user preferences",
    namespace_prefix="/wrong-namespace/"  # No records here!
)

# CORRECT: Use matching namespace or root
results = session.search_long_term_memories(
    query="user preferences",
    namespace_prefix="/"  # Search all namespaces
)
```

#### 2. Records Haven't Been Extracted Yet
```python
# WRONG: Query before extraction completes
session.add_turns(messages=[ConversationalMessage("I prefer Python", MessageRole.USER)])
results = session.search_long_term_memories("programming language")  # Empty!

# CORRECT: Wait for extraction
session.add_turns(messages=[ConversationalMessage("I prefer Python", MessageRole.USER)])
time.sleep(10)  # Wait for async extraction
results = session.search_long_term_memories("programming language")
```

#### 3. Query Too Specific/Different
```python
# Query semantics matter - try broader terms
results = session.search_long_term_memories(
    query="favorite programming language",  # May not match "I prefer Python"
    top_k=10  # Increase results
)
```

---

## Issue: Session Data Not Loading

### Symptoms
- `get_last_k_turns()` returns empty
- Previous conversation not available

### Root Causes & Solutions

#### 1. Wrong Session/Actor ID
```python
# WRONG: Different session ID
session1 = session_manager.create_memory_session(
    actor_id="user123",
    session_id="session-001"
)
session1.add_turns(messages=[...])

# Later, using different session ID
session2 = session_manager.create_memory_session(
    actor_id="user123",
    session_id="session-002"  # Different session!
)
turns = session2.get_last_k_turns(k=10)  # Empty!

# CORRECT: Use same session ID
session2 = session_manager.create_memory_session(
    actor_id="user123",
    session_id="session-001"  # Same session
)
turns = session2.get_last_k_turns(k=10)  # Has data!
```

#### 2. Wrong Memory ID
```python
# Verify memory ID is correct
print(f"Using memory: {memory_id}")
memory = memory_manager.get_memory(memory_id)
print(f"Memory status: {memory.get('status')}")
```

---

## Issue: Memory Creation Fails/Timeouts

### Symptoms
- `create_memory` hangs or times out
- Status stuck on CREATING

### Root Causes & Solutions

#### 1. Use Proper Waiting
```python
# WRONG: No wait for completion
memory = memory_manager.get_or_create_memory(name="MyMemory", strategies=[...])
# Status might still be CREATING

# CORRECT: Use create_and_wait or poll status
memory = memory_manager.create_memory_and_wait(
    name="MyMemory",
    strategies=[SemanticStrategy(...)],
    timeout=300  # 5 minute timeout
)
# Guaranteed ACTIVE status
```

#### 2. Check for Existing Memory with Same Name
```python
# Memory names must be unique
try:
    memory = memory_manager.get_or_create_memory(name="MyMemory", strategies=[...])
except Exception as e:
    if "ConflictException" in str(e):
        # Memory already exists - get it instead
        memories = memory_manager.list_memories()
        memory = next(m for m in memories if m['name'] == "MyMemory")
```

---

## Issue: Preferences Not Being Captured

### Symptoms
- User states preferences but they're not stored
- `get_user_preferences()` returns empty

### Root Causes & Solutions

#### 1. No UserPreferenceStrategy
```python
# WRONG: Only semantic strategy
memory = memory_manager.get_or_create_memory(
    name="MyMemory",
    strategies=[SemanticStrategy(...)]  # No preference extraction!
)

# CORRECT: Add preference strategy
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import UserPreferenceStrategy

memory = memory_manager.get_or_create_memory(
    name="MyMemory",
    strategies=[
        SemanticStrategy(name="facts", namespaces=['/facts/{actorId}']),
        UserPreferenceStrategy(
            name="preferences",
            namespaces=['/preferences/{actorId}']
        )
    ]
)
```

#### 2. Preference Not Explicit Enough
```python
# The extraction model needs clear preference signals
# WEAK: "Python is nice"
# STRONG: "I prefer Python for data science projects"
```

---

## Issue: Gateway Tools Not Accessible

### Symptoms
- Agent can't invoke gateway tools
- Tool discovery fails

### Root Causes & Solutions

#### 1. Gateway Not Connected
```python
# Verify gateway connection
async with ClientSession(
    url=gateway_config["gateway_url"],
    headers={"Authorization": f"Bearer {gateway_config['access_token']}"}
) as session:
    try:
        tools = await session.list_tools()
        print(f"Found {len(tools)} tools")
    except Exception as e:
        print(f"Gateway connection failed: {e}")
```

#### 2. Token Expired
```python
# Refresh token if needed
token = gateway_manager.get_access_token(
    gateway_id=gateway_id,
    client_id=client_id,
    client_secret=client_secret
)
```

---

## Diagnostic Checklist

When debugging memory issues, check these in order:

1. **Memory Resource**
   ```python
   memory = memory_manager.get_memory(memory_id)
   print(f"Status: {memory.get('status')}")  # Should be ACTIVE
   ```

2. **Strategy Status**
   ```python
   strategies = memory_manager.get_memory_strategies(memoryId=memory_id)
   for s in strategies:
       print(f"{s['name']}: {s['status']}")  # All should be ACTIVE
   ```

3. **Session Creation**
   ```python
   session = session_manager.create_memory_session(
       actor_id="test-user",
       session_id="test-session"
   )
   print(f"Session created: {session}")
   ```

4. **Short-Term Memory**
   ```python
   # Add a test message
   session.add_turns(messages=[
       ConversationalMessage("Test message", MessageRole.USER)
   ])
   turns = session.get_last_k_turns(k=5)
   print(f"Turns found: {len(turns)}")  # Should be >= 1
   ```

5. **Long-Term Memory (after waiting)**
   ```python
   time.sleep(10)
   records = session.list_long_term_memory_records("/")
   print(f"Records found: {len(records)}")
   ```

6. **Semantic Search**
   ```python
   results = session.search_long_term_memories("test", top_k=5)
   print(f"Search results: {len(results)}")
   ```

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| No long-term memory | Add `SemanticStrategy` to memory |
| Empty search results | Wait 10 seconds after adding turns |
| Preferences not saved | Add `UserPreferenceStrategy` |
| Session not found | Verify actor_id and session_id |
| Memory creation timeout | Use `create_memory_and_wait()` |
| Gateway tools missing | Check token and connection |
| Wrong namespace | Use "/" for root-level search |

---

## Related Documentation

- [Getting Started](./memory/getting-started.md)
- [Long-Term Memory](./memory/long-term-memory.md)
- [Memory Types](./memory/memory-types.md)
- [Strategies](./memory/strategies.md)
- [Semantic Search Example](./examples/semantic-search.md)
- [Session Management](./examples/session-management.md)
- [MemoryManager Reference](./sdk/memory-manager.md)
- [SessionManager Reference](./sdk/session-manager.md)
