# Enabling Long-Term Memory in Amazon Bedrock AgentCore

## Overview

Long-term memory extracts and retains information across sessions, enabling agents to remember user preferences, facts, and conversation summaries beyond individual sessions.

## Methods to Enable Long-Term Memory

Long-term memory can be enabled in two ways:
1. When creating a new AgentCore Memory
2. By updating an existing AgentCore Memory resource

## Method 1: Create New Memory with Long-Term Strategies

### CLI Example

```bash
agentcore memory create PersonalizedShoppingAgentMemory \
   --region us-west-2 \
   --strategies '[{"userPreferenceMemoryStrategy": {"name": "UserShoppingPreferences", "namespaces": ["/users/{actorId}/preferences"]}}]' \
   --wait
```

### Python Example

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import UserPreferenceStrategy

# Create memory manager
memory_manager = MemoryManager(region_name="us-west-2")

# Create memory resource with user preference strategy
memory = memory_manager.get_or_create_memory(
    name="PersonalizedShoppingAgentMemory",
    strategies=[
        UserPreferenceStrategy(
            name="UserShoppingPreferences",
            namespaces=["/users/{actorId}/preferences"]
        )
    ]
)

memory_id = memory.get('id')
print(f"Memory resource is now ACTIVE with ID: {memory_id}")
```

**Important:** You must wait for the AgentCore Memory status to become `ACTIVE` before using it.

## Method 2: Add Strategies to Existing Memory

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SummaryStrategy

# Create memory manager
memory_manager = MemoryManager(region_name="us-west-2")

# Update existing memory with new strategy
memory_id = "your-existing-memory-id"

summaryStrategy = SummaryStrategy(
    name="SessionSummarizer",
    description="Summarizes conversation sessions for context",
    namespaces=["/summaries/{actorId}/{sessionId}"]
)

memory = memory_manager.update_memory_strategies(
    memory_id=memory_id,
    add_strategies=[summaryStrategy]
)

# Validate strategy was added
memory_strategies = memory_manager.get_memory_strategies(memoryId=memory_id)
print(f"Memory strategies for memoryID: {memory_id} are: {memory_strategies}")
```

## Supported Strategies

### 1. SemanticStrategy
Extracts facts and information from conversations.

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SemanticStrategy

semantic = SemanticStrategy(
    name="semanticLongTermMemory",
    namespaces=['/strategies/{memoryStrategyId}/actors/{actorId}']
)
```

**Use Cases:**
- Extract customer order information
- Remember discussed topics
- Store factual information mentioned in conversations

### 2. UserPreferenceStrategy
Captures user preferences automatically.

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import UserPreferenceStrategy

preferences = UserPreferenceStrategy(
    name="UserShoppingPreferences",
    namespaces=["/users/{actorId}/preferences"]
)
```

**Use Cases:**
- "I prefer Python over JavaScript"
- "I like morning appointments"
- "Send me email notifications"

### 3. SummaryStrategy
Summarizes conversation sessions for context.

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SummaryStrategy

summary = SummaryStrategy(
    name="SessionSummarizer",
    description="Summarizes conversation sessions",
    namespaces=["/summaries/{actorId}/{sessionId}"]
)
```

**Use Cases:**
- End-of-session summaries
- Conversation digests
- Context handoffs between agents

## Multiple Strategies

You can combine multiple strategies:

```python
memory = memory_manager.get_or_create_memory(
    name="FullFeaturedMemory",
    strategies=[
        SemanticStrategy(
            name="factExtractor",
            namespaces=['/facts/{actorId}']
        ),
        UserPreferenceStrategy(
            name="userPrefs",
            namespaces=['/preferences/{actorId}']
        ),
        SummaryStrategy(
            name="sessionSummary",
            namespaces=['/summaries/{actorId}/{sessionId}']
        )
    ]
)
```

## Important Notes

1. **Async Processing**: Long-term memory extraction happens asynchronously (5-10 seconds after conversation events)

2. **Records Only After Strategy Active**: Long-term memory records are only extracted from conversational events stored AFTER a strategy becomes `ACTIVE`. Conversations stored before a strategy is added will NOT be processed.

3. **Retention Periods**:
   - Short-term memory: 7 days default
   - Long-term memory: 30 days default

4. **Strategy Status**: Use `get_memory_strategies()` to check if strategies are active before writing events.

## Querying Long-Term Memory

### List Records
```python
records = session.list_long_term_memory_records(
    namespace_prefix="/preferences/"
)
```

### Semantic Search
```python
results = session.search_long_term_memories(
    query="What are the user's language preferences?",
    namespace_prefix="/preferences/",
    top_k=5
)
```

## Strategy Management

### Modify Strategy
```python
memory_manager.modify_strategy(
    memory_id=memory_id,
    strategy_name="UserShoppingPreferences",
    new_description="Updated description"
)
```

### Delete Strategy
```python
memory_manager.delete_strategy(
    memory_id=memory_id,
    strategy_name="UserShoppingPreferences"
)
```

## Best Practices

1. **Use Specific Namespaces**: Design namespace patterns that match your data organization needs
2. **Wait for Active Status**: Always check strategy status before writing events
3. **Test Extraction Delay**: Account for 5-10 second async processing in your tests
4. **Clean Up Test Resources**: Delete test memories to avoid accumulating unused resources
