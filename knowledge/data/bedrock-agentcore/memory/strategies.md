# Memory Strategies in Amazon Bedrock AgentCore

## Overview

Memory strategies define how AgentCore extracts and stores information from conversations into long-term memory. Each strategy type focuses on a specific aspect of conversation understanding.

## Available Strategy Types

### 1. SemanticStrategy

Extracts facts and contextual information from conversations using semantic understanding.

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SemanticStrategy

semantic = SemanticStrategy(
    name="semanticLongTermMemory",
    namespaces=['/strategies/{memoryStrategyId}/actors/{actorId}']
)
```

**What It Extracts:**
- Factual statements ("My order number is #35476")
- Contextual information ("I'm calling about a delivery issue")
- Entity mentions (names, dates, numbers)
- Topic summaries

**Example Conversation:**
```
User: "Hi, I'm John. My order #12345 hasn't arrived yet. I ordered it on December 15th."
```

**Extracted Records:**
- User name: John
- Order number: #12345
- Order date: December 15th
- Issue type: Delivery problem

### 2. UserPreferenceStrategy

Automatically identifies and captures user preferences from natural conversation.

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import UserPreferenceStrategy

preferences = UserPreferenceStrategy(
    name="UserShoppingPreferences",
    namespaces=["/users/{actorId}/preferences"]
)
```

**What It Extracts:**
- Explicit preferences ("I prefer email over phone")
- Implicit preferences (inferred from behavior)
- Settings and configurations
- Personal choices

**Example Conversation:**
```
User: "I prefer Python for data science projects. Please send notifications in the morning."
```

**Extracted Records:**
- Programming language preference: Python
- Notification timing: Morning
- Domain interest: Data science

### 3. SummaryStrategy

Creates condensed summaries of conversation sessions.

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SummaryStrategy

summary = SummaryStrategy(
    name="SessionSummarizer",
    description="Summarizes conversation sessions for context",
    namespaces=["/summaries/{actorId}/{sessionId}"]
)
```

**What It Creates:**
- Session summaries
- Key points discussed
- Action items identified
- Resolution status

**Example Output:**
```
Session Summary: Customer John reported missing order #12345 from December 15th.
Agent confirmed delivery delay due to weather. Estimated delivery: December 22nd.
Customer accepted resolution.
```

## Strategy Configuration

### Creating Multiple Strategies

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import (
    SemanticStrategy,
    UserPreferenceStrategy,
    SummaryStrategy
)

memory_manager = MemoryManager(region_name="us-west-2")

memory = memory_manager.get_or_create_memory(
    name="ComprehensiveAgentMemory",
    description="Full-featured memory with all strategy types",
    strategies=[
        SemanticStrategy(
            name="factExtractor",
            namespaces=['/facts/{actorId}']
        ),
        UserPreferenceStrategy(
            name="prefCapture",
            namespaces=['/preferences/{actorId}']
        ),
        SummaryStrategy(
            name="sessionDigest",
            description="End of session summary",
            namespaces=['/summaries/{actorId}/{sessionId}']
        )
    ]
)
```

### Adding Strategy to Existing Memory

```python
# Add a new strategy to existing memory
memory_manager.update_memory_strategies(
    memory_id="existing-memory-id",
    add_strategies=[
        SummaryStrategy(
            name="newSummarizer",
            namespaces=['/new-summaries/{actorId}']
        )
    ]
)
```

### Modifying Strategy

```python
memory_manager.modify_strategy(
    memory_id=memory_id,
    strategy_name="factExtractor",
    new_description="Updated fact extraction strategy"
)
```

### Deleting Strategy

```python
memory_manager.delete_strategy(
    memory_id=memory_id,
    strategy_name="oldStrategy"
)
```

## Namespace Patterns

Namespaces organize extracted records hierarchically.

### Available Variables
| Variable | Description |
|----------|-------------|
| `{memoryStrategyId}` | ID of the strategy |
| `{actorId}` | User/actor identifier |
| `{sessionId}` | Session identifier |

### Pattern Examples

```python
# Per-user facts
"/users/{actorId}/facts"

# Per-session summaries
"/sessions/{actorId}/{sessionId}/summary"

# Strategy-organized records
"/strategies/{memoryStrategyId}/actors/{actorId}"

# Hierarchical preferences
"/preferences/{actorId}/communication"
"/preferences/{actorId}/shopping"
```

## Querying by Strategy

### List Records by Namespace

```python
# Get all facts for a user
facts = session.list_long_term_memory_records(
    namespace_prefix="/facts/"
)

# Get preferences
prefs = session.list_long_term_memory_records(
    namespace_prefix="/preferences/"
)
```

### Semantic Search by Namespace

```python
# Search within specific namespace
results = session.search_long_term_memories(
    query="programming language preferences",
    namespace_prefix="/preferences/",
    top_k=5
)
```

## Strategy Status Lifecycle

1. **CREATING**: Strategy is being initialized
2. **ACTIVE**: Strategy is ready to process conversations
3. **UPDATING**: Strategy is being modified
4. **DELETING**: Strategy is being removed
5. **FAILED**: Strategy creation/update failed

### Checking Status

```python
strategies = memory_manager.get_memory_strategies(memoryId=memory_id)
for strategy in strategies:
    print(f"{strategy['name']}: {strategy['status']}")
```

## Processing Characteristics

| Strategy | Processing Time | Extraction Trigger |
|----------|-----------------|-------------------|
| SemanticStrategy | 5-10 seconds | After each conversation turn |
| UserPreferenceStrategy | 5-10 seconds | When preferences detected |
| SummaryStrategy | 5-10 seconds | End of session or periodic |

## Best Practices

1. **Wait for ACTIVE Status**: Always verify strategies are active before writing events
2. **Use Specific Namespaces**: Design namespaces that match your query patterns
3. **Combine Strategies**: Use multiple strategies for comprehensive memory
4. **Test Extraction**: Verify extraction works as expected with sample conversations
5. **Monitor Extraction Jobs**: Use `list_memory_extraction_jobs()` to track processing
6. **Plan Retention**: Consider data lifecycle and privacy requirements

## Common Patterns

### Customer Support Agent
```python
strategies=[
    SemanticStrategy(name="issueTracker", namespaces=['/issues/{actorId}']),
    UserPreferenceStrategy(name="contactPrefs", namespaces=['/contact/{actorId}']),
    SummaryStrategy(name="ticketSummary", namespaces=['/tickets/{actorId}/{sessionId}'])
]
```

### Personal Assistant
```python
strategies=[
    UserPreferenceStrategy(name="userPrefs", namespaces=['/prefs/{actorId}']),
    SemanticStrategy(name="taskMemory", namespaces=['/tasks/{actorId}']),
    SummaryStrategy(name="dailyDigest", namespaces=['/digest/{actorId}/{sessionId}'])
]
```

### E-Commerce Agent
```python
strategies=[
    SemanticStrategy(name="orderHistory", namespaces=['/orders/{actorId}']),
    UserPreferenceStrategy(name="shoppingPrefs", namespaces=['/shopping/{actorId}']),
    SummaryStrategy(name="interactionLog", namespaces=['/logs/{actorId}/{sessionId}'])
]
```
