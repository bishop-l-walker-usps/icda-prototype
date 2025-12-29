# Memory Types in Amazon Bedrock AgentCore

## Overview

Amazon Bedrock AgentCore supports two primary memory types, each serving different purposes in agent conversation management.

## Short-Term Memory (STM)

### Characteristics
- **Scope**: Session-scoped only
- **Storage**: Raw conversation turns
- **Processing**: Instant retrieval (no processing delay)
- **Retention**: 7 days by default
- **Strategies**: Empty strategies array `[]`

### Creation

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager

memory_manager = MemoryManager(region_name="us-west-2")

# Create short-term memory (no strategies)
stm = memory_manager.get_or_create_memory(
    name="ConversationBuffer",
    description="Short-term conversation memory",
    strategies=[]  # Empty = short-term only
)
```

### Usage

```python
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

session_manager = MemorySessionManager(
    memory_id=stm.get("id"),
    region_name="us-west-2"
)

session = session_manager.create_memory_session(
    actor_id="user123",
    session_id="chat001"
)

# Add turns
session.add_turns(messages=[
    ConversationalMessage("Hello!", MessageRole.USER)
])

# Retrieve immediately (no delay)
turns = session.get_last_k_turns(k=10)
```

### Use Cases
- Conversation context within a single chat session
- Immediate message history retrieval
- Stateless agent deployments needing session continuity

## Long-Term Memory (LTM)

### Characteristics
- **Scope**: Cross-session persistence
- **Storage**: Extracted facts, preferences, summaries
- **Processing**: Async extraction (5-10 seconds delay)
- **Retention**: 30 days by default
- **Strategies**: One or more extraction strategies

### Creation

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import (
    SemanticStrategy,
    UserPreferenceStrategy
)

memory_manager = MemoryManager(region_name="us-west-2")

# Create long-term memory with strategies
ltm = memory_manager.get_or_create_memory(
    name="PersonalizedAgent",
    description="Long-term personalized memory",
    strategies=[
        SemanticStrategy(
            name="factExtraction",
            namespaces=['/facts/{actorId}']
        ),
        UserPreferenceStrategy(
            name="preferences",
            namespaces=['/prefs/{actorId}']
        )
    ]
)
```

### Usage

```python
# After adding conversation turns and waiting for extraction...
import time

session.add_turns(messages=[
    ConversationalMessage("I prefer Python for data science", MessageRole.USER)
])

# Wait for async extraction
time.sleep(10)

# Query long-term memory
preferences = session.search_long_term_memories(
    query="programming language preferences",
    namespace_prefix="/prefs/",
    top_k=5
)

facts = session.list_long_term_memory_records(
    namespace_prefix="/facts/"
)
```

### Use Cases
- Personalized agent experiences
- Cross-session user preference retention
- Knowledge accumulation over time
- User profile building

## Comparison Table

| Feature | Short-Term Memory | Long-Term Memory |
|---------|-------------------|------------------|
| **Scope** | Single session | Cross-session |
| **Data Type** | Raw conversation turns | Extracted insights |
| **Retrieval Speed** | Instant | Requires extraction delay |
| **Strategies Required** | No (empty array) | Yes (1+ strategies) |
| **Default Retention** | 7 days | 30 days |
| **Semantic Search** | No | Yes |
| **Creation Time** | Fast | 2-3 minutes (strategy activation) |
| **Processing** | Synchronous | Asynchronous (5-10 sec) |

## Hybrid Approach

Most production agents use both memory types:

```python
# Create memory with both capabilities
hybrid_memory = memory_manager.get_or_create_memory(
    name="HybridAgentMemory",
    strategies=[
        SemanticStrategy(
            name="longTermFacts",
            namespaces=['/ltm/facts/{actorId}']
        )
    ]
)

session = session_manager.create_memory_session(
    actor_id="user123",
    session_id="session001"
)

# Short-term: immediate retrieval
recent_turns = session.get_last_k_turns(k=5)

# Long-term: semantic search
relevant_facts = session.search_long_term_memories(
    query="user preferences",
    top_k=3
)
```

## Memory Hooks for Agents

Implement automatic memory operations using hooks:

```python
class MemoryHook:
    def __init__(self, session):
        self.session = session

    def on_init(self):
        """Load conversation history on agent start"""
        return self.session.get_last_k_turns(k=10)

    def on_message(self, message, role):
        """Persist each message"""
        self.session.add_turns(messages=[
            ConversationalMessage(message, role)
        ])

    def get_context(self, query):
        """Retrieve relevant long-term memories"""
        return self.session.search_long_term_memories(
            query=query,
            top_k=5
        )
```

## Best Practices

1. **Start with STM**: Begin with short-term memory, add long-term strategies as needed
2. **Test Extraction Delays**: Account for async processing in integration tests
3. **Design Namespace Hierarchy**: Plan namespace patterns before implementation
4. **Monitor Retention**: Set appropriate retention periods for your use case
5. **Clean Up Test Resources**: Delete unused memory resources to avoid clutter
