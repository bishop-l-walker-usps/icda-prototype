# Getting Started with Amazon Bedrock AgentCore Memory

## Overview

Amazon Bedrock AgentCore Memory enables creation and management of memory resources that store conversation context for AI agents, supporting both short-term and long-term memory features.

## Prerequisites

- AWS Account with configured credentials (`aws configure`)
- Python 3.10+
- AWS CLI configured with appropriate permissions

## Installation

```bash
mkdir agentcore-memory-quickstart
cd agentcore-memory-quickstart
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

pip install bedrock-agentcore
pip install bedrock-agentcore-starter-toolkit
```

## Step 1: Create an AgentCore Memory

### Using Starter Toolkit CLI

```bash
agentcore memory create CustomerSupportSemantic \
  --region us-west-2 \
  --description "Customer support memory store" \
  --strategies '[{"semanticMemoryStrategy": {"name": "semanticLongTermMemory", "namespaces": ["/strategies/{memoryStrategyId}/actors/{actorId}"]}}]' \
  --wait
```

List available memories:
```bash
agentcore memory list --region us-west-2
```

### Using Python Starter Toolkit

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SemanticStrategy

# Initialize memory manager
memory_manager = MemoryManager(region_name="us-west-2")

# Create memory with semantic strategy
memory = memory_manager.get_or_create_memory(
    name="CustomerSupportSemantic",
    description="Customer support memory store",
    strategies=[
        SemanticStrategy(
            name="semanticLongTermMemory",
            namespaces=['/strategies/{memoryStrategyId}/actors/{actorId}'],
        )
    ]
)

print(f"Memory ID: {memory.get('id')}")

# List all memories
memories = memory_manager.list_memories()
for mem in memories:
    print(f"  - {mem.get('name')}: {mem.get('id')}")
```

**Important**: Memory creation takes 2-3 minutes with semantic strategy enabled. Wait for status to become `ACTIVE`.

## Step 2: Write Events to Memory

```python
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

# Create session manager
session_manager = MemorySessionManager(
    memory_id=memory.get("id"),
    region_name="us-west-2"
)

# Create a session for a specific user
session = session_manager.create_memory_session(
    actor_id="User1",
    session_id="OrderSupportSession1"
)

# Add conversation turns
session.add_turns(
    messages=[
        ConversationalMessage(
            "Hi, how can I help you today?",
            MessageRole.ASSISTANT
        )
    ]
)

session.add_turns(
    messages=[
        ConversationalMessage(
            "Hi, I am a new customer. I just made an order, but it hasn't arrived. The Order number is #35476",
            MessageRole.USER
        )
    ]
)

session.add_turns(
    messages=[
        ConversationalMessage(
            "I'm sorry to hear that. Let me look up your order.",
            MessageRole.ASSISTANT
        )
    ]
)

# Retrieve last k turns
turns = session.get_last_k_turns(k=5)
for turn in turns:
    print(f"Turn: {turn}")
```

## Step 3: Retrieve Records from Long-Term Memory

### List All Memory Records

```python
memory_records = session.list_long_term_memory_records(
    namespace_prefix="/"
)

for record in memory_records:
    print(f"Memory record: {record}")
```

### Semantic Search

```python
# Search for relevant memories using natural language
memory_records = session.search_long_term_memories(
    query="can you summarize the support issue",
    namespace_prefix="/",
    top_k=3
)

for record in memory_records:
    print(f"Relevant memory: {record}")
```

## Step 4: Cleanup

### Using CLI
```bash
agentcore memory delete <memory-id> --region us-west-2 --wait
```

### Using Python
```python
memory_manager.delete_memory(memory_id=memory.get("id"))
print("Memory deleted successfully")
```

## Key Concepts

| Component | Purpose |
|-----------|---------|
| `MemoryManager` | Control plane operations (create, list, get, delete memories) |
| `MemorySessionManager` | Data plane operations (manage sessions within memories) |
| `ConversationalMessage` | Message wrapper with role (ASSISTANT/USER) |
| `MessageRole` | Enum for message roles |
| `SemanticStrategy` | Long-term memory extraction configuration |

## Namespace Patterns

Namespaces organize memory records hierarchically:

```
/strategies/{memoryStrategyId}/actors/{actorId}
/users/{actorId}/preferences
/summaries/{actorId}/{sessionId}
```

Variables are substituted at runtime:
- `{memoryStrategyId}` - ID of the strategy
- `{actorId}` - User/actor identifier
- `{sessionId}` - Session identifier

## Next Steps

- See `long-term-memory.md` for enabling long-term memory strategies
- See `semantic-search.md` for detailed search examples
- See `memory-gateway-agent.md` for integrating memory with agent gateways
