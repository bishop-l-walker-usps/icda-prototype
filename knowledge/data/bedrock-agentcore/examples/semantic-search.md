# Semantic Search Memory Example

## Overview

This example demonstrates implementing semantic search functionality within the Bedrock AgentCore Memory system for managing conversational memory with long-term storage capabilities.

## Complete Implementation

### Prerequisites

```bash
pip install bedrock-agentcore
pip install bedrock-agentcore-starter-toolkit
```

### Full Code Example

```python
"""
Semantic Search Memory Example
Demonstrates creating memory with semantic search and querying it.
"""

from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SemanticStrategy
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
import time

# Configuration
REGION = "us-west-2"
MEMORY_NAME = "CustomerSupportSemantic"

def create_memory_with_semantic_strategy():
    """Create a memory resource with semantic search enabled."""
    memory_manager = MemoryManager(region_name=REGION)

    memory = memory_manager.get_or_create_memory(
        name=MEMORY_NAME,
        description="Customer support memory with semantic search",
        strategies=[
            SemanticStrategy(
                name="semanticLongTermMemory",
                namespaces=['/strategies/{memoryStrategyId}/actors/{actorId}']
            )
        ]
    )

    print(f"Memory created with ID: {memory.get('id')}")
    return memory

def create_session(memory_id: str, actor_id: str, session_id: str):
    """Initialize a memory session for a specific user."""
    session_manager = MemorySessionManager(
        memory_id=memory_id,
        region_name=REGION
    )

    session = session_manager.create_memory_session(
        actor_id=actor_id,
        session_id=session_id
    )

    return session

def add_conversation(session):
    """Add sample conversation turns to the session."""

    # Assistant greeting
    session.add_turns(
        messages=[
            ConversationalMessage(
                "Hi, how can I help you today?",
                MessageRole.ASSISTANT
            )
        ]
    )

    # User describes issue
    session.add_turns(
        messages=[
            ConversationalMessage(
                "Hi, I am a new customer. I just made an order, but it hasn't arrived. The Order number is #35476",
                MessageRole.USER
            )
        ]
    )

    # Assistant response
    session.add_turns(
        messages=[
            ConversationalMessage(
                "I'm sorry to hear that. Let me look up your order #35476. I can see it was shipped on December 20th.",
                MessageRole.ASSISTANT
            )
        ]
    )

    # More context from user
    session.add_turns(
        messages=[
            ConversationalMessage(
                "I live at 123 Main Street, Seattle. The delivery was supposed to arrive by December 22nd.",
                MessageRole.USER
            )
        ]
    )

    print("Conversation added to session")

def retrieve_short_term_memory(session, k: int = 5):
    """Retrieve recent conversation turns."""
    turns = session.get_last_k_turns(k=k)

    print(f"\n=== Last {k} Turns ===")
    for turn in turns:
        print(f"  {turn}")

    return turns

def list_long_term_records(session, namespace_prefix: str = "/"):
    """List all long-term memory records."""
    records = session.list_long_term_memory_records(
        namespace_prefix=namespace_prefix
    )

    print(f"\n=== Long-Term Memory Records ===")
    for record in records:
        print(f"  {record}")

    return records

def semantic_search(session, query: str, top_k: int = 3):
    """Perform semantic search on long-term memory."""
    results = session.search_long_term_memories(
        query=query,
        namespace_prefix="/",
        top_k=top_k
    )

    print(f"\n=== Semantic Search: '{query}' ===")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result}")

    return results

def cleanup(memory_manager, memory_id: str):
    """Delete memory resource."""
    memory_manager.delete_memory(memory_id=memory_id)
    print(f"Memory {memory_id} deleted")

def main():
    """Run the complete semantic search example."""

    # Step 1: Create memory with semantic strategy
    print("Creating memory resource...")
    memory = create_memory_with_semantic_strategy()
    memory_id = memory.get('id')

    try:
        # Step 2: Create session
        print("\nCreating session...")
        session = create_session(
            memory_id=memory_id,
            actor_id="User1",
            session_id="OrderSupportSession1"
        )

        # Step 3: Add conversation
        print("\nAdding conversation...")
        add_conversation(session)

        # Step 4: Retrieve short-term memory (immediate)
        retrieve_short_term_memory(session, k=5)

        # Step 5: Wait for semantic extraction (async process)
        print("\nWaiting for semantic extraction (10 seconds)...")
        time.sleep(10)

        # Step 6: List long-term records
        list_long_term_records(session)

        # Step 7: Semantic search
        semantic_search(session, "order delivery issue", top_k=3)
        semantic_search(session, "customer address", top_k=3)
        semantic_search(session, "order number", top_k=3)

    finally:
        # Cleanup (optional - comment out to keep memory for inspection)
        # memory_manager = MemoryManager(region_name=REGION)
        # cleanup(memory_manager, memory_id)
        pass

if __name__ == "__main__":
    main()
```

## Key Functions

### Memory Creation
```python
memory_manager.get_or_create_memory(
    name="MemoryName",
    strategies=[SemanticStrategy(...)]
)
```

### Session Management
```python
session_manager.create_memory_session(
    actor_id="user_identifier",
    session_id="session_identifier"
)
```

### Adding Turns
```python
session.add_turns(messages=[
    ConversationalMessage(text, MessageRole.USER)
])
```

### Short-Term Retrieval
```python
turns = session.get_last_k_turns(k=5)
```

### Long-Term Listing
```python
records = session.list_long_term_memory_records(
    namespace_prefix="/"
)
```

### Semantic Search
```python
results = session.search_long_term_memories(
    query="natural language query",
    namespace_prefix="/",
    top_k=3
)
```

## Expected Output

```
Creating memory resource...
Memory created with ID: mem-abc123def456

Creating session...

Adding conversation...
Conversation added to session

=== Last 5 Turns ===
  Turn(role=ASSISTANT, text="Hi, how can I help you today?")
  Turn(role=USER, text="Hi, I am a new customer...")
  Turn(role=ASSISTANT, text="I'm sorry to hear that...")
  Turn(role=USER, text="I live at 123 Main Street...")

Waiting for semantic extraction (10 seconds)...

=== Long-Term Memory Records ===
  Record(namespace="/strategies/.../actors/User1", content="Order #35476")
  Record(namespace="/strategies/.../actors/User1", content="Address: 123 Main Street, Seattle")
  Record(namespace="/strategies/.../actors/User1", content="Expected delivery: December 22nd")

=== Semantic Search: 'order delivery issue' ===
  1. Record(content="Order #35476 - delivery delay", score=0.92)
  2. Record(content="Expected delivery December 22nd", score=0.87)
  3. Record(content="Customer is new", score=0.65)
```

## Integration Patterns

### With Strands Agents

```python
from strands import Agent

class MemoryEnabledAgent(Agent):
    def __init__(self, session):
        self.session = session
        super().__init__()

    def get_context(self, user_message: str) -> str:
        """Retrieve relevant context from memory."""
        results = self.session.search_long_term_memories(
            query=user_message,
            top_k=3
        )
        return "\n".join([r.content for r in results])

    def save_interaction(self, user_msg: str, assistant_msg: str):
        """Save interaction to memory."""
        self.session.add_turns(messages=[
            ConversationalMessage(user_msg, MessageRole.USER),
            ConversationalMessage(assistant_msg, MessageRole.ASSISTANT)
        ])
```

### With LangChain

```python
from langchain.memory import BaseMemory

class AgentCoreMemory(BaseMemory):
    def __init__(self, session):
        self.session = session

    def load_memory_variables(self, inputs):
        query = inputs.get("input", "")
        results = self.session.search_long_term_memories(
            query=query, top_k=5
        )
        return {"history": results}

    def save_context(self, inputs, outputs):
        self.session.add_turns(messages=[
            ConversationalMessage(inputs["input"], MessageRole.USER),
            ConversationalMessage(outputs["output"], MessageRole.ASSISTANT)
        ])
```

## Troubleshooting

### No Long-Term Records Found
- Ensure you waited for async extraction (10+ seconds)
- Verify strategy status is ACTIVE
- Check namespace prefix matches your query

### Memory Creation Timeout
- Memory with strategies takes 2-3 minutes to activate
- Use `--wait` flag or poll for ACTIVE status

### Search Returns No Results
- Verify records exist with `list_long_term_memory_records()`
- Try broader namespace prefix "/"
- Increase `top_k` value
