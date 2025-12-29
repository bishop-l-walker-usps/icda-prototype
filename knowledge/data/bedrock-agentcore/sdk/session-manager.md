# MemorySessionManager Class Reference

## Overview

The `MemorySessionManager` manages sessions within AgentCore Memory resources, handling data plane operations like adding conversation turns, retrieving history, and querying long-term memory.

## Installation

```bash
pip install bedrock-agentcore
```

## Import

```python
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
```

## Class Definition

```python
class MemorySessionManager:
    """Manager for memory sessions and events."""

    def __init__(
        self,
        memory_id: str,
        region_name: str = "us-west-2"
    ):
        """
        Initialize session manager.

        Args:
            memory_id: AgentCore Memory resource ID
            region_name: AWS region
        """
```

## Core Methods

### Session Creation

#### create_memory_session

```python
def create_memory_session(
    self,
    actor_id: str,
    session_id: str
) -> MemorySession:
    """
    Create or get a memory session.

    Args:
        actor_id: User/actor identifier
        session_id: Session identifier

    Returns:
        MemorySession object for operations
    """
```

**Example:**
```python
session_manager = MemorySessionManager(
    memory_id="mem-abc123",
    region_name="us-west-2"
)

session = session_manager.create_memory_session(
    actor_id="user123",
    session_id="chat001"
)
```

## MemorySession Methods

### Short-Term Memory Operations

#### add_turns

```python
def add_turns(
    self,
    messages: List[ConversationalMessage]
) -> None:
    """
    Add conversation turns to session.

    Args:
        messages: List of ConversationalMessage objects
    """
```

**Example:**
```python
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

# Add single message
session.add_turns(messages=[
    ConversationalMessage("Hello, how can I help?", MessageRole.ASSISTANT)
])

# Add user message
session.add_turns(messages=[
    ConversationalMessage("I need help with my order", MessageRole.USER)
])

# Add multiple messages at once
session.add_turns(messages=[
    ConversationalMessage("What's your order number?", MessageRole.ASSISTANT),
    ConversationalMessage("Order #12345", MessageRole.USER)
])
```

#### get_last_k_turns

```python
def get_last_k_turns(
    self,
    k: int = 10
) -> List[Turn]:
    """
    Retrieve last k conversation turns.

    Args:
        k: Number of turns to retrieve

    Returns:
        List of Turn objects
    """
```

**Example:**
```python
# Get last 5 turns
turns = session.get_last_k_turns(k=5)

for turn in turns:
    print(f"{turn.role}: {turn.text}")
```

### Long-Term Memory Operations

#### list_long_term_memory_records

```python
def list_long_term_memory_records(
    self,
    namespace_prefix: str = "/"
) -> List[MemoryRecord]:
    """
    List all long-term memory records.

    Args:
        namespace_prefix: Filter by namespace prefix

    Returns:
        List of MemoryRecord objects
    """
```

**Example:**
```python
# List all records
all_records = session.list_long_term_memory_records(
    namespace_prefix="/"
)

# List only preferences
preferences = session.list_long_term_memory_records(
    namespace_prefix="/preferences/"
)

# List facts for specific user
facts = session.list_long_term_memory_records(
    namespace_prefix="/facts/user123/"
)
```

#### search_long_term_memories

```python
def search_long_term_memories(
    self,
    query: str,
    namespace_prefix: str = "/",
    top_k: int = 5
) -> List[MemoryRecord]:
    """
    Semantic search across long-term memories.

    Args:
        query: Natural language search query
        namespace_prefix: Filter by namespace prefix
        top_k: Maximum results to return

    Returns:
        List of MemoryRecord objects ranked by relevance
    """
```

**Example:**
```python
# Search for order-related memories
results = session.search_long_term_memories(
    query="order delivery issue",
    namespace_prefix="/",
    top_k=3
)

for record in results:
    print(f"Score: {record.score}, Content: {record.content}")

# Search within preferences namespace
prefs = session.search_long_term_memories(
    query="programming language",
    namespace_prefix="/preferences/",
    top_k=5
)
```

### Event Operations

#### create_event

```python
def create_event(
    self,
    event_type: str,
    payload: Dict[str, Any]
) -> str:
    """
    Create a custom event.

    Args:
        event_type: Type of event
        payload: Event data

    Returns:
        Event ID
    """
```

#### list_events

```python
def list_events(
    self,
    event_type: Optional[str] = None
) -> List[Event]:
    """
    List events in session.

    Args:
        event_type: Filter by event type

    Returns:
        List of Event objects
    """
```

## Data Classes

### ConversationalMessage

```python
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

# Create user message
user_msg = ConversationalMessage(
    text="Hello",
    role=MessageRole.USER
)

# Create assistant message
assistant_msg = ConversationalMessage(
    text="Hi there!",
    role=MessageRole.ASSISTANT
)
```

### MessageRole

```python
from bedrock_agentcore.memory.constants import MessageRole

MessageRole.USER       # User messages
MessageRole.ASSISTANT  # Assistant/agent messages
```

### Turn

Returned by `get_last_k_turns()`:

```python
class Turn:
    role: MessageRole
    text: str
    timestamp: datetime
```

### MemoryRecord

Returned by long-term memory operations:

```python
class MemoryRecord:
    namespace: str      # Record namespace
    content: str        # Extracted content
    score: float        # Relevance score (for search)
    created_at: datetime
```

## Complete Example

```python
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
import time

# Initialize
session_manager = MemorySessionManager(
    memory_id="mem-abc123def456",
    region_name="us-west-2"
)

# Create session
session = session_manager.create_memory_session(
    actor_id="user123",
    session_id="support-session-001"
)

# Add conversation
session.add_turns(messages=[
    ConversationalMessage(
        "Hi, how can I help you today?",
        MessageRole.ASSISTANT
    )
])

session.add_turns(messages=[
    ConversationalMessage(
        "I'm having trouble with order #12345. It hasn't arrived yet.",
        MessageRole.USER
    )
])

session.add_turns(messages=[
    ConversationalMessage(
        "I understand. Let me look up order #12345 for you.",
        MessageRole.ASSISTANT
    )
])

# Retrieve short-term memory
print("=== Recent Conversation ===")
turns = session.get_last_k_turns(k=5)
for turn in turns:
    print(f"  {turn.role}: {turn.text[:50]}...")

# Wait for long-term extraction
print("\nWaiting for extraction...")
time.sleep(10)

# List long-term records
print("\n=== Long-Term Memory ===")
records = session.list_long_term_memory_records("/")
for record in records:
    print(f"  [{record.namespace}] {record.content}")

# Semantic search
print("\n=== Semantic Search: 'order problem' ===")
results = session.search_long_term_memories(
    query="order problem",
    top_k=3
)
for i, result in enumerate(results, 1):
    print(f"  {i}. (score: {result.score:.2f}) {result.content}")
```

## Integration Patterns

### With Strands Agent

```python
from strands import Agent

class MemoryAgent(Agent):
    def __init__(self, session):
        self.session = session
        self._load_context()
        super().__init__()

    def _load_context(self):
        """Load conversation history on init."""
        self.history = self.session.get_last_k_turns(k=10)

    def get_relevant_context(self, query: str) -> str:
        """Get relevant memories for query."""
        results = self.session.search_long_term_memories(
            query=query,
            top_k=5
        )
        return "\n".join([r.content for r in results])

    def process_message(self, user_input: str) -> str:
        # Save user message
        self.session.add_turns(messages=[
            ConversationalMessage(user_input, MessageRole.USER)
        ])

        # Get context
        context = self.get_relevant_context(user_input)

        # Generate response (simplified)
        response = self.invoke(user_input, context=context)

        # Save response
        self.session.add_turns(messages=[
            ConversationalMessage(response, MessageRole.ASSISTANT)
        ])

        return response
```

### With FastAPI

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    actor_id: str
    session_id: str

def get_session(request: ChatRequest):
    manager = MemorySessionManager(
        memory_id=os.environ["MEMORY_ID"],
        region_name="us-west-2"
    )
    return manager.create_memory_session(
        actor_id=request.actor_id,
        session_id=request.session_id
    )

@app.post("/chat")
async def chat(request: ChatRequest, session = Depends(get_session)):
    # Save user message
    session.add_turns(messages=[
        ConversationalMessage(request.message, MessageRole.USER)
    ])

    # Get relevant context
    context = session.search_long_term_memories(
        query=request.message,
        top_k=3
    )

    # Generate response
    response = generate_response(request.message, context)

    # Save response
    session.add_turns(messages=[
        ConversationalMessage(response, MessageRole.ASSISTANT)
    ])

    return {"response": response}
```

## Error Handling

```python
from botocore.exceptions import ClientError

try:
    session = session_manager.create_memory_session(
        actor_id="user123",
        session_id="session001"
    )
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'ResourceNotFoundException':
        print("Memory resource not found")
    elif error_code == 'AccessDeniedException':
        print("Insufficient permissions")
    else:
        raise
```

## Best Practices

1. **Reuse Sessions**: Create session once and reuse for conversation
2. **Batch Messages**: Group related messages in single `add_turns` call
3. **Wait for Extraction**: Long-term memory needs 5-10 seconds for async processing
4. **Use Namespace Prefixes**: Filter queries with specific prefixes for efficiency
5. **Handle Errors**: Wrap operations in try/except for graceful degradation
