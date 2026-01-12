# Session Management in Bedrock AgentCore

## Overview

This documentation covers implementing conversation state persistence using session identifiers within the AWS Bedrock AgentCore framework.

## Basic Session Handler

```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.context import RequestContext

app = BedrockAgentCoreApp()
sessions = {}  # In-memory storage (use database in production)

@app.entrypoint
def chat_handler(payload, context: RequestContext):
    """Handle chat with session management."""
    session_id = context.session_id or "default"
    message = payload.get("message", "")

    # Initialize session if new
    if session_id not in sessions:
        sessions[session_id] = {"messages": [], "count": 0}

    # Store message
    sessions[session_id]["messages"].append(message)
    sessions[session_id]["count"] += 1

    count = sessions[session_id]["count"]
    return {
        "response": f"Message {count}: You said '{message}'",
        "session_id": session_id,
        "message_count": count
    }

if __name__ == "__main__":
    app.run()
```

## Key Components

### RequestContext
Provides session information:
- `context.session_id` - Unique session identifier
- `context.actor_id` - User/actor identifier
- `context.memory_id` - Associated memory resource

### Session Storage Patterns

#### In-Memory (Development Only)
```python
sessions = {}

def get_session(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = {"messages": [], "metadata": {}}
    return sessions[session_id]
```

#### Redis (Production)
```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_session(session_id: str):
    data = redis_client.get(f"session:{session_id}")
    if data:
        return json.loads(data)
    return {"messages": [], "metadata": {}}

def save_session(session_id: str, session_data: dict):
    redis_client.setex(
        f"session:{session_id}",
        3600,  # 1 hour TTL
        json.dumps(session_data)
    )
```

#### AgentCore Memory (Recommended)
```python
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

def get_memory_session(memory_id: str, actor_id: str, session_id: str):
    session_manager = MemorySessionManager(
        memory_id=memory_id,
        region_name="us-west-2"
    )
    return session_manager.create_memory_session(
        actor_id=actor_id,
        session_id=session_id
    )
```

## CLI Usage

### Configure and Deploy
```bash
agentcore configure --entrypoint handler.py
agentcore deploy
```

### Invoke with Session
```bash
# Start conversation with explicit session
agentcore invoke '{"message": "Hello"}' --session-id conv1

# Continue same session
agentcore invoke '{"message": "How are you?"}' --session-id conv1

# Start new session
agentcore invoke '{"message": "Hello"}' --session-id conv2
```

## Advanced Session Management

### Session with Memory Integration

```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.context import RequestContext
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
import os

app = BedrockAgentCoreApp()

MEMORY_ID = os.environ.get("MEMORY_ID")
REGION = "us-west-2"

@app.entrypoint
def chat_handler(payload, context: RequestContext):
    """Chat handler with persistent memory."""

    # Get session from AgentCore Memory
    session_manager = MemorySessionManager(
        memory_id=MEMORY_ID,
        region_name=REGION
    )

    session = session_manager.create_memory_session(
        actor_id=context.actor_id or "default_user",
        session_id=context.session_id or "default_session"
    )

    # Load conversation history
    history = session.get_last_k_turns(k=10)

    # Process user message
    user_message = payload.get("message", "")

    # Save user message
    session.add_turns(messages=[
        ConversationalMessage(user_message, MessageRole.USER)
    ])

    # Generate response (simplified)
    response = f"I received: {user_message}"

    # Save assistant response
    session.add_turns(messages=[
        ConversationalMessage(response, MessageRole.ASSISTANT)
    ])

    return {
        "response": response,
        "session_id": context.session_id,
        "history_length": len(history)
    }

if __name__ == "__main__":
    app.run()
```

### Session Isolation

```python
class SessionManager:
    """Manages isolated sessions per user."""

    def __init__(self, memory_id: str, region: str = "us-west-2"):
        self.memory_id = memory_id
        self.region = region
        self._sessions = {}

    def get_session(self, actor_id: str, session_id: str):
        """Get or create an isolated session."""
        key = f"{actor_id}:{session_id}"

        if key not in self._sessions:
            manager = MemorySessionManager(
                memory_id=self.memory_id,
                region_name=self.region
            )
            self._sessions[key] = manager.create_memory_session(
                actor_id=actor_id,
                session_id=session_id
            )

        return self._sessions[key]

    def clear_session(self, actor_id: str, session_id: str):
        """Clear a session from cache."""
        key = f"{actor_id}:{session_id}"
        if key in self._sessions:
            del self._sessions[key]
```

## Session Lifecycle

### 1. Session Creation
```python
session = session_manager.create_memory_session(
    actor_id="user123",
    session_id="chat001"
)
```

### 2. Adding Events
```python
session.add_turns(messages=[
    ConversationalMessage("Hello", MessageRole.USER)
])
```

### 3. Retrieving History
```python
# Last k turns
turns = session.get_last_k_turns(k=10)

# All events
events = session.list_events()
```

### 4. Session Cleanup
Sessions are automatically managed by AgentCore with configurable retention.

## Production Considerations

1. **Use Persistent Storage**: Replace in-memory dicts with Redis, DynamoDB, or AgentCore Memory
2. **Session Expiry**: Implement TTL for session cleanup
3. **Error Handling**: Gracefully handle session creation failures
4. **Concurrency**: Use thread-safe session access patterns
5. **Monitoring**: Log session creation, access, and errors

## Environment Variables

```bash
export MEMORY_ID="your-memory-id"
export AWS_REGION="us-west-2"
export SESSION_TTL="3600"
```

## Testing Sessions

```python
import pytest

def test_session_persistence():
    """Test that session maintains state."""
    session_id = "test-session-001"

    # First message
    response1 = chat_handler(
        {"message": "Hello"},
        MockContext(session_id=session_id)
    )
    assert response1["message_count"] == 1

    # Second message (same session)
    response2 = chat_handler(
        {"message": "World"},
        MockContext(session_id=session_id)
    )
    assert response2["message_count"] == 2

def test_session_isolation():
    """Test that sessions are isolated."""
    response1 = chat_handler(
        {"message": "Hello"},
        MockContext(session_id="session-a")
    )

    response2 = chat_handler(
        {"message": "Hello"},
        MockContext(session_id="session-b")
    )

    # Each session has count of 1
    assert response1["message_count"] == 1
    assert response2["message_count"] == 1
```
