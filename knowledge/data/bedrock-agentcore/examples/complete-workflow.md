# Complete AgentCore Memory Workflow

## Overview

This document provides a complete end-to-end implementation workflow for AgentCore Memory, from setup to production deployment.

## Prerequisites

```bash
# Install dependencies
pip install bedrock-agentcore
pip install bedrock-agentcore-starter-toolkit

# Configure AWS
aws configure
# or set AWS_PROFILE
export AWS_PROFILE=your-profile
```

## Step 1: Create Memory Resource

```python
"""
Step 1: Create a memory resource with strategies.
"""

from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import (
    SemanticStrategy,
    UserPreferenceStrategy,
    SummaryStrategy
)

def create_memory():
    """Create a comprehensive memory resource."""
    memory_manager = MemoryManager(region_name="us-west-2")

    memory = memory_manager.get_or_create_memory(
        name="ProductionAgentMemory",
        description="Full-featured agent memory with all strategies",
        strategies=[
            # Extract facts from conversations
            SemanticStrategy(
                name="factExtraction",
                namespaces=['/facts/{actorId}']
            ),
            # Capture user preferences
            UserPreferenceStrategy(
                name="userPreferences",
                namespaces=['/preferences/{actorId}']
            ),
            # Summarize sessions
            SummaryStrategy(
                name="sessionSummaries",
                description="End of session summaries",
                namespaces=['/summaries/{actorId}/{sessionId}']
            )
        ]
    )

    print(f"Memory ID: {memory.get('id')}")
    print(f"Status: {memory.get('status')}")

    # Wait for strategies to be active
    import time
    while True:
        strategies = memory_manager.get_memory_strategies(memoryId=memory.get('id'))
        all_active = all(s.get('status') == 'ACTIVE' for s in strategies)
        if all_active:
            print("All strategies active!")
            break
        print("Waiting for strategies to activate...")
        time.sleep(10)

    return memory

if __name__ == "__main__":
    memory = create_memory()
```

## Step 2: Create Session Manager

```python
"""
Step 2: Create a session manager for memory operations.
"""

from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
import os

class AgentMemory:
    """Wrapper for AgentCore memory operations."""

    def __init__(self, memory_id: str, region: str = "us-west-2"):
        self.memory_id = memory_id
        self.region = region
        self._session_manager = None
        self._current_session = None

    @property
    def session_manager(self):
        if self._session_manager is None:
            self._session_manager = MemorySessionManager(
                memory_id=self.memory_id,
                region_name=self.region
            )
        return self._session_manager

    def start_session(self, actor_id: str, session_id: str):
        """Start or resume a session."""
        self._current_session = self.session_manager.create_memory_session(
            actor_id=actor_id,
            session_id=session_id
        )
        return self._current_session

    def add_user_message(self, message: str):
        """Add a user message to the session."""
        if self._current_session is None:
            raise ValueError("No active session. Call start_session first.")

        self._current_session.add_turns(messages=[
            ConversationalMessage(message, MessageRole.USER)
        ])

    def add_assistant_message(self, message: str):
        """Add an assistant message to the session."""
        if self._current_session is None:
            raise ValueError("No active session. Call start_session first.")

        self._current_session.add_turns(messages=[
            ConversationalMessage(message, MessageRole.ASSISTANT)
        ])

    def get_recent_history(self, k: int = 10):
        """Get recent conversation turns."""
        if self._current_session is None:
            return []
        return self._current_session.get_last_k_turns(k=k)

    def search_memories(self, query: str, namespace: str = "/", top_k: int = 5):
        """Search long-term memories."""
        if self._current_session is None:
            return []
        return self._current_session.search_long_term_memories(
            query=query,
            namespace_prefix=namespace,
            top_k=top_k
        )

    def get_user_preferences(self, actor_id: str = None):
        """Get user preferences from memory."""
        if self._current_session is None:
            return []
        prefix = f"/preferences/{actor_id}/" if actor_id else "/preferences/"
        return self._current_session.list_long_term_memory_records(
            namespace_prefix=prefix
        )

    def get_facts(self, actor_id: str = None):
        """Get extracted facts from memory."""
        if self._current_session is None:
            return []
        prefix = f"/facts/{actor_id}/" if actor_id else "/facts/"
        return self._current_session.list_long_term_memory_records(
            namespace_prefix=prefix
        )
```

## Step 3: Integrate with Agent

```python
"""
Step 3: Integrate memory with an AI agent.
"""

from strands import Agent
import os

class MemoryEnabledAgent:
    """Agent with persistent memory."""

    def __init__(self, memory_id: str):
        self.memory = AgentMemory(memory_id=memory_id)
        self.agent = Agent(
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        )

    def start_conversation(self, user_id: str, session_id: str):
        """Start a new conversation session."""
        self.user_id = user_id
        self.session_id = session_id
        self.memory.start_session(actor_id=user_id, session_id=session_id)

        # Load existing context
        self._load_context()

    def _load_context(self):
        """Load context from memory."""
        # Get recent conversation history
        history = self.memory.get_recent_history(k=10)

        # Get user preferences
        preferences = self.memory.get_user_preferences(self.user_id)

        # Get relevant facts
        facts = self.memory.get_facts(self.user_id)

        # Build context for agent
        context_parts = []

        if preferences:
            prefs_text = "\n".join([f"- {p.content}" for p in preferences])
            context_parts.append(f"User Preferences:\n{prefs_text}")

        if facts:
            facts_text = "\n".join([f"- {f.content}" for f in facts])
            context_parts.append(f"Known Facts:\n{facts_text}")

        self.context = "\n\n".join(context_parts)

    def chat(self, user_message: str) -> str:
        """Process a user message."""
        # Save user message
        self.memory.add_user_message(user_message)

        # Search for relevant memories
        relevant = self.memory.search_memories(user_message, top_k=3)
        relevant_context = "\n".join([r.content for r in relevant]) if relevant else ""

        # Build full context
        full_context = f"""
{self.context}

Relevant Memories:
{relevant_context}
""" if relevant_context else self.context

        # Generate response
        system_prompt = f"""You are a helpful assistant with access to user memory.

Context:
{full_context}

Use this context to provide personalized, contextual responses.
"""

        response = self.agent.invoke(
            user_message,
            system=system_prompt
        )

        # Save assistant response
        self.memory.add_assistant_message(response)

        return response


def main():
    """Run the memory-enabled agent."""
    memory_id = os.environ.get("MEMORY_ID")
    if not memory_id:
        print("Set MEMORY_ID environment variable")
        return

    agent = MemoryEnabledAgent(memory_id=memory_id)

    # Start conversation
    agent.start_conversation(
        user_id="user123",
        session_id="session001"
    )

    # Chat loop
    print("Agent ready. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response = agent.chat(user_input)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
```

## Step 4: Deploy to AgentCore Runtime

### agent.py

```python
"""
Step 4: Deploy-ready agent for AgentCore Runtime.
"""

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.context import RequestContext
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
from strands import Agent
import os

app = BedrockAgentCoreApp()

MEMORY_ID = os.environ.get("MEMORY_ID")
REGION = os.environ.get("AWS_REGION", "us-west-2")

@app.entrypoint
def handler(payload, context: RequestContext):
    """Main agent handler."""

    # Extract request info
    message = payload.get("message", "")
    actor_id = context.actor_id or payload.get("user_id", "default")
    session_id = context.session_id or payload.get("session_id", "default")

    # Initialize memory session
    session_manager = MemorySessionManager(
        memory_id=MEMORY_ID,
        region_name=REGION
    )
    session = session_manager.create_memory_session(
        actor_id=actor_id,
        session_id=session_id
    )

    # Load context
    history = session.get_last_k_turns(k=10)
    relevant = session.search_long_term_memories(
        query=message,
        namespace_prefix="/",
        top_k=5
    )

    # Build context
    context_text = ""
    if relevant:
        context_text = "Relevant memories:\n" + "\n".join([r.content for r in relevant])

    # Save user message
    session.add_turns(messages=[
        ConversationalMessage(message, MessageRole.USER)
    ])

    # Generate response
    agent = Agent(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    response = agent.invoke(
        message,
        system=f"You are a helpful assistant.\n\n{context_text}"
    )

    # Save response
    session.add_turns(messages=[
        ConversationalMessage(response, MessageRole.ASSISTANT)
    ])

    return {
        "response": response,
        "session_id": session_id,
        "actor_id": actor_id
    }

if __name__ == "__main__":
    app.run()
```

### requirements.txt

```
bedrock-agentcore>=0.1.0
bedrock-agentcore-starter-toolkit>=0.1.0
strands-agents>=0.1.0
```

### Deploy

```bash
# Configure
agentcore configure --entrypoint agent.py

# Set environment
export MEMORY_ID="mem-abc123"

# Deploy
agentcore deploy

# Test
agentcore invoke '{"message": "Hello!", "user_id": "user123"}' --session-id test1
```

## Step 5: Monitor and Maintain

```python
"""
Step 5: Monitoring and maintenance utilities.
"""

from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager

def check_memory_health(memory_id: str):
    """Check memory resource health."""
    manager = MemoryManager(region_name="us-west-2")

    # Get memory status
    memory = manager.get_memory(memory_id)
    print(f"Memory Status: {memory.get('status')}")

    # Check strategies
    strategies = manager.get_memory_strategies(memoryId=memory_id)
    print("\nStrategies:")
    for s in strategies:
        print(f"  {s['name']}: {s['status']}")

    return memory.get('status') == 'ACTIVE'


def list_all_memories():
    """List all memory resources."""
    manager = MemoryManager(region_name="us-west-2")

    memories = manager.list_memories()
    print("Available Memories:")
    for m in memories:
        print(f"  {m['name']}: {m['id']} ({m['status']})")

    return memories


def cleanup_memory(memory_id: str):
    """Delete a memory resource."""
    manager = MemoryManager(region_name="us-west-2")

    print(f"Deleting memory {memory_id}...")
    manager.delete_memory_and_wait(memory_id=memory_id)
    print("Deleted successfully")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python monitor.py <check|list|cleanup> [memory_id]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "check":
        memory_id = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MEMORY_ID")
        check_memory_health(memory_id)

    elif command == "list":
        list_all_memories()

    elif command == "cleanup":
        memory_id = sys.argv[2]
        cleanup_memory(memory_id)
```

## Common Issues and Solutions

### Memory Creation Timeout
- Memory with strategies takes 2-3 minutes to activate
- Use `get_or_create_memory` which handles waiting

### No Long-Term Records Found
- Wait 10+ seconds after adding turns for extraction
- Check strategy status is ACTIVE
- Verify namespace prefix in queries

### Session Not Found
- Sessions are created on first use
- Use consistent actor_id and session_id

### Rate Limiting
- Implement exponential backoff for retries
- Batch operations where possible

## Best Practices Summary

1. **Memory Design**: Plan namespace hierarchy before implementation
2. **Session Reuse**: Create session once per conversation
3. **Context Loading**: Load relevant context at conversation start
4. **Async Awareness**: Account for 5-10 second extraction delay
5. **Error Handling**: Gracefully handle memory unavailability
6. **Cleanup**: Delete test resources to avoid clutter
7. **Monitoring**: Check memory and strategy status regularly
