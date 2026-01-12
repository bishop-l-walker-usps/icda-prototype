# Memory & Gateway Agent Integration

## Overview

This guide demonstrates building production-ready AI agents with persistent memory and tool access through Amazon Bedrock AgentCore Gateway using the Model Context Protocol (MCP).

## Architecture

```
User Request
    │
    ▼
┌─────────────────────┐
│  AgentCore Runtime  │
│  ┌───────────────┐  │
│  │   MemoryHook  │◄─┼── Load history on init
│  │   (on_init)   │  │
│  └───────┬───────┘  │
│          ▼          │
│  ┌───────────────┐  │
│  │  Strands Agent│  │
│  │  (Claude 3.7) │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │ Gateway Tools │◄─┼── MCP Tool Discovery
│  │  (Calculator) │  │
│  └───────┬───────┘  │
│          │          │
│  ┌───────▼───────┐  │
│  │   MemoryHook  │──┼─► Persist after message
│  │ (on_message)  │  │
│  └───────────────┘  │
└─────────────────────┘
    │
    ▼
AgentCore Memory Service
```

## Memory Types

### Short-Term Memory (STM)

Stores raw conversation turns within sessions only.

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager

memory_manager = MemoryManager(region_name="us-west-2")

# Create STM (empty strategies)
stm = memory_manager.get_or_create_memory(
    name="AgentSTM",
    description="Short-term conversation buffer",
    strategies=[]  # Empty = short-term only
)
```

**Characteristics:**
- Instant retrieval (no processing delay)
- 7-day retention by default
- Session-scoped only

### Long-Term Memory (LTM)

Extracts and retains information across sessions.

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import (
    SemanticStrategy,
    UserPreferenceStrategy
)

# Create LTM with extraction strategies
ltm = memory_manager.get_or_create_memory(
    name="AgentLTM",
    description="Long-term personalized memory",
    strategies=[
        UserPreferenceStrategy(
            name="userPreferences",
            namespaces=['/preferences/{actorId}']
        ),
        SemanticStrategy(
            name="factExtraction",
            namespaces=['/facts/{actorId}']
        )
    ]
)
```

**Characteristics:**
- Async extraction (5-10 seconds)
- 30-day retention by default
- Cross-session persistence

## Memory Hook Implementation

```python
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole
import os

class MemoryHook:
    """Automatic memory operations for agents."""

    def __init__(self, memory_id: str, actor_id: str, session_id: str):
        self.session_manager = MemorySessionManager(
            memory_id=memory_id,
            region_name=os.environ.get("AWS_REGION", "us-west-2")
        )
        self.session = self.session_manager.create_memory_session(
            actor_id=actor_id,
            session_id=session_id
        )

    def on_init(self) -> list:
        """Load conversation history on agent initialization."""
        return self.session.get_last_k_turns(k=10)

    def on_message(self, message: str, role: MessageRole):
        """Persist each message to memory."""
        self.session.add_turns(messages=[
            ConversationalMessage(message, role)
        ])

    def get_relevant_context(self, query: str, top_k: int = 5) -> list:
        """Retrieve relevant long-term memories."""
        return self.session.search_long_term_memories(
            query=query,
            namespace_prefix="/",
            top_k=top_k
        )

    def get_preferences(self) -> list:
        """Get user preferences from memory."""
        return self.session.list_long_term_memory_records(
            namespace_prefix="/preferences/"
        )
```

## Gateway Setup

### Create OAuth Authorizer

```python
from bedrock_agentcore_starter_toolkit.operations.gateway.manager import GatewayManager

gateway_manager = GatewayManager(region_name="us-west-2")

# Create OAuth authorizer with Cognito
authorizer = gateway_manager.create_oauth_authorizer_with_cognito(
    name="AgentGatewayAuth",
    client_name="agent-client"
)
```

### Create Gateway with Tools

```python
# Define calculator tool schema
calculator_tool = {
    "name": "calculator",
    "description": "Perform mathematical operations",
    "inputSchema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["operation", "a", "b"]
    }
}

# Create gateway
gateway = gateway_manager.create_gateway(
    name="AgentToolGateway",
    tools=[calculator_tool],
    authorizer_id=authorizer['id']
)

# Get access token
access_token = gateway_manager.get_access_token(
    gateway_id=gateway['id'],
    client_id=authorizer['clientId'],
    client_secret=authorizer['clientSecret']
)

# Save config for agent
import json
with open("gateway_config.json", "w") as f:
    json.dump({
        "gateway_url": gateway['url'],
        "gateway_id": gateway['id'],
        "access_token": access_token
    }, f)
```

## Agent with Gateway Integration

```python
from strands import Agent
from mcp import ClientSession
import json
import os

class MemoryGatewayAgent:
    """Agent with memory and gateway tool access."""

    def __init__(self):
        # Load gateway config
        with open("gateway_config.json") as f:
            self.gateway_config = json.load(f)

        # Initialize memory
        self.memory_hook = MemoryHook(
            memory_id=os.environ["MEMORY_ID"],
            actor_id=os.environ.get("ACTOR_ID", "default"),
            session_id=os.environ.get("SESSION_ID", "default")
        )

        # Load history
        self.history = self.memory_hook.on_init()

        # Initialize agent
        self.agent = Agent(
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            system_prompt=self._build_system_prompt()
        )

        # Discover gateway tools
        self._load_gateway_tools()

    def _build_system_prompt(self) -> str:
        """Build system prompt with memory context."""
        preferences = self.memory_hook.get_preferences()
        pref_text = "\n".join([p.content for p in preferences]) if preferences else "None"

        return f"""You are a helpful assistant with memory and tool access.

User Preferences:
{pref_text}

You have access to a calculator tool for mathematical operations.
"""

    async def _load_gateway_tools(self):
        """Discover tools from gateway via MCP."""
        try:
            async with ClientSession(
                url=self.gateway_config["gateway_url"],
                headers={"Authorization": f"Bearer {self.gateway_config['access_token']}"}
            ) as session:
                tools = await session.list_tools()
                self.agent.tools = tools
                print(f"Loaded {len(tools)} tools from gateway")
        except Exception as e:
            print(f"Gateway connection failed: {e}")
            self.agent.tools = []

    def chat(self, user_message: str) -> str:
        """Process user message with memory."""
        # Get relevant context
        context = self.memory_hook.get_relevant_context(user_message)

        # Save user message
        self.memory_hook.on_message(user_message, MessageRole.USER)

        # Generate response
        response = self.agent.invoke(
            user_message,
            context=context
        )

        # Save assistant response
        self.memory_hook.on_message(response, MessageRole.ASSISTANT)

        return response
```

## Deployment

### Configure
```bash
agentcore configure -e agent_with_gateway.py
```

### Set Environment
```bash
export MEMORY_ID="your-memory-id"
export ACTOR_ID="user123"
export SESSION_ID="session001"
```

### Deploy
```bash
agentcore deploy
```

## Testing

### Short-Term Memory Test
```bash
# Session 1: Tell agent your name
agentcore invoke '{"message": "My name is Alice"}' --session-id test1

# Same session: Query name
agentcore invoke '{"message": "What is my name?"}' --session-id test1
# Expected: "Your name is Alice"
```

### Long-Term Memory Test
```bash
# Session A: Establish preference
agentcore invoke '{"message": "I prefer Python"}' --session-id sessionA

# Wait for extraction
sleep 10

# Session B: Query preference
agentcore invoke '{"message": "What language do I prefer?"}' --session-id sessionB
# Expected: "You prefer Python"
```

### Gateway Calculator Test
```bash
agentcore invoke '{"message": "Calculate 25 multiplied by 18"}' --session-id calc1
# Expected: "450"
```

## Requirements

```
bedrock-agentcore
bedrock-agentcore-starter-toolkit
strands-agents
mcp
```

## Configuration Files

### requirements.txt
```
bedrock-agentcore>=0.1.0
bedrock-agentcore-starter-toolkit>=0.1.0
strands-agents>=0.1.0
mcp>=0.1.0
```

### gateway_config.json (generated)
```json
{
    "gateway_url": "https://gateway.agentcore.aws/...",
    "gateway_id": "gw-abc123",
    "access_token": "eyJ..."
}
```

## Best Practices

1. **Separate STM and LTM**: Use STM for quick session context, LTM for persistent facts
2. **Wait for Extraction**: Account for 5-10 second async extraction in tests
3. **Handle Gateway Failures**: Gracefully degrade when gateway unavailable
4. **Secure Tokens**: Never commit access tokens, use environment variables
5. **Monitor Memory Usage**: Track memory size and clean up old sessions
