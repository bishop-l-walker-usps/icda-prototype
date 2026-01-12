# Bedrock AgentCore Starter Toolkit

## Overview

The Bedrock AgentCore Starter Toolkit is a CLI and Python library enabling deployment of AI agents to Amazon Bedrock AgentCore with zero infrastructure management.

## Installation

### Recommended (uv)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install bedrock-agentcore-starter-toolkit
```

### Alternative (pip)
```bash
pip install bedrock-agentcore-starter-toolkit
```

### With All Features
```bash
pip install bedrock-agentcore-starter-toolkit[all]
```

## Quick Start

### Create New Project
```bash
agentcore create my-agent
cd my-agent
```

This scaffolds a new project with:
- Agent implementation file
- Requirements.txt with dependencies
- Configuration template

### Configure Agent
```bash
agentcore configure --entrypoint agent.py
```

### Deploy
```bash
agentcore deploy
```

### Invoke
```bash
agentcore invoke '{"message": "Hello!"}'
```

## Core Features

### 1. Runtime
Serverless deployment for AI agents with:
- Extended runtime support (beyond Lambda limits)
- Fast cold starts
- Automatic scaling
- Session isolation

### 2. Memory
Context-aware agents with:
- Short-term conversation memory (session-scoped)
- Long-term shared memory (cross-session)
- Semantic search and retrieval
- Multiple extraction strategies

### 3. Gateway
Managed Model Context Protocol (MCP) server:
- Convert APIs into agent-accessible tools
- OAuth2 authentication
- Tool discovery via MCP
- Secure access control

### 4. Code Interpreter
Secure, isolated code execution:
- Python execution environment
- Package installation support
- Sandbox isolation

### 5. Browser
Cloud-based browser runtime:
- Website interaction at scale
- Headless browser automation
- Session management

### 6. Observability
Production monitoring with:
- OpenTelemetry-compatible tracing
- CloudWatch integration
- Performance metrics

### 7. Evaluation
Agent performance measurement:
- Built-in evaluators
- Custom evaluation support
- Quality metrics

### 8. Identity
Agent identity management:
- AWS IAM integration
- OAuth2 support
- Existing provider compatibility

### 9. Policy
Deterministic action control:
- Cedar policy language
- Fine-grained permissions
- Audit logging

## CLI Commands

### Memory Operations

```bash
# Create memory
agentcore memory create MyMemory \
  --region us-west-2 \
  --description "Agent memory store" \
  --strategies '[{"semanticMemoryStrategy": {"name": "facts", "namespaces": ["/facts/{actorId}"]}}]' \
  --wait

# List memories
agentcore memory list --region us-west-2

# Get memory details
agentcore memory get <memory-id> --region us-west-2

# Delete memory
agentcore memory delete <memory-id> --region us-west-2 --wait
```

### Runtime Operations

```bash
# Configure agent entrypoint
agentcore configure --entrypoint agent.py

# Deploy agent
agentcore deploy

# Get deployment status
agentcore status

# View logs
agentcore logs

# Invoke agent
agentcore invoke '{"message": "Hello"}' --session-id session1

# Destroy deployment
agentcore destroy
```

### Gateway Operations

```bash
# Create gateway
agentcore gateway create MyGateway --region us-west-2

# List gateways
agentcore gateway list --region us-west-2

# Add tool to gateway
agentcore gateway add-tool <gateway-id> --tool-config tool.json

# Delete gateway
agentcore gateway delete <gateway-id> --region us-west-2
```

## Python SDK

### MemoryManager

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import (
    SemanticStrategy,
    UserPreferenceStrategy,
    SummaryStrategy
)

# Initialize
memory_manager = MemoryManager(region_name="us-west-2")

# Create memory
memory = memory_manager.get_or_create_memory(
    name="AgentMemory",
    strategies=[
        SemanticStrategy(
            name="facts",
            namespaces=['/facts/{actorId}']
        )
    ]
)

# List memories
memories = memory_manager.list_memories()

# Get strategies
strategies = memory_manager.get_memory_strategies(memoryId=memory['id'])

# Add strategy
memory_manager.add_user_preference_strategy(
    memory_id=memory['id'],
    name="prefs",
    namespaces=['/prefs/{actorId}']
)

# Delete memory
memory_manager.delete_memory(memory_id=memory['id'])
```

### GatewayManager

```python
from bedrock_agentcore_starter_toolkit.operations.gateway.manager import GatewayManager

# Initialize
gateway_manager = GatewayManager(region_name="us-west-2")

# Create OAuth authorizer
authorizer = gateway_manager.create_oauth_authorizer_with_cognito(
    name="MyAuth",
    client_name="agent-client"
)

# Create gateway
gateway = gateway_manager.create_gateway(
    name="ToolGateway",
    tools=[{
        "name": "calculator",
        "description": "Math operations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "a": {"type": "number"},
                "b": {"type": "number"}
            }
        }
    }],
    authorizer_id=authorizer['id']
)

# Get access token
token = gateway_manager.get_access_token(
    gateway_id=gateway['id'],
    client_id=authorizer['clientId'],
    client_secret=authorizer['clientSecret']
)
```

### ObservabilityManager

```python
from bedrock_agentcore_starter_toolkit.operations.observability.delivery import ObservabilityDeliveryManager

obs_manager = ObservabilityDeliveryManager(region_name="us-west-2")

# Enable CloudWatch logging
obs_manager.enable_cloudwatch(
    memory_id=memory['id'],
    log_group="/aws/agentcore/memory"
)
```

## Agent Framework Integration

### Strands Agents

```python
from strands import Agent
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def handler(payload, context):
    agent = Agent(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    )
    response = agent.invoke(payload.get("message"))
    return {"response": response}

if __name__ == "__main__":
    app.run()
```

### LangGraph

```python
from langgraph.graph import StateGraph
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def handler(payload, context):
    graph = StateGraph(...)  # Your graph definition
    result = graph.invoke(payload)
    return result
```

### CrewAI

```python
from crewai import Crew, Agent, Task
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def handler(payload, context):
    crew = Crew(
        agents=[...],
        tasks=[...]
    )
    result = crew.kickoff()
    return {"result": result}
```

## Environment Variables

```bash
# Required
export AWS_REGION="us-west-2"
export AWS_PROFILE="default"

# Memory (if using)
export MEMORY_ID="mem-abc123"

# Gateway (if using)
export GATEWAY_URL="https://..."
export GATEWAY_TOKEN="..."

# Session
export SESSION_ID="session-001"
export ACTOR_ID="user-123"
```

## Project Structure

```
my-agent/
├── agent.py              # Main agent implementation
├── requirements.txt      # Dependencies
├── agentcore.yaml        # AgentCore configuration
├── gateway_config.json   # Gateway credentials (generated)
└── tests/
    └── test_agent.py     # Agent tests
```

## Dependencies

### Core
```
bedrock-agentcore>=0.1.0
bedrock-agentcore-starter-toolkit>=0.1.0
```

### Agent Frameworks
```
strands-agents>=0.1.0
langchain>=0.2.0
langgraph>=0.2.0
crewai>=0.1.0
```

### Gateway
```
mcp>=0.1.0
```

## Resources

- [Documentation](https://docs.aws.amazon.com/bedrock-agentcore/)
- [GitHub Repository](https://github.com/aws/bedrock-agentcore-starter-toolkit)
- [Discord Community](https://discord.gg/bedrockagentcore-preview)
- [Examples](https://github.com/aws/bedrock-agentcore-starter-toolkit/tree/main/documentation/docs/examples)

## License

Apache 2.0
