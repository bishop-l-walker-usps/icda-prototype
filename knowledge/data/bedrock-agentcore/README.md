# Amazon Bedrock AgentCore - Knowledge Base

This folder contains comprehensive documentation and examples for Amazon Bedrock AgentCore Memory system, extracted for RAG indexing and reference.

## Contents

### `/memory/`
- `getting-started.md` - Quick start guide for AgentCore Memory
- `long-term-memory.md` - Enabling and configuring long-term memory strategies
- `memory-types.md` - Short-term vs long-term memory comparison
- `strategies.md` - Memory strategies (semantic, user preference, summary)

### `/examples/`
- `semantic-search.md` - Semantic search implementation example
- `session-management.md` - Session state persistence example
- `memory-gateway-agent.md` - Memory + Gateway integration example
- `complete-workflow.md` - End-to-end implementation workflow

### `/sdk/`
- `memory-manager.md` - MemoryManager class documentation
- `session-manager.md` - MemorySessionManager documentation
- `starter-toolkit.md` - Starter toolkit installation and usage

### `/api/`
- `data-plane.md` - BedrockAgentCore API reference (40+ operations)
- `control-plane.md` - BedrockAgentCoreControl API reference

## Quick Reference

### Installation
```bash
pip install bedrock-agentcore
pip install bedrock-agentcore-starter-toolkit
```

### Key Classes
| Class | Purpose |
|-------|---------|
| `MemoryManager` | Create, list, get, delete memory resources |
| `MemorySessionManager` | Manage sessions within memory resources |
| `ConversationalMessage` | Message objects for events |
| `SemanticStrategy` | Long-term memory extraction configuration |
| `UserPreferenceStrategy` | User preference extraction strategy |
| `SummaryStrategy` | Session summarization strategy |

### Memory Types
- **Short-term Memory (STM)**: Raw conversation turns, session-scoped, 7-day retention
- **Long-term Memory (LTM)**: Extracted facts/preferences, cross-session, 30-day retention

## Source Links
- [AWS Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-get-started.html)
- [Starter Toolkit GitHub](https://github.com/aws/bedrock-agentcore-starter-toolkit)
- [boto3 Data Plane](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agentcore.html)
- [boto3 Control Plane](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agentcore-control.html)

---
**Last Updated**: 2025-12-29
**Purpose**: Memory system reference for ICDA prototype agent development
