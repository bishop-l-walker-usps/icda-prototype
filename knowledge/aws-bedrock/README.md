---
title: AWS Bedrock Knowledge Base
category: aws-bedrock
tags:
  - aws
  - bedrock
  - llm
  - foundation-models
  - ai
---

# AWS Bedrock Knowledge Base

Comprehensive documentation for Amazon Bedrock, AWS's fully managed foundation model service. This knowledge base is automatically indexed by the RAG system for agent knowledge retrieval.

## Contents

| File | Description |
|------|-------------|
| `01-overview-getting-started.md` | Service overview, prerequisites, IAM setup, account configuration |
| `02-supported-models.md` | Complete list of foundation models: Claude, Titan, Llama, Nova, Mistral, etc. |
| `03-converse-api.md` | Converse API for multi-turn conversations, message formats, streaming |
| `04-tool-use-function-calling.md` | Tool use/function calling with complete Python examples |
| `05-agents.md` | Amazon Bedrock Agents: action groups, orchestration, deployment |
| `06-knowledge-bases-rag.md` | Knowledge bases, RAG architecture, vector stores, embeddings |
| `07-guardrails.md` | Content filtering, PII detection, denied topics, safety controls |
| `08-python-sdk-examples.md` | Complete Python/boto3 code examples for all major operations |
| `09-prompt-engineering.md` | Prompt engineering best practices, few-shot, templates |

## Quick Reference

### Key Model IDs

| Provider | Model | Model ID |
|----------|-------|----------|
| Amazon | Nova Pro | `amazon.nova-pro-v1:0` |
| Amazon | Nova Lite | `amazon.nova-lite-v1:0` |
| Amazon | Titan Text Embeddings V2 | `amazon.titan-embed-text-v2:0` |
| Anthropic | Claude 3.7 Sonnet | `anthropic.claude-3-7-sonnet-20250219-v1:0` |
| Anthropic | Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` |
| Meta | Llama 3.3 70B | `meta.llama3-3-70b-instruct-v1:0` |

### Quick Start Code

```python
import boto3

# Create client
brt = boto3.client("bedrock-runtime")

# Simple inference
response = brt.converse(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    messages=[
        {"role": "user", "content": [{"text": "Hello!"}]}
    ],
    inferenceConfig={"maxTokens": 200}
)

print(response["output"]["message"]["content"][0]["text"])
```

## Usage with RAG

These documents are automatically indexed when:
1. The MCP Knowledge Server starts
2. The main ICDA application starts
3. A document is uploaded via MCP tools

### Search Examples

```python
# Search for Bedrock API information
search_knowledge(query="how to use converse api", tags=["aws-bedrock"])

# Search for tool use examples
search_knowledge(query="tool use function calling python", tags=["tool-use"])

# Search for guardrails configuration
search_knowledge(query="PII detection guardrails", tags=["guardrails"])
```

## Related ICDA Modules

| Module | Integration |
|--------|-------------|
| `icda/nova.py` | Bedrock Nova client with Converse API |
| `icda/embeddings.py` | Titan embedding client |
| `icda/agents/nova_agent.py` | 8-agent query orchestrator |

## Source Documentation

All content sourced from official AWS documentation:
- https://docs.aws.amazon.com/bedrock/latest/userguide/

Last Updated: December 2025