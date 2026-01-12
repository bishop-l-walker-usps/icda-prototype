---
title: Amazon Bedrock Converse API
category: aws-bedrock
tags:
  - aws
  - bedrock
  - converse
  - api
  - multi-turn
  - conversations
  - streaming
---

# Converse API in Amazon Bedrock

## Overview

The **Converse API** is Amazon Bedrock's recommended interface for building conversational applications. It enables you to send and receive messages to and from Amazon Bedrock models while maintaining conversation context across multiple turns.

### Use Cases
- Chat bots with multi-turn conversations
- Applications with persona or tone customization
- Technical support assistants

## Core Operations

The Converse API provides two main operations:

1. **Converse** - Standard request/response (requires `bedrock:InvokeModel` permission)
2. **ConverseStream** - Streaming responses (requires `bedrock:InvokeModelWithResponseStream` permission)

### Alternative (Legacy) Operations
- `InvokeModel`
- `InvokeModelWithResponseStream`

> **Recommendation**: Use Converse API as it provides a consistent interface across all supported Bedrock models, allowing you to write code once and use it with different models.

## Key Features

- **Multi-turn conversations** - Maintain context across message exchanges
- **Tool use** - Integrate tool/function calling for extended functionality
- **Guardrails** - Apply safety guardrails to conversations
- **Streaming** - Real-time response streaming via ConverseStream
- **Model-specific parameters** - Pass unique inference parameters

---

## Request Structure

### Required Parameters

- **modelId** - Resource identifier for inference (required, in header)

### Message Structure

#### messages
Array of Message objects containing:
- **role** - `user` or `assistant`
- **content** - Array of ContentBlock objects

```json
[
    {
        "role": "user",
        "content": [
            {
                "text": "Create a list of 3 pop songs."
            }
        ]
    }
]
```

#### system
Defines instructions or context for the model:

```json
[
    {
        "text": "You are an app that creates playlists for a radio station that plays rock and pop music. Only return song names and the artist."
    }
]
```

#### inferenceConfig
Common inference parameters:
- **maxTokens** - Maximum tokens in generated response
- **stopSequences** - Sequences that cause model to stop generating
- **temperature** - Likelihood of selecting higher-probability options (0.0-1.0)
- **topP** - Percentage of most-likely candidates for next token

```json
{"temperature": 0.5, "maxTokens": 512, "topP": 0.9}
```

#### additionalModelRequestFields
Model-specific inference parameters (e.g., `top_k` for Anthropic Claude):

```json
{"top_k": 200}
```

---

## Content Block Types

### text
Plain text prompt:

```json
{
    "role": "user",
    "content": [
        {
            "text": "Create a list of 3 pop songs."
        }
    ]
}
```

### image
Image content with base64 encoding or S3 URI:

```json
{
    "role": "user",
    "content": [
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": "image in bytes"
                }
            }
        }
    ]
}
```

### document
Document content (PDF, etc.) with required accompanying text:

```json
{
    "role": "user",
    "content": [
        {
            "text": "Summarize this document"
        },
        {
            "document": {
                "format": "pdf",
                "name": "MyDocument",
                "source": {
                    "bytes": "document in bytes"
                }
            }
        }
    ]
}
```

### video
Video content:

```json
{
    "role": "user",
    "content": [
        {
            "video": {
                "format": "mp4",
                "source": {
                    "s3Location": {
                        "uri": "s3://bucket/myVideo"
                    }
                }
            }
        }
    ]
}
```

---

## Response Structure

### Converse Response

```json
{
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {
                    "text": "Here is a list of 3 pop songs..."
                }
            ]
        }
    },
    "stopReason": "end_turn",
    "usage": {
        "inputTokens": 125,
        "outputTokens": 60,
        "totalTokens": 185
    },
    "metrics": {
        "latencyMs": 1175
    }
}
```

**Response Fields:**
- **output** - Generated message
- **stopReason** - Why model stopped (`end_turn`, `max_tokens`, `tool_use`)
- **usage** - Token counts (input, output, total)
- **metrics** - Performance metrics (latencyMs)

### ConverseStream Response Events

Streaming response with sequential events:

1. **messageStart** - Message start with role
2. **contentBlockStart** - Content block start
3. **contentBlockDelta** - Partial content (text, toolUse, reasoningContent)
4. **contentBlockStop** - Content block end
5. **messageStop** - Message end with stopReason
6. **metadata** - Token usage and metrics

---

## Python Code Examples

### Basic Converse API

```python
import boto3
from botocore.exceptions import ClientError

# Create client
brt = boto3.client("bedrock-runtime")

# Set model
model_id = "amazon.titan-text-express-v1"

# Create conversation
user_message = "Describe the purpose of a 'hello world' program in one line."
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    response = brt.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)
except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
```

### Multi-Turn Conversation

```python
import boto3

brt = boto3.client("bedrock-runtime")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Initialize conversation
messages = []

# System prompt
system = [{"text": "You are a helpful assistant that answers questions concisely."}]

# First turn
messages.append({
    "role": "user",
    "content": [{"text": "What is Python?"}]
})

response = brt.converse(
    modelId=model_id,
    messages=messages,
    system=system,
    inferenceConfig={"maxTokens": 200}
)

# Add assistant response to conversation
messages.append(response["output"]["message"])

# Second turn
messages.append({
    "role": "user",
    "content": [{"text": "What are its main uses?"}]
})

response = brt.converse(
    modelId=model_id,
    messages=messages,
    system=system,
    inferenceConfig={"maxTokens": 200}
)

print(response["output"]["message"]["content"][0]["text"])
```

---

## Key Differences: InvokeModel vs Converse

| Feature | InvokeModel | Converse |
|---------|-------------|----------|
| **Use Case** | Native model-specific requests | Unified inference API |
| **Multi-turn** | Manual management | Simplified |
| **Recommendation** | Legacy | **Recommended** |
| **Request Format** | Model-specific | Unified conversation format |
| **Tool Use** | Model-specific | Standardized |

---

## Content Restrictions

- Maximum 20 images (each <= 3.75 MB, 8,000 px x 8,000 px)
- Maximum 5 documents (each <= 4.5 MB)
- Images and documents only with `role: "user"`

---

## Additional Resources

- [Converse API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- [ConverseStream API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html)
- [Tool Use Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html)
- [Guardrails Integration](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html)