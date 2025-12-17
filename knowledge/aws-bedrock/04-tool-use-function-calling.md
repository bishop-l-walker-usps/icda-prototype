---
title: Amazon Bedrock Tool Use (Function Calling)
category: aws-bedrock
tags:
  - aws
  - bedrock
  - tool-use
  - function-calling
  - agents
  - converse-api
---

# Tool Use (Function Calling) in Amazon Bedrock

## Overview

Tool use in Amazon Bedrock allows models to request tools to help generate responses. The model doesn't directly call tools; instead, it responds with a request for the user to call a tool with specific parameters.

## Tool Use Workflow

1. **Supply tool definition(s)** when sending a message to the model
2. **Model determines** if it needs a tool and responds with a request
3. **Include required input parameters** in the model's response
4. **You implement and call** the tool on the model's behalf
5. **Continue conversation** by supplying tool results to the model
6. **Model generates final response** incorporating tool results

## Recommended API

**Converse API** (recommended for tool use):
- `Converse` - Standard inference operation
- `ConverseStream` - Streaming response operation
- Provides consistent API across all tool-supporting models

---

## Tool Definition Structure

### Tool Configuration

```json
{
    "tools": [
        {
            "toolSpec": {
                "name": "top_song",
                "description": "Get the most popular song played on a radio station.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "sign": {
                                "type": "string",
                                "description": "The call sign for the radio station"
                            }
                        },
                        "required": ["sign"]
                    }
                }
            }
        }
    ]
}
```

### Tool Definition Fields

- **name** - Unique identifier for the tool
- **description** - Clear description of what the tool does
- **inputSchema** - JSON Schema defining required parameters

---

## Complete Python Example (Synchronous)

```python
import logging
import json
import boto3
from botocore.exceptions import ClientError

class StationNotFoundError(Exception):
    """Raised when a radio station isn't found."""
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_top_song(call_sign):
    """Returns the most popular song for the requested station."""
    if call_sign == 'WZPZ':
        return "Elemental Hotel", "8 Storey Hike"
    else:
        raise StationNotFoundError(f"Station {call_sign} not found.")

def generate_text(bedrock_client, model_id, tool_config, input_text):
    """Generates text using tool use."""
    logger.info("Generating text with model %s", model_id)

    # Create initial message
    messages = [{
        "role": "user",
        "content": [{"text": input_text}]
    }]

    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        toolConfig=tool_config
    )

    output_message = response['output']['message']
    messages.append(output_message)
    stop_reason = response['stopReason']

    if stop_reason == 'tool_use':
        # Tool use requested - call the tool
        tool_requests = response['output']['message']['content']
        for tool_request in tool_requests:
            if 'toolUse' in tool_request:
                tool = tool_request['toolUse']
                logger.info("Requesting tool %s. Request: %s",
                            tool['name'], tool['toolUseId'])

                if tool['name'] == 'top_song':
                    tool_result = {}
                    try:
                        song, artist = get_top_song(tool['input']['sign'])
                        tool_result = {
                            "toolUseId": tool['toolUseId'],
                            "content": [{"json": {"song": song, "artist": artist}}]
                        }
                    except StationNotFoundError as err:
                        tool_result = {
                            "toolUseId": tool['toolUseId'],
                            "content": [{"text": err.args[0]}],
                            "status": 'error'
                        }

                    tool_result_message = {
                        "role": "user",
                        "content": [{"toolResult": tool_result}]
                    }
                    messages.append(tool_result_message)

                    # Send tool result to model
                    response = bedrock_client.converse(
                        modelId=model_id,
                        messages=messages,
                        toolConfig=tool_config
                    )
                    output_message = response['output']['message']

    # Print final response
    for content in output_message['content']:
        print(json.dumps(content, indent=4))

def main():
    model_id = "cohere.command-r-v1:0"
    input_text = "What is the most popular song on WZPZ?"

    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "top_song",
                    "description": "Get the most popular song played on a radio station.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "sign": {
                                    "type": "string",
                                    "description": "The call sign for the radio station"
                                }
                            },
                            "required": ["sign"]
                        }
                    }
                }
            }
        ]
    }

    bedrock_client = boto3.client(service_name='bedrock-runtime')

    try:
        print(f"Question: {input_text}")
        generate_text(bedrock_client, model_id, tool_config, input_text)
    except ClientError as err:
        logger.error("A client error occurred: %s", err.response['Error']['Message'])

if __name__ == "__main__":
    main()
```

---

## Complete Python Example (Streaming)

```python
import logging
import json
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class StationNotFoundError(Exception):
    pass

def get_top_song(call_sign):
    if call_sign == 'WZPZ':
        return "Elemental Hotel", "8 Storey Hike"
    raise StationNotFoundError(f"Station {call_sign} not found.")

def stream_messages(bedrock_client, model_id, messages, tool_config):
    """Sends a message and streams the response."""
    response = bedrock_client.converse_stream(
        modelId=model_id,
        messages=messages,
        toolConfig=tool_config
    )

    stop_reason = ""
    message = {'content': []}
    text = ''
    tool_use = {}

    for chunk in response['stream']:
        if 'messageStart' in chunk:
            message['role'] = chunk['messageStart']['role']
        elif 'contentBlockStart' in chunk:
            tool = chunk['contentBlockStart']['start']['toolUse']
            tool_use['toolUseId'] = tool['toolUseId']
            tool_use['name'] = tool['name']
        elif 'contentBlockDelta' in chunk:
            delta = chunk['contentBlockDelta']['delta']
            if 'toolUse' in delta:
                if 'input' not in tool_use:
                    tool_use['input'] = ''
                tool_use['input'] += delta['toolUse']['input']
            elif 'text' in delta:
                text += delta['text']
                print(delta['text'], end='')
        elif 'contentBlockStop' in chunk:
            if 'input' in tool_use:
                tool_use['input'] = json.loads(tool_use['input'])
                message['content'].append({'toolUse': tool_use})
                tool_use = {}
            else:
                message['content'].append({'text': text})
                text = ''
        elif 'messageStop' in chunk:
            stop_reason = chunk['messageStop']['stopReason']

    return stop_reason, message

def main():
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    input_text = "What is the most popular song on WZPZ?"

    bedrock_client = boto3.client(service_name='bedrock-runtime')

    messages = [{"role": "user", "content": [{"text": input_text}]}]

    tool_config = {
        "tools": [{
            "toolSpec": {
                "name": "top_song",
                "description": "Get the most popular song played on a radio station.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "sign": {"type": "string", "description": "Radio station call sign"}
                        },
                        "required": ["sign"]
                    }
                }
            }
        }]
    }

    try:
        stop_reason, message = stream_messages(
            bedrock_client, model_id, messages, tool_config)
        messages.append(message)

        if stop_reason == "tool_use":
            for content in message['content']:
                if 'toolUse' in content:
                    tool = content['toolUse']
                    if tool['name'] == 'top_song':
                        try:
                            song, artist = get_top_song(tool['input']['sign'])
                            tool_result = {
                                "toolUseId": tool['toolUseId'],
                                "content": [{"json": {"song": song, "artist": artist}}]
                            }
                        except StationNotFoundError as err:
                            tool_result = {
                                "toolUseId": tool['toolUseId'],
                                "content": [{"text": str(err)}],
                                "status": 'error'
                            }

                        messages.append({
                            "role": "user",
                            "content": [{"toolResult": tool_result}]
                        })

        stop_reason, message = stream_messages(
            bedrock_client, model_id, messages, tool_config)

    except ClientError as err:
        logger.error("Client error: %s", err.response['Error']['Message'])

if __name__ == "__main__":
    main()
```

---

## Key Structures

### Tool Result Success

```python
tool_result = {
    "toolUseId": tool['toolUseId'],
    "content": [{"json": {"song": song, "artist": artist}}]
}
```

### Tool Result Error

```python
tool_result = {
    "toolUseId": tool['toolUseId'],
    "content": [{"text": err.args[0]}],
    "status": 'error'
}
```

### Tool Result Message

```python
tool_result_message = {
    "role": "user",
    "content": [{"toolResult": tool_result}]
}
```

---

## Differences: Converse vs. ConverseStream

| Aspect | Converse | ConverseStream |
|--------|----------|----------------|
| **Execution** | Synchronous | Asynchronous streaming |
| **Response** | Complete response object | Stream of chunks |
| **Tool Handling** | Same logic | Same logic |

---

## Best Practices

1. **Clear Tool Descriptions** - Write detailed descriptions for accurate tool selection
2. **Proper Error Handling** - Return errors with `status: 'error'` for graceful handling
3. **Parameter Validation** - Validate tool inputs before execution
4. **Conversation History** - Maintain full message history for context
5. **Streaming for UX** - Use ConverseStream for better user experience