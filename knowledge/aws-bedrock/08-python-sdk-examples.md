---
title: Amazon Bedrock Python SDK Examples
category: aws-bedrock
tags:
  - aws
  - bedrock
  - python
  - boto3
  - sdk
  - examples
  - code
---

# Amazon Bedrock Python SDK Examples

## Prerequisites

Before running examples, ensure:

- **AWS Account & Permissions**: User or role with proper Amazon Bedrock permissions
- **Boto3 Installation**: Install and configure the AWS SDK for Python
- **Credentials**: Set up credentials for programmatic access

```bash
pip install boto3
```

---

## List Available Foundation Models

Use the `ListFoundationModels` operation to discover available models:

```python
"""Lists the available Amazon Bedrock models in an AWS Region."""
import logging
import json
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_foundation_models(bedrock_client):
    """
    Gets a list of available Amazon Bedrock foundation models.
    :return: The list of available bedrock foundation models.
    """
    try:
        response = bedrock_client.list_foundation_models()
        models = response["modelSummaries"]
        logger.info("Got %s foundation models.", len(models))
        return models
    except ClientError:
        logger.error("Couldn't list foundation models.")
        raise

def main():
    """Entry point for the example."""
    aws_region = "us-east-1"
    bedrock_client = boto3.client(service_name="bedrock", region_name=aws_region)

    fm_models = list_foundation_models(bedrock_client)
    for model in fm_models:
        print(f"Model: {model['modelName']}")
        print(json.dumps(model, indent=2))
        print("---------------------------\n")

if __name__ == "__main__":
    main()
```

---

## InvokeModel: Text Generation

Use `InvokeModel` to submit prompts with native model structure:

```python
import boto3
import json
from botocore.exceptions import ClientError

# Create an Amazon Bedrock Runtime client
brt = boto3.client("bedrock-runtime")

# Set the model ID
model_id = "amazon.titan-text-express-v1"

# Define the prompt
prompt = "Describe the purpose of a 'hello world' program in one line."

# Format the request payload using the model's native structure
native_request = {
    "inputText": prompt,
    "textGenerationConfig": {
        "maxTokenCount": 512,
        "temperature": 0.5,
        "topP": 0.9
    },
}

# Convert to JSON
request = json.dumps(native_request)

try:
    # Invoke the model
    response = brt.invoke_model(modelId=model_id, body=request)
except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Decode response body
model_response = json.loads(response["body"].read())

# Extract and print response text
response_text = model_response["results"][0]["outputText"]
print(response_text)
```

---

## Converse API (Recommended)

Use `Converse` for unified inference and multi-turn conversations:

```python
import boto3
from botocore.exceptions import ClientError

# Create an Amazon Bedrock Runtime client
brt = boto3.client("bedrock-runtime")

# Set the model ID
model_id = "amazon.titan-text-express-v1"

# Start a conversation
user_message = "Describe the purpose of a 'hello world' program in one line."
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send message using Converse API with inference configuration
    response = brt.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print response text
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)
except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
```

---

## Multi-Turn Conversation

```python
import boto3
from botocore.exceptions import ClientError

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
assistant_message = response["output"]["message"]
messages.append(assistant_message)
print(f"Assistant: {assistant_message['content'][0]['text']}\n")

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

print(f"Assistant: {response['output']['message']['content'][0]['text']}")
```

---

## Streaming Response

```python
import boto3
from botocore.exceptions import ClientError

brt = boto3.client("bedrock-runtime")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

messages = [
    {
        "role": "user",
        "content": [{"text": "Write a short poem about coding."}]
    }
]

try:
    response = brt.converse_stream(
        modelId=model_id,
        messages=messages,
        inferenceConfig={"maxTokens": 500}
    )

    # Process streaming response
    for event in response['stream']:
        if 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']
            if 'text' in delta:
                print(delta['text'], end='', flush=True)
        elif 'messageStop' in event:
            print()  # New line at end

except ClientError as e:
    print(f"ERROR: {e.response['Error']['Message']}")
```

---

## Using Claude Models

### Claude with System Prompt

```python
import boto3

brt = boto3.client("bedrock-runtime")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

response = brt.converse(
    modelId=model_id,
    messages=[
        {
            "role": "user",
            "content": [{"text": "Explain quantum computing in simple terms."}]
        }
    ],
    system=[
        {"text": "You are a science teacher explaining concepts to a 10-year-old. Use simple analogies and avoid technical jargon."}
    ],
    inferenceConfig={
        "maxTokens": 500,
        "temperature": 0.7,
        "topP": 0.9
    }
)

print(response["output"]["message"]["content"][0]["text"])
```

### Claude with Images

```python
import boto3
import base64

brt = boto3.client("bedrock-runtime")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Read and encode image
with open("image.png", "rb") as f:
    image_bytes = base64.standard_b64encode(f.read()).decode("utf-8")

response = brt.converse(
    modelId=model_id,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": image_bytes}
                    }
                },
                {"text": "What's in this image?"}
            ]
        }
    ],
    inferenceConfig={"maxTokens": 200}
)

print(response["output"]["message"]["content"][0]["text"])
```

---

## Using Amazon Nova Models

```python
import boto3

brt = boto3.client("bedrock-runtime")
model_id = "amazon.nova-lite-v1:0"

response = brt.converse(
    modelId=model_id,
    messages=[
        {
            "role": "user",
            "content": [{"text": "What are the benefits of cloud computing?"}]
        }
    ],
    inferenceConfig={
        "maxTokens": 300,
        "temperature": 0.5
    }
)

print(response["output"]["message"]["content"][0]["text"])
```

---

## Error Handling Best Practices

```python
import boto3
from botocore.exceptions import ClientError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def invoke_with_retry(model_id, messages, max_retries=3):
    """Invoke model with retry logic."""
    brt = boto3.client("bedrock-runtime")

    for attempt in range(max_retries):
        try:
            response = brt.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig={"maxTokens": 500}
            )
            return response["output"]["message"]["content"][0]["text"]

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']

            if error_code == 'ThrottlingException':
                logger.warning(f"Throttled, attempt {attempt + 1}/{max_retries}")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
            elif error_code == 'ModelTimeoutException':
                logger.warning(f"Timeout, attempt {attempt + 1}/{max_retries}")
            elif error_code == 'ValidationException':
                logger.error(f"Validation error: {error_message}")
                raise
            else:
                logger.error(f"Client error: {error_code} - {error_message}")
                raise

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    raise Exception("Max retries exceeded")

# Usage
result = invoke_with_retry(
    "anthropic.claude-3-haiku-20240307-v1:0",
    [{"role": "user", "content": [{"text": "Hello!"}]}]
)
print(result)
```

---

## Key Differences: InvokeModel vs Converse

| Feature | InvokeModel | Converse |
|---------|-------------|----------|
| **Use Case** | Native model-specific requests | Unified inference API |
| **Multi-turn** | Manual management | Simplified |
| **Recommendation** | Legacy | **Recommended** |
| **Request Format** | Model-specific native structure | Unified conversation format |
| **Tool Use** | Model-specific | Standardized |

---

## Additional Resources

- [Boto3 Bedrock Runtime Reference](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html)
- [Converse API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
- [InvokeModel API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html)