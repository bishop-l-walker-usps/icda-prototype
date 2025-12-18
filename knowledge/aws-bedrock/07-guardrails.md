---
title: Amazon Bedrock Guardrails
category: aws-bedrock
tags:
  - aws
  - bedrock
  - guardrails
  - safety
  - content-filtering
  - pii
  - responsible-ai
---

# Amazon Bedrock Guardrails

## Overview

Amazon Bedrock Guardrails provides configurable safeguards for generative AI applications based on use cases and responsible AI policies. You can create multiple guardrails tailored to different use cases and apply them across multiple foundation models (FMs), providing consistent user experience and standardizing safety and privacy controls.

## Use Cases

- **Chatbot applications** - Filter harmful user inputs and toxic model responses
- **Banking applications** - Block queries or responses associated with investment advice
- **Call center applications** - Redact personally identifiable information (PII) in conversation transcripts

---

## Safeguards (Policies)

### Content Filters

Detect and filter harmful text or image content in input prompts or model responses.

**Predefined harmful content categories:**
- Hate
- Insults
- Sexual
- Violence
- Misconduct
- Prompt Attack

**Configuration:**
- Adjustable filter strength for each category
- **Standard tier**: Extended protection for harmful content in code elements (comments, variable/function names, string literals)

### Denied Topics

Define undesirable topics in your application context.
- Blocks detected topics in user queries or model responses
- Custom topic definitions
- Extended protection in code elements (Standard tier)

### Word Filters

Block undesirable words, phrases, and profanity (exact match).
- Offensive terms
- Competitor names
- Custom word lists

### Sensitive Information Filters

Block or mask sensitive information (PII, custom regex).

**Probabilistic detection of standard formats:**
- SSN numbers
- Date of Birth
- Addresses
- Credit card numbers
- Phone numbers
- Email addresses
- Custom regex patterns for identifiers

### Contextual Grounding Checks

Detect and filter hallucinations in model responses.
- Validate grounding in source material
- Ensure relevance to user query
- Reference-based validation

### Automated Reasoning Checks

Validate foundation model response accuracy against logical rules.
- Detect hallucinations
- Suggest corrections
- Highlight unstated assumptions

---

## Configuration & Testing

### Development Workflow

1. **Working Draft** - Automatically created for iterative modification
2. **Test Window** - Built-in testing to validate configurations
3. **Versions** - Create versions when satisfied
4. **Deployment** - Apply versions to supported foundation models

### Testing Process

1. Define test cases representing your use case
2. Run test cases against guardrail configuration
3. Review results and adjust filter strengths
4. Iterate until requirements are met

---

## API Usage

### Direct FM Integration

Specify guardrail ID and version during inference API invocation:

```python
import boto3

client = boto3.client('bedrock-runtime')

response = client.converse(
    modelId='anthropic.claude-3-haiku-20240307-v1:0',
    messages=[
        {
            'role': 'user',
            'content': [{'text': 'Tell me about investment strategies'}]
        }
    ],
    guardrailConfig={
        'guardrailIdentifier': 'GUARDRAIL_ID',
        'guardrailVersion': '1'
    }
)
```

### ApplyGuardrail API

Use guardrails directly without invoking foundation models:

```python
response = client.apply_guardrail(
    guardrailIdentifier='GUARDRAIL_ID',
    guardrailVersion='1',
    source='INPUT',  # or 'OUTPUT'
    content=[
        {
            'text': {
                'text': 'Content to evaluate'
            }
        }
    ]
)

# Check if content was blocked
if response['action'] == 'GUARDRAIL_INTERVENED':
    print("Content blocked by guardrail")
    print(response['outputs'][0]['text'])
```

### Selective Evaluation (RAG/Conversational)

Apply tags to user input to selectively evaluate specific sections:

```python
messages = [
    {
        'role': 'user',
        'content': [
            {
                'text': 'System instructions here',
                'qualifiers': []  # Not evaluated
            },
            {
                'text': 'User input to evaluate',
                'qualifiers': ['guard_content']  # Evaluated
            }
        ]
    }
]
```

---

## Configuration Options

### Filter Strength Levels

| Level | Description |
|-------|-------------|
| None | No filtering |
| Low | Block only severe violations |
| Medium | Balanced filtering |
| High | Aggressive filtering |

### Custom Guardrail Messages

Configure custom messages returned when content is blocked:

```json
{
    "blockedInputMessaging": "I'm sorry, but I cannot process this request.",
    "blockedOutputsMessaging": "I apologize, but I cannot provide that information."
}
```

---

## Guardrail Response Structure

### When Content is Allowed

```json
{
    "action": "NONE",
    "outputs": [
        {
            "text": "Original content"
        }
    ],
    "assessments": [
        {
            "topicPolicy": {
                "topics": []
            },
            "contentPolicy": {
                "filters": []
            }
        }
    ]
}
```

### When Content is Blocked

```json
{
    "action": "GUARDRAIL_INTERVENED",
    "outputs": [
        {
            "text": "I'm sorry, but I cannot process this request."
        }
    ],
    "assessments": [
        {
            "topicPolicy": {
                "topics": [
                    {
                        "name": "Investment Advice",
                        "type": "DENY",
                        "action": "BLOCKED"
                    }
                ]
            }
        }
    ]
}
```

---

## PII Detection and Handling

### Detection Types

| PII Type | Description |
|----------|-------------|
| ADDRESS | Physical addresses |
| AGE | Age information |
| AWS_ACCESS_KEY | AWS access keys |
| AWS_SECRET_KEY | AWS secret keys |
| CREDIT_DEBIT_CARD_CVV | Card CVV numbers |
| CREDIT_DEBIT_CARD_EXPIRY | Card expiration dates |
| CREDIT_DEBIT_CARD_NUMBER | Card numbers |
| DRIVER_ID | Driver's license numbers |
| EMAIL | Email addresses |
| IP_ADDRESS | IP addresses |
| LICENSE_PLATE | Vehicle license plates |
| NAME | Person names |
| PASSWORD | Passwords |
| PHONE | Phone numbers |
| PIN | PIN numbers |
| SSN | Social Security Numbers |
| USERNAME | Usernames |
| VEHICLE_IDENTIFICATION_NUMBER | VINs |

### PII Actions

| Action | Description |
|--------|-------------|
| BLOCK | Block content containing PII |
| ANONYMIZE | Replace PII with placeholder |

---

## Best Practices

### Implementation

1. **Start with baseline** - Begin with medium filter strength
2. **Test thoroughly** - Use representative test cases
3. **Layer defenses** - Combine multiple safeguard types
4. **Monitor and iterate** - Review blocked content regularly

### Filter Configuration

1. **Content filters** - Adjust based on application context
2. **Denied topics** - Be specific in topic definitions
3. **PII handling** - Choose BLOCK or ANONYMIZE based on use case
4. **Word filters** - Update regularly for new terms

### Production

1. **Version control** - Create new versions for changes
2. **A/B testing** - Test guardrail changes before full deployment
3. **Logging** - Enable logging for audit and debugging
4. **Fallback handling** - Handle blocked content gracefully

---

## Additional Resources

- [How Guardrails Works](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-how.html)
- [Supported Regions and Models](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-supported.html)
- [Create Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-components.html)
- [Testing Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-test.html)
- [Deploy Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-deploy.html)