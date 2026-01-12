# BedrockAgentCore Data Plane API Reference

## Overview

The Amazon Bedrock AgentCore Data Plane API handles runtime data operations for memory, sessions, events, and agent invocations.

## boto3 Client

```python
import boto3

client = boto3.client('bedrock-agentcore', region_name='us-west-2')
```

## Memory Record Operations

### batch_create_memory_records

Create multiple memory records at once.

```python
response = client.batch_create_memory_records(
    memoryId='mem-abc123',
    records=[
        {
            'namespace': '/facts/user123',
            'content': 'User prefers Python',
            'metadata': {'source': 'conversation'}
        },
        {
            'namespace': '/facts/user123',
            'content': 'Order #12345 was discussed',
            'metadata': {'source': 'conversation'}
        }
    ]
)
```

### batch_update_memory_records

Update multiple existing records.

```python
response = client.batch_update_memory_records(
    memoryId='mem-abc123',
    updates=[
        {
            'recordId': 'rec-123',
            'content': 'Updated content'
        }
    ]
)
```

### batch_delete_memory_records

Delete multiple records.

```python
response = client.batch_delete_memory_records(
    memoryId='mem-abc123',
    recordIds=['rec-123', 'rec-456']
)
```

### get_memory_record

Retrieve a specific record.

```python
response = client.get_memory_record(
    memoryId='mem-abc123',
    recordId='rec-123'
)

record = response['record']
print(f"Content: {record['content']}")
```

### delete_memory_record

Delete a single record.

```python
client.delete_memory_record(
    memoryId='mem-abc123',
    recordId='rec-123'
)
```

### list_memory_records

List all records with optional filtering.

```python
response = client.list_memory_records(
    memoryId='mem-abc123',
    namespacePrefix='/facts/',
    maxResults=100
)

for record in response['records']:
    print(f"{record['namespace']}: {record['content']}")

# Handle pagination
while 'nextToken' in response:
    response = client.list_memory_records(
        memoryId='mem-abc123',
        namespacePrefix='/facts/',
        nextToken=response['nextToken']
    )
    for record in response['records']:
        print(f"{record['namespace']}: {record['content']}")
```

### retrieve_memory_records

Query memory records (semantic search).

```python
response = client.retrieve_memory_records(
    memoryId='mem-abc123',
    query='order delivery problem',
    namespacePrefix='/',
    topK=5
)

for record in response['records']:
    print(f"Score: {record['score']}, Content: {record['content']}")
```

## Memory Extraction Jobs

### start_memory_extraction_job

Trigger extraction from conversation events.

```python
response = client.start_memory_extraction_job(
    memoryId='mem-abc123',
    strategyName='semanticLongTermMemory'
)

job_id = response['jobId']
print(f"Started extraction job: {job_id}")
```

### list_memory_extraction_jobs

List extraction jobs.

```python
response = client.list_memory_extraction_jobs(
    memoryId='mem-abc123',
    status='COMPLETED'
)

for job in response['jobs']:
    print(f"Job {job['jobId']}: {job['status']}")
```

## Event Operations

### create_event

Create a conversation event.

```python
response = client.create_event(
    memoryId='mem-abc123',
    sessionId='session-001',
    actorId='user123',
    eventType='CONVERSATION',
    payload={
        'role': 'USER',
        'content': 'Hello, I need help with my order'
    }
)

event_id = response['eventId']
```

### get_event

Retrieve a specific event.

```python
response = client.get_event(
    memoryId='mem-abc123',
    eventId='evt-123'
)

event = response['event']
```

### delete_event

Delete an event.

```python
client.delete_event(
    memoryId='mem-abc123',
    eventId='evt-123'
)
```

### list_events

List events with filtering.

```python
response = client.list_events(
    memoryId='mem-abc123',
    sessionId='session-001',
    actorId='user123'
)

for event in response['events']:
    print(f"{event['eventType']}: {event['payload']}")
```

## Session Operations

### list_sessions

List all sessions.

```python
response = client.list_sessions(
    memoryId='mem-abc123'
)

for session in response['sessions']:
    print(f"Session: {session['sessionId']}, Actor: {session['actorId']}")
```

### stop_runtime_session

Stop a running session.

```python
client.stop_runtime_session(
    runtimeId='rt-abc123',
    sessionId='session-001'
)
```

## Browser Session Operations

### start_browser_session

Start a cloud browser session.

```python
response = client.start_browser_session(
    browserId='browser-123',
    sessionConfig={
        'headless': True,
        'viewport': {'width': 1920, 'height': 1080}
    }
)

session_id = response['sessionId']
```

### get_browser_session

Get browser session status.

```python
response = client.get_browser_session(
    browserId='browser-123',
    sessionId='session-001'
)
```

### stop_browser_session

Stop a browser session.

```python
client.stop_browser_session(
    browserId='browser-123',
    sessionId='session-001'
)
```

### list_browser_sessions

List browser sessions.

```python
response = client.list_browser_sessions(
    browserId='browser-123'
)
```

### update_browser_stream

Update browser stream settings.

```python
client.update_browser_stream(
    browserId='browser-123',
    sessionId='session-001',
    streamConfig={
        'quality': 'high'
    }
)
```

## Code Interpreter Operations

### start_code_interpreter_session

Start a code execution session.

```python
response = client.start_code_interpreter_session(
    interpreterId='interp-123'
)

session_id = response['sessionId']
```

### invoke_code_interpreter

Execute code.

```python
response = client.invoke_code_interpreter(
    interpreterId='interp-123',
    sessionId='session-001',
    code='print("Hello, World!")',
    language='python'
)

output = response['output']
```

### get_code_interpreter_session

Get session status.

```python
response = client.get_code_interpreter_session(
    interpreterId='interp-123',
    sessionId='session-001'
)
```

### stop_code_interpreter_session

Stop session.

```python
client.stop_code_interpreter_session(
    interpreterId='interp-123',
    sessionId='session-001'
)
```

### list_code_interpreter_sessions

List sessions.

```python
response = client.list_code_interpreter_sessions(
    interpreterId='interp-123'
)
```

## Agent Runtime Operations

### invoke_agent_runtime

Invoke an agent.

```python
response = client.invoke_agent_runtime(
    runtimeId='rt-abc123',
    endpointId='ep-123',
    payload={
        'message': 'Hello, I need assistance'
    },
    sessionId='session-001'
)

result = response['result']
```

### get_agent_card

Get agent metadata.

```python
response = client.get_agent_card(
    runtimeId='rt-abc123'
)

agent_info = response['agentCard']
```

### list_actors

List actors/users.

```python
response = client.list_actors(
    memoryId='mem-abc123'
)

for actor in response['actors']:
    print(f"Actor: {actor['actorId']}")
```

## Authentication Operations

### get_workload_access_token

Get access token for workload.

```python
response = client.get_workload_access_token(
    workloadIdentityId='wid-123'
)

token = response['accessToken']
```

### get_resource_api_key

Get API key for resource.

```python
response = client.get_resource_api_key(
    resourceId='res-123'
)

api_key = response['apiKey']
```

### get_resource_oauth2_token

Get OAuth2 token.

```python
response = client.get_resource_oauth2_token(
    resourceId='res-123',
    clientId='client-123',
    clientSecret='secret'
)

token = response['accessToken']
```

## Evaluation Operations

### evaluate

Run evaluation.

```python
response = client.evaluate(
    evaluatorId='eval-123',
    input={
        'query': 'test query',
        'response': 'test response'
    }
)

score = response['score']
```

## Pagination

Six operations support pagination:

```python
# Using paginator
paginator = client.get_paginator('list_memory_records')

for page in paginator.paginate(memoryId='mem-abc123'):
    for record in page['records']:
        print(record['content'])
```

Paginated operations:
- `list_actors`
- `list_events`
- `list_memory_extraction_jobs`
- `list_memory_records`
- `list_sessions`
- `retrieve_memory_records`

## Error Handling

```python
from botocore.exceptions import ClientError

try:
    response = client.get_memory_record(
        memoryId='mem-abc123',
        recordId='rec-123'
    )
except ClientError as e:
    error_code = e.response['Error']['Code']
    error_message = e.response['Error']['Message']

    if error_code == 'ResourceNotFoundException':
        print("Record not found")
    elif error_code == 'AccessDeniedException':
        print("Access denied")
    elif error_code == 'ThrottlingException':
        print("Rate limited, retry later")
    else:
        raise
```

## Common Error Codes

| Code | Description |
|------|-------------|
| `ResourceNotFoundException` | Resource doesn't exist |
| `AccessDeniedException` | Insufficient permissions |
| `ValidationException` | Invalid input parameters |
| `ThrottlingException` | Rate limit exceeded |
| `ServiceUnavailableException` | Service temporarily unavailable |
| `InternalServerException` | Internal error |
