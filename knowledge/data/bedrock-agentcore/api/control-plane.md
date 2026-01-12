# BedrockAgentCoreControl Control Plane API Reference

## Overview

The Amazon Bedrock AgentCore Control Plane API manages configuration, creation, modification, and monitoring of AgentCore resources.

## boto3 Client

```python
import boto3

client = boto3.client('bedrock-agentcore-control', region_name='us-west-2')
```

## Memory Management

### create_memory

Create a new memory resource.

```python
response = client.create_memory(
    name='CustomerSupportMemory',
    description='Memory for customer support agent',
    strategies=[
        {
            'semanticMemoryStrategy': {
                'name': 'factExtraction',
                'namespaces': ['/facts/{actorId}']
            }
        },
        {
            'userPreferenceMemoryStrategy': {
                'name': 'preferences',
                'namespaces': ['/prefs/{actorId}']
            }
        }
    ],
    tags={
        'Environment': 'production'
    }
)

memory_id = response['memoryId']
print(f"Created memory: {memory_id}")
```

### get_memory

Get memory details.

```python
response = client.get_memory(
    memoryId='mem-abc123'
)

memory = response['memory']
print(f"Name: {memory['name']}")
print(f"Status: {memory['status']}")
print(f"Strategies: {memory['strategies']}")
```

### update_memory

Update memory configuration.

```python
response = client.update_memory(
    memoryId='mem-abc123',
    description='Updated description',
    addStrategies=[
        {
            'summaryMemoryStrategy': {
                'name': 'sessionSummary',
                'namespaces': ['/summaries/{actorId}/{sessionId}']
            }
        }
    ]
)
```

### delete_memory

Delete a memory resource.

```python
client.delete_memory(
    memoryId='mem-abc123'
)
```

### list_memories

List all memories.

```python
response = client.list_memories(
    maxResults=50
)

for memory in response['memories']:
    print(f"{memory['name']}: {memory['memoryId']} ({memory['status']})")

# Handle pagination
while 'nextToken' in response:
    response = client.list_memories(
        nextToken=response['nextToken']
    )
    for memory in response['memories']:
        print(f"{memory['name']}: {memory['memoryId']}")
```

## Agent Runtime Management

### create_agent_runtime

Create a runtime environment.

```python
response = client.create_agent_runtime(
    name='CustomerSupportRuntime',
    description='Runtime for customer support agent',
    agentRuntimeConfig={
        'entrypoint': 'agent.py',
        'memoryId': 'mem-abc123',
        'environmentVariables': {
            'LOG_LEVEL': 'INFO'
        }
    },
    tags={
        'Environment': 'production'
    }
)

runtime_id = response['agentRuntimeId']
```

### get_agent_runtime

Get runtime details.

```python
response = client.get_agent_runtime(
    agentRuntimeId='rt-abc123'
)

runtime = response['agentRuntime']
print(f"Status: {runtime['status']}")
```

### update_agent_runtime

Update runtime configuration.

```python
response = client.update_agent_runtime(
    agentRuntimeId='rt-abc123',
    agentRuntimeConfig={
        'environmentVariables': {
            'LOG_LEVEL': 'DEBUG'
        }
    }
)
```

### delete_agent_runtime

Delete runtime.

```python
client.delete_agent_runtime(
    agentRuntimeId='rt-abc123'
)
```

### list_agent_runtimes

List runtimes.

```python
response = client.list_agent_runtimes()

for runtime in response['agentRuntimes']:
    print(f"{runtime['name']}: {runtime['status']}")
```

### list_agent_runtime_versions

List runtime versions.

```python
response = client.list_agent_runtime_versions(
    agentRuntimeId='rt-abc123'
)

for version in response['versions']:
    print(f"Version: {version['version']}")
```

## Runtime Endpoint Management

### create_agent_runtime_endpoint

Create an endpoint.

```python
response = client.create_agent_runtime_endpoint(
    agentRuntimeId='rt-abc123',
    name='ProductionEndpoint',
    endpointConfig={
        'concurrency': 10
    }
)

endpoint_id = response['endpointId']
```

### update_agent_runtime_endpoint

Update endpoint.

```python
client.update_agent_runtime_endpoint(
    agentRuntimeId='rt-abc123',
    endpointId='ep-123',
    endpointConfig={
        'concurrency': 20
    }
)
```

### delete_agent_runtime_endpoint

Delete endpoint.

```python
client.delete_agent_runtime_endpoint(
    agentRuntimeId='rt-abc123',
    endpointId='ep-123'
)
```

### list_agent_runtime_endpoints

List endpoints.

```python
response = client.list_agent_runtime_endpoints(
    agentRuntimeId='rt-abc123'
)
```

## Gateway Management

### create_gateway

Create an MCP gateway.

```python
response = client.create_gateway(
    name='ToolGateway',
    description='Gateway for agent tools',
    gatewayConfig={
        'authorizerId': 'auth-123'
    }
)

gateway_id = response['gatewayId']
```

### get_gateway

Get gateway details.

```python
response = client.get_gateway(
    gatewayId='gw-abc123'
)

gateway = response['gateway']
print(f"URL: {gateway['url']}")
```

### update_gateway

Update gateway.

```python
client.update_gateway(
    gatewayId='gw-abc123',
    description='Updated description'
)
```

### delete_gateway

Delete gateway.

```python
client.delete_gateway(
    gatewayId='gw-abc123'
)
```

### list_gateways

List gateways.

```python
response = client.list_gateways()

for gateway in response['gateways']:
    print(f"{gateway['name']}: {gateway['gatewayId']}")
```

## Gateway Target Management

### create_gateway_target

Add a tool target to gateway.

```python
response = client.create_gateway_target(
    gatewayId='gw-abc123',
    name='Calculator',
    targetConfig={
        'toolSchema': {
            'name': 'calculator',
            'description': 'Math operations',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'operation': {'type': 'string'},
                    'a': {'type': 'number'},
                    'b': {'type': 'number'}
                }
            }
        }
    }
)
```

### update_gateway_target

Update target.

```python
client.update_gateway_target(
    gatewayId='gw-abc123',
    targetId='target-123',
    targetConfig={...}
)
```

### delete_gateway_target

Delete target.

```python
client.delete_gateway_target(
    gatewayId='gw-abc123',
    targetId='target-123'
)
```

### list_gateway_targets

List targets.

```python
response = client.list_gateway_targets(
    gatewayId='gw-abc123'
)
```

### synchronize_gateway_targets

Sync all targets.

```python
client.synchronize_gateway_targets(
    gatewayId='gw-abc123'
)
```

## Credential Provider Management

### API Key Providers

```python
# Create
response = client.create_api_key_credential_provider(
    name='MyAPIKeyProvider',
    credentialProviderConfig={...}
)

# Get
response = client.get_api_key_credential_provider(
    credentialProviderId='cp-123'
)

# Update
client.update_api_key_credential_provider(
    credentialProviderId='cp-123',
    credentialProviderConfig={...}
)

# Delete
client.delete_api_key_credential_provider(
    credentialProviderId='cp-123'
)

# List
response = client.list_api_key_credential_providers()
```

### OAuth2 Providers

```python
# Create
response = client.create_oauth2_credential_provider(
    name='MyOAuth2Provider',
    credentialProviderConfig={
        'clientId': 'client-123',
        'clientSecret': 'secret',
        'tokenEndpoint': 'https://...'
    }
)

# Get, Update, Delete, List - similar pattern
```

## Code Interpreter Management

```python
# Create
response = client.create_code_interpreter(
    name='PythonInterpreter',
    interpreterConfig={
        'language': 'python',
        'packages': ['numpy', 'pandas']
    }
)

# Get
response = client.get_code_interpreter(
    codeInterpreterId='interp-123'
)

# Delete
client.delete_code_interpreter(
    codeInterpreterId='interp-123'
)

# List
response = client.list_code_interpreters()
```

## Browser Management

```python
# Create
response = client.create_browser(
    name='ChromeBrowser',
    browserConfig={
        'headless': True
    }
)

# Get
response = client.get_browser(
    browserId='browser-123'
)

# Delete
client.delete_browser(
    browserId='browser-123'
)

# List
response = client.list_browsers()
```

## Policy Management

### create_policy

Create a Cedar policy.

```python
response = client.create_policy(
    name='AgentAccessPolicy',
    policyDocument='''
    permit (
        principal,
        action == Action::"invoke",
        resource
    );
    '''
)

policy_id = response['policyId']
```

### get_policy

Get policy.

```python
response = client.get_policy(
    policyId='policy-123'
)
```

### update_policy

Update policy.

```python
client.update_policy(
    policyId='policy-123',
    policyDocument='...'
)
```

### delete_policy

Delete policy.

```python
client.delete_policy(
    policyId='policy-123'
)
```

## Policy Engine Management

```python
# Create
response = client.create_policy_engine(
    name='MainPolicyEngine',
    policyIds=['policy-123', 'policy-456']
)

# Get
response = client.get_policy_engine(
    policyEngineId='engine-123'
)

# Update
client.update_policy_engine(
    policyEngineId='engine-123',
    policyIds=[...]
)

# Delete
client.delete_policy_engine(
    policyEngineId='engine-123'
)
```

## Workload Identity Management

```python
# Create
response = client.create_workload_identity(
    name='AgentIdentity',
    identityConfig={...}
)

# Get
response = client.get_workload_identity(
    workloadIdentityId='wid-123'
)

# Update
client.update_workload_identity(
    workloadIdentityId='wid-123',
    identityConfig={...}
)

# Delete
client.delete_workload_identity(
    workloadIdentityId='wid-123'
)

# List
response = client.list_workload_identities()
```

## Resource Policy Management

```python
# Put (create/update)
client.put_resource_policy(
    resourceArn='arn:aws:bedrock-agentcore:...',
    policy='...'
)

# Get
response = client.get_resource_policy(
    resourceArn='arn:aws:bedrock-agentcore:...'
)

# Delete
client.delete_resource_policy(
    resourceArn='arn:aws:bedrock-agentcore:...'
)
```

## Waiters

Six waiters for polling status:

```python
# Wait for memory to be created
waiter = client.get_waiter('memory_created')
waiter.wait(
    memoryId='mem-abc123',
    WaiterConfig={
        'Delay': 10,
        'MaxAttempts': 30
    }
)

# Available waiters:
# - memory_created
# - policy_active
# - policy_deleted
# - policy_engine_active
# - policy_engine_deleted
# - policy_generation_completed
```

## Paginators

15 operations support pagination:

```python
paginator = client.get_paginator('list_memories')

for page in paginator.paginate():
    for memory in page['memories']:
        print(memory['name'])
```

## Error Handling

```python
from botocore.exceptions import ClientError

try:
    response = client.create_memory(
        name='TestMemory',
        strategies=[]
    )
except ClientError as e:
    error_code = e.response['Error']['Code']

    if error_code == 'ConflictException':
        print("Memory with this name already exists")
    elif error_code == 'ValidationException':
        print("Invalid parameters")
    elif error_code == 'ServiceQuotaExceededException':
        print("Quota exceeded")
    else:
        raise
```

## Common Error Codes

| Code | Description |
|------|-------------|
| `ConflictException` | Resource already exists |
| `ResourceNotFoundException` | Resource not found |
| `ValidationException` | Invalid parameters |
| `AccessDeniedException` | Insufficient permissions |
| `ServiceQuotaExceededException` | Quota limit reached |
| `ThrottlingException` | Rate limited |
| `InternalServerException` | Internal error |
