# MemoryManager Class Reference

## Overview

The `MemoryManager` provides a high-level client for managing the lifecycle of AgentCore Memory resources, handling control plane CRUD operations.

## Installation

```bash
pip install bedrock-agentcore-starter-toolkit
```

## Import

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import (
    SemanticStrategy,
    UserPreferenceStrategy,
    SummaryStrategy,
    BaseStrategy
)
```

## Class Definition

```python
class MemoryManager:
    """High-level client for managing AgentCore Memory resources."""

    def __init__(
        self,
        region_name: str = "us-west-2",
        boto_session: Optional[boto3.Session] = None,
        config: Optional[BotocoreConfig] = None
    ):
        """
        Initialize MemoryManager.

        Args:
            region_name: AWS region for memory operations
            boto_session: Optional custom boto3 session
            config: Optional botocore configuration
        """
```

## Core Methods

### Memory Lifecycle

#### get_or_create_memory

```python
def get_or_create_memory(
    self,
    name: str,
    description: Optional[str] = None,
    strategies: List[BaseStrategy] = None
) -> Dict[str, Any]:
    """
    Get existing memory by name or create new one.

    Args:
        name: Unique memory name
        description: Optional description
        strategies: List of memory strategies

    Returns:
        Memory resource dictionary with 'id', 'name', 'status', etc.
    """
```

**Example:**
```python
memory = memory_manager.get_or_create_memory(
    name="CustomerSupport",
    description="Customer support memory",
    strategies=[
        SemanticStrategy(
            name="facts",
            namespaces=['/facts/{actorId}']
        )
    ]
)
print(f"Memory ID: {memory.get('id')}")
```

#### create_memory_and_wait

```python
def create_memory_and_wait(
    self,
    name: str,
    description: Optional[str] = None,
    strategies: List[BaseStrategy] = None,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Create memory and wait for ACTIVE status.

    Args:
        name: Unique memory name
        description: Optional description
        strategies: List of memory strategies
        timeout: Max wait time in seconds

    Returns:
        Memory resource dictionary

    Raises:
        TimeoutError: If memory doesn't become active
    """
```

#### get_memory

```python
def get_memory(self, memory_id: str) -> Dict[str, Any]:
    """
    Retrieve memory resource by ID.

    Args:
        memory_id: Memory resource identifier

    Returns:
        Memory resource dictionary
    """
```

#### delete_memory

```python
def delete_memory(self, memory_id: str) -> None:
    """
    Delete memory resource.

    Args:
        memory_id: Memory resource identifier
    """
```

#### delete_memory_and_wait

```python
def delete_memory_and_wait(
    self,
    memory_id: str,
    timeout: int = 300
) -> None:
    """
    Delete memory and wait for completion.

    Args:
        memory_id: Memory resource identifier
        timeout: Max wait time in seconds
    """
```

#### list_memories

```python
def list_memories(self) -> List[Dict[str, Any]]:
    """
    List all memory resources.

    Returns:
        List of memory resource dictionaries
    """
```

### Strategy Management

#### add_semantic_strategy

```python
def add_semantic_strategy(
    self,
    memory_id: str,
    name: str,
    namespaces: List[str],
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add semantic extraction strategy.

    Args:
        memory_id: Memory resource ID
        name: Strategy name
        namespaces: List of namespace patterns
        description: Optional description

    Returns:
        Updated memory resource
    """
```

#### add_user_preference_strategy

```python
def add_user_preference_strategy(
    self,
    memory_id: str,
    name: str,
    namespaces: List[str],
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add user preference extraction strategy.

    Args:
        memory_id: Memory resource ID
        name: Strategy name
        namespaces: List of namespace patterns
        description: Optional description

    Returns:
        Updated memory resource
    """
```

#### add_summary_strategy

```python
def add_summary_strategy(
    self,
    memory_id: str,
    name: str,
    namespaces: List[str],
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add session summary strategy.

    Args:
        memory_id: Memory resource ID
        name: Strategy name
        namespaces: List of namespace patterns
        description: Optional description

    Returns:
        Updated memory resource
    """
```

#### update_memory_strategies

```python
def update_memory_strategies(
    self,
    memory_id: str,
    add_strategies: List[BaseStrategy] = None,
    remove_strategies: List[str] = None
) -> Dict[str, Any]:
    """
    Update memory strategies.

    Args:
        memory_id: Memory resource ID
        add_strategies: Strategies to add
        remove_strategies: Strategy names to remove

    Returns:
        Updated memory resource
    """
```

#### get_memory_strategies

```python
def get_memory_strategies(self, memoryId: str) -> List[Dict[str, Any]]:
    """
    List strategies for a memory.

    Args:
        memoryId: Memory resource ID

    Returns:
        List of strategy dictionaries
    """
```

#### modify_strategy

```python
def modify_strategy(
    self,
    memory_id: str,
    strategy_name: str,
    new_description: Optional[str] = None,
    new_namespaces: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Modify existing strategy.

    Args:
        memory_id: Memory resource ID
        strategy_name: Name of strategy to modify
        new_description: Updated description
        new_namespaces: Updated namespace patterns

    Returns:
        Updated memory resource
    """
```

#### delete_strategy

```python
def delete_strategy(
    self,
    memory_id: str,
    strategy_name: str
) -> None:
    """
    Delete strategy from memory.

    Args:
        memory_id: Memory resource ID
        strategy_name: Name of strategy to delete
    """
```

### Status & Monitoring

#### get_memory_status

```python
def get_memory_status(self, memory_id: str) -> str:
    """
    Get current memory status.

    Args:
        memory_id: Memory resource ID

    Returns:
        Status string: CREATING, ACTIVE, UPDATING, DELETING, FAILED
    """
```

### Observability

#### enable_observability

```python
def enable_observability(
    self,
    memory_id: str,
    cloudwatch_log_group: str
) -> None:
    """
    Enable CloudWatch logging for memory.

    Args:
        memory_id: Memory resource ID
        cloudwatch_log_group: Log group name
    """
```

#### disable_observability

```python
def disable_observability(self, memory_id: str) -> None:
    """
    Disable CloudWatch logging.

    Args:
        memory_id: Memory resource ID
    """
```

## Strategy Models

### SemanticStrategy

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SemanticStrategy

strategy = SemanticStrategy(
    name="factExtractor",
    namespaces=['/facts/{actorId}'],
    description="Extract facts from conversations"
)
```

### UserPreferenceStrategy

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import UserPreferenceStrategy

strategy = UserPreferenceStrategy(
    name="preferences",
    namespaces=['/prefs/{actorId}'],
    description="Capture user preferences"
)
```

### SummaryStrategy

```python
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import SummaryStrategy

strategy = SummaryStrategy(
    name="sessionSummary",
    namespaces=['/summaries/{actorId}/{sessionId}'],
    description="Summarize sessions"
)
```

## Complete Example

```python
from bedrock_agentcore_starter_toolkit.operations.memory.manager import MemoryManager
from bedrock_agentcore_starter_toolkit.operations.memory.models.strategies import (
    SemanticStrategy,
    UserPreferenceStrategy
)

# Initialize manager
memory_manager = MemoryManager(region_name="us-west-2")

# Create memory with strategies
memory = memory_manager.get_or_create_memory(
    name="MyAgentMemory",
    description="Agent persistent memory",
    strategies=[
        SemanticStrategy(
            name="facts",
            namespaces=['/facts/{actorId}']
        ),
        UserPreferenceStrategy(
            name="preferences",
            namespaces=['/prefs/{actorId}']
        )
    ]
)

memory_id = memory.get('id')
print(f"Created memory: {memory_id}")

# Check status
status = memory_manager.get_memory_status(memory_id)
print(f"Status: {status}")

# List strategies
strategies = memory_manager.get_memory_strategies(memoryId=memory_id)
for s in strategies:
    print(f"  Strategy: {s['name']} - {s['status']}")

# Add another strategy later
memory_manager.add_summary_strategy(
    memory_id=memory_id,
    name="summaries",
    namespaces=['/summaries/{actorId}/{sessionId}']
)

# List all memories
all_memories = memory_manager.list_memories()
print(f"Total memories: {len(all_memories)}")

# Cleanup
# memory_manager.delete_memory_and_wait(memory_id=memory_id)
```

## Error Handling

```python
from botocore.exceptions import ClientError

try:
    memory = memory_manager.get_or_create_memory(
        name="TestMemory",
        strategies=[]
    )
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'ResourceNotFoundException':
        print("Memory not found")
    elif error_code == 'ConflictException':
        print("Memory already exists with different config")
    else:
        raise
```

## Constants

```python
from bedrock_agentcore_starter_toolkit.operations.memory.constants import (
    MemoryStatus,
    MemoryStrategyStatus,
    StrategyType
)

# Memory status values
MemoryStatus.CREATING
MemoryStatus.ACTIVE
MemoryStatus.UPDATING
MemoryStatus.DELETING
MemoryStatus.FAILED

# Strategy status values
MemoryStrategyStatus.CREATING
MemoryStrategyStatus.ACTIVE
MemoryStrategyStatus.UPDATING
MemoryStrategyStatus.DELETING
MemoryStrategyStatus.FAILED

# Strategy types
StrategyType.SEMANTIC
StrategyType.USER_PREFERENCE
StrategyType.SUMMARY
```
