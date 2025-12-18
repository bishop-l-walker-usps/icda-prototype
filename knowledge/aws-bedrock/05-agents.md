---
title: Amazon Bedrock Agents
category: aws-bedrock
tags:
  - aws
  - bedrock
  - agents
  - action-groups
  - orchestration
  - automation
---

# Amazon Bedrock Agents

## Overview

**Amazon Bedrock Agents** enables building and configuring autonomous agents in your application. An agent helps end-users complete actions based on organization data and user input.

### Key Capabilities

- Orchestrate interactions between foundation models (FMs), data sources, software applications, and user conversations
- Automatically call APIs to take actions
- Invoke knowledge bases to supplement information
- No capacity provisioning, infrastructure management, or custom code required
- Amazon Bedrock manages: prompt engineering, memory, monitoring, encryption, user permissions, and API invocation

---

## Agent Core Functions

Agents perform the following tasks:

1. **Understand & Break Down Requests** - Extend foundation models to understand user requests and decompose tasks into smaller steps

2. **Collect Information** - Gather additional information from users through natural conversation

3. **Take Actions** - Fulfill customer requests by making API calls to company systems

4. **Augment Performance** - Query data sources to improve accuracy and performance

---

## Implementation Workflow

### Step 1: Create Knowledge Base (Optional)

Store private data in a knowledge base for agent augmentation.

### Step 2: Configure Agent Components

#### Action Groups (Required - At Least One)
- Define actions the agent can perform
- Examples: CreateBooking, GetBooking, CancelBooking

#### Knowledge Base Integration (Optional)
- Associate knowledge bases with the agent
- Augment agent performance and response accuracy

### Step 3: Customize Agent Behavior (Optional)

Modify prompt templates for:
- Pre-processing
- Orchestration
- Knowledge base response generation
- Post-processing

### Step 4: Test Agent

- Test in Amazon Bedrock console or via API calls to `TSTALIASID`
- Use traces to examine reasoning process at each orchestration step

### Step 5: Deploy Agent

Create an alias pointing to an agent version for production deployment.

### Step 6: Integrate into Application

Set up application to make API calls to agent alias.

### Step 7: Iterate

Create additional versions and aliases as needed.

---

## Action Groups

An action group defines actions that an agent can help users perform. Action groups organize related actions.

### Example: Hotel Booking Action Group

A `BookHotel` action group might include:

- **CreateBooking** - Helps users book a hotel
- **GetBooking** - Helps users get information about a booked hotel
- **CancelBooking** - Helps users cancel a booking

### Creating an Action Group

1. **Define Parameters** - Define the parameters and information that the agent must elicit from the user for each action
2. **Handle Fulfillment** - Decide how the agent handles the parameters and where it sends the information

### Action Group Components

| Component | Description |
|-----------|-------------|
| **Action Name** | Unique identifier for the action |
| **Action Description** | What the action does |
| **Parameters** | Input fields with types and descriptions |
| **Fulfillment** | Lambda function or API call |

---

## Use Cases

### Insurance Claims Processing
- Agent assists customers through claims process
- Collects required information
- Submits claims to backend systems

### Travel Reservations
- Agent helps customers make travel bookings
- Searches availability
- Processes reservations

### Customer Support
- Agent answers questions from knowledge base
- Escalates complex issues
- Performs account actions

---

## Agent Architecture

```
User Query
    |
    v
[Pre-processing] --> Parse and classify request
    |
    v
[Orchestration] --> Determine action sequence
    |
    v
[Action Execution]
    |-- [API Call] --> External systems
    |-- [Knowledge Base Query] --> RAG retrieval
    |-- [Tool Use] --> Custom functions
    |
    v
[Post-processing] --> Format response
    |
    v
Response to User
```

---

## Best Practices

### Agent Design

1. **Clear Action Descriptions** - Write detailed descriptions for accurate action selection
2. **Minimal Required Parameters** - Only require essential information
3. **Graceful Error Handling** - Handle failures elegantly
4. **Comprehensive Knowledge Base** - Include relevant documentation

### Testing

1. **Use Test Alias** - Test with `TSTALIASID` before deployment
2. **Examine Traces** - Review reasoning at each step
3. **Edge Cases** - Test unusual inputs and error scenarios
4. **Multi-turn Conversations** - Test complex conversation flows

### Deployment

1. **Version Control** - Create versions for each release
2. **Aliases** - Use aliases for environment management
3. **Monitoring** - Enable logging and metrics
4. **Iteration** - Continuously improve based on feedback

---

## Integration with Knowledge Bases

Agents can query knowledge bases to:

- Retrieve relevant information for responses
- Augment answers with domain-specific content
- Provide citations from source documents
- Handle questions outside of action scope

### Configuration

1. Associate knowledge base with agent
2. Configure when to query (always, or based on intent)
3. Set up response augmentation

---

## Additional Resources

- [Action Groups Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-action-create.html)
- [Knowledge Base Integration](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-kb-add.html)
- [Advanced Prompt Templates](https://docs.aws.amazon.com/bedrock/latest/userguide/advanced-prompts.html)
- [Testing & Troubleshooting](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-test.html)