---
title: Amazon Bedrock Prompt Engineering
category: aws-bedrock
tags:
  - aws
  - bedrock
  - prompt-engineering
  - prompts
  - few-shot
  - zero-shot
  - best-practices
---

# Prompt Engineering for Amazon Bedrock

## Overview

**Prompt Engineering** refers to the practice of optimizing textual input to a Large Language Model (LLM) to obtain desired responses. It helps LLMs perform a wide variety of tasks including:

- Classification
- Question answering
- Code generation
- Creative writing
- Summarization

The quality of prompts directly impacts the quality of model responses.

---

## What is a Prompt?

Prompts are specific sets of inputs provided by users that guide LLMs on Amazon Bedrock to generate appropriate responses for given tasks or instructions.

**Example:**
```
User Prompt: Who invented the airplane?
Output: The Wright brothers, Orville and Wilbur Wright are widely credited with inventing
and manufacturing the world's first successful airplane.
```

---

## Components of a Prompt

A single prompt includes several components:

1. **Task/Instruction** - What you want the LLM to perform
2. **Context** - Description of the relevant domain
3. **Demonstration Examples** - Few-shot examples (optional)
4. **Input Text** - The content the LLM operates on

### Example with Components

```
User Prompt: The following is text from a restaurant review: "[review text]"
Summarize the above restaurant review in one sentence.

Output: Alessandro's Brilliant Pizza is a fantastic restaurant in Seattle with a beautiful
view over Puget Sound, decadent and delicious food, and excellent service.
```

---

## Zero-Shot vs Few-Shot Prompting

### Zero-Shot Prompting

No example input-output pairs provided:

```
User Prompt: Tell me the sentiment of the following headline and categorize it as either
positive, negative or neutral: New airline between Seattle and San Francisco offers a great
opportunity for both passengers and investors.

Output: Positive
```

### Few-Shot Prompting

Includes demonstration examples to calibrate output:

```
User Prompt: Tell me the sentiment of the following headline and categorize it as either
positive, negative or neutral. Here are some examples:

Research firm fends off allegations of impropriety over new technology. Answer: Negative
Offshore windfarms continue to thrive as vocal minority in opposition dwindles. Answer: Positive
Manufacturing plant is the latest target in investigation by state officials. Answer:

Output: Negative
```

---

## Few-Shot with Anthropic Claude Models

Use `<example>` tags and `H:` / `A:` delimiters:

```
User Prompt: Human: Please classify the given email as "Personal" or "Commercial" related emails.
Here are some examples.

<example>
H: Hi Tom, it's been long time since we met last time. We plan to have a party at my house
this weekend. Will you be able to come over?
A: Personal
</example>

<example>
H: Hi Tom, we have a special offer for you. For a limited time, our customers can save up to
35% of their total expense when you make reservations within two days. Book now and save money!
A: Commercial
</example>

H: Hi Tom, Have you heard that we have launched all-new set of products. Order now, you will
save $100 for the new products. Please check our website.

Output: Commercial
```

---

## Prompt Engineering Best Practices

### 1. Be Clear and Specific

**Bad:**
```
Tell me about dogs.
```

**Good:**
```
List the top 5 most popular dog breeds in the United States, including their typical size
and temperament characteristics.
```

### 2. Provide Context

**Without context:**
```
What's the best approach?
```

**With context:**
```
I'm building a web application that needs to handle 10,000 concurrent users. The application
uses a PostgreSQL database and is deployed on AWS. What's the best approach for implementing
caching to reduce database load?
```

### 3. Use Clear Output Format Instructions

```
Analyze the following customer feedback and provide:
1. Overall sentiment (Positive/Negative/Neutral)
2. Key themes mentioned (list up to 3)
3. Actionable recommendations (list up to 2)

Format your response as JSON:
{
  "sentiment": "",
  "themes": [],
  "recommendations": []
}
```

### 4. Use System Prompts for Persona

```python
system = [
    {"text": """You are a senior software architect with 20 years of experience.
    When answering questions:
    - Consider scalability, maintainability, and security
    - Provide practical, production-ready solutions
    - Include trade-offs for different approaches
    - Use industry best practices"""}
]
```

### 5. Chain-of-Thought Prompting

Encourage step-by-step reasoning:

```
Solve the following problem step by step, showing your work:

A store sells apples for $2 each and oranges for $3 each. If a customer buys 5 apples
and 3 oranges, and pays with a $20 bill, how much change do they receive?

Think through this step by step before providing the final answer.
```

---

## XML Tags for Structure (Claude Models)

### Using Tags for Clear Structure

```
<task>
Analyze the following code for potential security vulnerabilities.
</task>

<code>
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    return result
</code>

<requirements>
- Identify all security issues
- Explain why each is a problem
- Provide fixed code
</requirements>
```

### Using Tags for Examples

```
<examples>
<example>
Input: The product arrived damaged
Output: Negative - Product Quality
</example>

<example>
Input: Fast shipping, exactly as described!
Output: Positive - Delivery & Accuracy
</example>
</examples>

Now classify this review:
Input: The color was different from the pictures but shipping was fast
```

---

## Role-Based Prompts

### Expert Role

```
You are an expert data scientist specializing in machine learning. A junior developer
asks you the following question. Provide a detailed but accessible explanation.

Question: What is the difference between supervised and unsupervised learning?
```

### Teaching Role

```
You are a patient teacher explaining concepts to someone new to programming.
Use simple analogies and avoid jargon.

Explain what an API is and why it's useful.
```

### Reviewer Role

```
You are a code reviewer conducting a thorough review. For the following code:
1. Identify bugs or issues
2. Suggest improvements
3. Note any best practices violations
4. Rate the code quality (1-10)

[code to review]
```

---

## Output Format Control

### JSON Output

```
Extract the following information from this text and return as JSON:
- Names of people mentioned
- Dates mentioned
- Locations mentioned

Text: "John Smith met with Sarah Johnson in New York on January 15, 2024 to discuss
the upcoming conference in London scheduled for March."

Return format:
{
  "people": [],
  "dates": [],
  "locations": []
}
```

### Markdown Output

```
Generate a technical documentation page for the following API endpoint.
Use proper markdown formatting with:
- Headers (##, ###)
- Code blocks for examples
- Tables for parameters
- Bullet points for lists

Endpoint: POST /api/users
Description: Creates a new user account
```

### Structured Lists

```
Provide a comparison of React, Vue, and Angular for building web applications.

Format your response as a comparison table with the following categories:
| Feature | React | Vue | Angular |
|---------|-------|-----|---------|
| Learning Curve | | | |
| Performance | | | |
| Community Support | | | |
| Best Use Case | | | |
```

---

## Prompt Templates for Common Tasks

### Code Generation

```
Write a [language] function that [description].

Requirements:
- [requirement 1]
- [requirement 2]

Include:
- Type hints/annotations
- Error handling
- Docstring with examples
- Unit tests
```

### Code Review

```
Review the following [language] code:

<code>
[paste code here]
</code>

Provide:
1. Security issues (if any)
2. Performance concerns
3. Code style improvements
4. Suggested refactoring
```

### Summarization

```
Summarize the following text in [X] sentences/bullet points.

Focus on:
- Key findings/conclusions
- Important numbers/statistics
- Action items (if any)

Text:
[paste text here]
```

### Q&A with Context

```
Answer the question based ONLY on the provided context. If the answer cannot be
determined from the context, say "I cannot answer this based on the provided information."

Context:
[paste context here]

Question: [question here]
```

---

## Common Pitfalls to Avoid

1. **Vague Instructions** - Be specific about what you want
2. **Missing Context** - Provide relevant background information
3. **Ambiguous Output Format** - Specify exactly how you want the response
4. **Too Many Tasks** - Break complex requests into smaller prompts
5. **Conflicting Instructions** - Ensure all instructions are consistent
6. **No Examples** - Use few-shot prompting for complex tasks

---

## Additional Resources

- [Anthropic Claude Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Amazon Bedrock Prompt Guidelines](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-engineering-guidelines.html)
- [XML Tags in Claude](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags)