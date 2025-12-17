# ICDA Knowledge Base

Internal documentation for the ICDA (Intelligent Customer Data Access) system.
This folder is automatically indexed by the MCP Knowledge Server for RAG retrieval.

## Folder Structure

```
knowledge/
├── README.md                    # This file
├── address-standards/           # Address format standards and rules
│   ├── puerto-rico-urbanization-addressing.md
│   └── usps-address-format-standards.md (planned)
├── aws-bedrock/                 # AWS Bedrock documentation (AI/ML)
│   ├── README.md                # Overview and quick reference
│   ├── 01-overview-getting-started.md
│   ├── 02-supported-models.md
│   ├── 03-converse-api.md
│   ├── 04-tool-use-function-calling.md
│   ├── 05-agents.md
│   ├── 06-knowledge-bases-rag.md
│   ├── 07-guardrails.md
│   ├── 08-python-sdk-examples.md
│   └── 09-prompt-engineering.md
├── patterns/                    # Domain patterns and validation rules
│   ├── pr-urbanization-patterns.md (planned)
│   ├── verification-rules.md (planned)
│   └── error-handling-patterns.md (planned)
├── examples/                    # Query and address examples
│   ├── address-query-examples.md (planned)
│   ├── verification-examples.md (planned)
│   └── edge-cases.md (planned)
└── api/                         # API documentation
    └── icda-endpoints.md (planned)
```

## Document Categories

### address-standards
Official address format standards including:
- Puerto Rico urbanization rules (ZIP 006-009)
- USPS address formatting guidelines
- State-specific address quirks

### aws-bedrock
Amazon Bedrock AI/ML documentation including:
- Service overview and getting started
- Supported foundation models (Claude, Titan, Nova, Llama, Mistral)
- Converse API for multi-turn conversations
- Tool use / function calling
- Bedrock Agents and action groups
- Knowledge bases and RAG architecture
- Guardrails for content safety
- Python SDK examples
- Prompt engineering best practices

### patterns
Domain patterns for address verification:
- PR urbanization detection patterns
- Validation rule definitions
- Error handling strategies

### examples
Practical examples for testing and reference:
- NLP query examples
- Address verification input/output pairs
- Edge cases and complex scenarios

### api
API documentation for developers:
- REST endpoint reference
- Nova tool calling schemas
- Guardrails configuration

## Auto-Indexing

Documents in this folder are automatically indexed when:
1. The MCP Knowledge Server starts
2. The main ICDA application starts (via `main.py`)
3. A document is uploaded via MCP tools

### Supported Formats
- Markdown (.md) - Preferred
- Plain text (.txt)
- PDF (.pdf)
- Word documents (.docx)

### Document Best Practices

1. **Use descriptive filenames**: `puerto-rico-urbanization-addressing.md` not `pr-addr.md`
2. **Add frontmatter tags**: Help with filtering and categorization
3. **Include examples**: Concrete examples improve RAG retrieval
4. **Keep chunks reasonable**: Avoid very long paragraphs

### Frontmatter Example

```markdown
---
title: Puerto Rico Urbanization Addressing
category: address-standards
tags:
  - puerto-rico
  - urbanization
  - usps
  - zip-codes
---
```

## Usage with MCP Knowledge Server

```bash
# Search for PR urbanization rules
search_knowledge(query="how to detect puerto rico addresses", tags=["puerto-rico"])

# List all address standards
list_documents(category="address-standards")

# Upload new document
upload_document(file_path="/path/to/doc.md", category="patterns", tags=["validation"])
```

## Related Files

- `.mcp.json` - MCP server configuration
- `mcp-knowledge-server/` - Knowledge server implementation
- `icda/knowledge.py` - ICDA knowledge manager
- `main.py` - Auto-indexing on startup
