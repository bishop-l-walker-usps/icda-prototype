---
title: Amazon Bedrock Knowledge Bases (RAG)
category: aws-bedrock
tags:
  - aws
  - bedrock
  - knowledge-bases
  - rag
  - retrieval-augmented-generation
  - vector-store
  - embeddings
---

# Amazon Bedrock Knowledge Bases (RAG)

## Overview

Amazon Bedrock Knowledge Bases enable you to integrate proprietary information into generative-AI applications using **Retrieval Augmented Generation (RAG)** - a technique that uses information from data sources to improve the relevancy and accuracy of generated responses.

## Key Capabilities

Amazon Bedrock Knowledge Bases allow you to:

- **Answer queries** by returning relevant information from data sources
- **Generate accurate responses** using retrieved information to augment prompts
- **Include citations** in generated responses to reference original data sources
- **Handle multimodal content** including documents with visual resources
- **Search with images** using multimodal embedding models
- **Query structured databases** by converting natural language into SQL
- **Update data sources** and sync changes directly into the knowledge base
- **Use reranking models** to influence retrieved results
- **Integrate with Amazon Bedrock Agents** for workflow automation

---

## Data Source Support

Knowledge bases can connect to:

- **Unstructured data sources** (documents, text)
- **Structured data stores** (databases with SQL query transformation)
- **Multimodal content** (documents with images)
- **Amazon Kendra GenAI indexes**
- **Amazon Neptune Analytics graphs**

---

## Vector Store Options

### Quick Create (Amazon Bedrock Creates)

#### Amazon OpenSearch Serverless
- Automatically configured vector search collection and index
- Supports Confluence, Microsoft SharePoint, and Salesforce data sources

#### Amazon Aurora PostgreSQL Serverless
- Sets up PostgreSQL vector store
- Transforms text data into chunks and vectors

#### Amazon Neptune Analytics
- RAG with graphs for enhanced responses

#### Amazon S3 Vectors
- Creates S3 vector bucket and index
- Up to 1 KB metadata per vector
- Up to 35 metadata keys per vector

### Use Existing Vector Store

- Select a supported vector store
- Identify vector field names and metadata field names
- Configure mapping to existing index

---

## Creating a Knowledge Base

### Console Setup Steps

1. **Initial Configuration**
   - Sign in to AWS Console with IAM identity
   - Open Amazon Bedrock console
   - Choose Knowledge bases > Create

2. **Service Role Configuration**
   - Choose IAM role with required permissions
   - Let Amazon Bedrock create role OR use custom role

3. **Data Source Selection**
   - Connect to your data source
   - Configure sync settings

4. **Embeddings Model Selection**
   - Choose embeddings model for vectorization
   - For multimodal data, select multimodal embedding model:
     - Amazon Titan Multimodal Embeddings G1
     - Cohere Embed v3

5. **Vector Database Configuration**
   - Quick create OR use existing
   - Configure field mappings

6. **Review and Create**

---

## Embeddings Configuration

### Model Options

| Model | Use Case | Features |
|-------|----------|----------|
| Amazon Titan Text Embeddings V2 | Text-only | 1024 dimensions, binary/float |
| Amazon Titan Multimodal Embeddings G1 | Images | Optimized for image search |
| Cohere Embed v3 | Mixed content | Text and image datasets |

### Additional Configurations

| Configuration | Description |
|---------------|-------------|
| **Embeddings type** | float32 (precise, costly) or binary (less precise, less costly) |
| **Vector dimensions** | Higher values improve accuracy but increase cost and latency |

---

## API Creation

### CreateKnowledgeBase Request

```json
PUT /knowledgebases/ HTTP/1.1
Content-type: application/json

{
  "name": "MyKB",
  "description": "My knowledge base",
  "roleArn": "arn:aws:iam::111122223333:role/AmazonBedrockExecutionRoleForKnowledgeBase_123",
  "knowledgeBaseConfiguration": {
    "type": "VECTOR",
    "vectorKnowledgeBaseConfiguration": {
      "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
      "embeddingModelConfiguration": {
        "bedrockEmbeddingModelConfiguration": {
          "dimensions": 1024,
          "embeddingDataType": "BINARY"
        }
      }
    }
  },
  "storageConfiguration": {
    "opensearchServerlessConfiguration": {
      "collectionArn": "arn:aws:aoss:us-east-1:111122223333:collection/abcdefghij1234567890",
      "fieldMapping": {
        "metadataField": "metadata",
        "textField": "text",
        "vectorField": "vector"
      },
      "vectorIndexName": "MyVectorIndex"
    }
  }
}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `name` | Name for the knowledge base |
| `roleArn` | ARN of service role |
| `knowledgeBaseConfiguration` | Contains vector configuration |
| `storageConfiguration` | Vector store configuration |

---

## Data Source Sync

Once your knowledge base is ready:

1. Select your knowledge base in console
2. Select **Sync** within data source overview
3. Sync whenever you want to update content

### Sync Triggers
- Manual sync from console
- Scheduled sync
- API-triggered sync

---

## Chunking Strategies

### Default Chunking
- Automatic chunking based on document structure
- Respects paragraph and section boundaries

### Fixed-Size Chunking
- Consistent chunk sizes
- Configurable overlap

### Semantic Chunking
- Groups semantically related content
- Better for complex documents

### No Chunking
- Treats each document as a single chunk
- For small documents

---

## Querying Knowledge Bases

### RetrieveAndGenerate API

Retrieves relevant information and generates response:

```python
import boto3

client = boto3.client('bedrock-agent-runtime')

response = client.retrieve_and_generate(
    input={
        'text': 'What are the company policies on remote work?'
    },
    retrieveAndGenerateConfiguration={
        'type': 'KNOWLEDGE_BASE',
        'knowledgeBaseConfiguration': {
            'knowledgeBaseId': 'KNOWLEDGE_BASE_ID',
            'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0'
        }
    }
)

print(response['output']['text'])
```

### Retrieve API

Retrieves relevant chunks without generation:

```python
response = client.retrieve(
    knowledgeBaseId='KNOWLEDGE_BASE_ID',
    retrievalQuery={
        'text': 'remote work policies'
    },
    retrievalConfiguration={
        'vectorSearchConfiguration': {
            'numberOfResults': 5
        }
    }
)

for result in response['retrievalResults']:
    print(result['content']['text'])
    print(result['score'])
```

---

## Best Practices

### Data Preparation
1. **Clean data** - Remove duplicates and irrelevant content
2. **Structure documents** - Use clear headings and sections
3. **Include metadata** - Add tags for filtering
4. **Regular updates** - Keep content current

### Chunking
1. **Match chunk size to query type** - Smaller for precise, larger for context
2. **Use overlap** - Prevent context loss at boundaries
3. **Test different strategies** - Compare retrieval quality

### Performance
1. **Use appropriate embeddings model** - Match to content type
2. **Configure reranking** - Improve result quality
3. **Monitor latency** - Optimize for use case
4. **Use caching** - For repeated queries

---

## Multimodal Support

### For Image-Heavy Content

- **Amazon Titan Multimodal Embeddings G1**:
  - Requires S3 content bucket
  - Optimized for image search
  - Use default parser

- **Cohere Embed v3**:
  - Supports mixed text and image
  - Works with any parser

### Storage
- Multimodal data stored in S3
- Images extracted from documents
- Can be returned during queries

---

## Additional Resources

- [Data Source Connectors](https://docs.aws.amazon.com/bedrock/latest/userguide/data-source-connectors.html)
- [Customize Ingestion](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-data-source-customize-ingestion.html)
- [Security Configurations](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-create-security.html)
- [Supported Models and Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-supported.html)