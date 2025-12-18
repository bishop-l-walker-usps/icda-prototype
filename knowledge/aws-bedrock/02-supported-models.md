---
title: Amazon Bedrock Supported Foundation Models
category: aws-bedrock
tags:
  - aws
  - bedrock
  - models
  - claude
  - titan
  - llama
  - nova
  - mistral
---

# Supported Foundation Models in Amazon Bedrock

## Overview

Amazon Bedrock supports a comprehensive selection of foundation models from multiple providers. This document lists all supported models with their Model IDs, regional availability, input/output modalities, and streaming capabilities.

## Model Providers

- **Amazon** (Nova series, Titan series)
- **Anthropic** (Claude series)
- **AI21 Labs** (Jamba series)
- **Cohere** (Command R, Embed series)
- **DeepSeek** (R1, V3.1)
- **Google** (Gemma 3 series)
- **Meta** (Llama 3, 3.1, 3.2, 3.3, 4 series)
- **Mistral AI** (Mistral, Ministral, Magistral, Pixtral, Voxtral)
- **NVIDIA** (Nemotron series)
- **Stability AI** (Stable Diffusion, Stable Image series)
- **Writer** (Palmyra X4, X5)

---

## Amazon Models

### Nova Series (Multimodal & Text)

| Model | Model ID | Input | Output | Streaming |
|-------|----------|-------|--------|-----------|
| Nova Premier | `amazon.nova-premier-v1:0` | Text, Image, Video | Text | Yes |
| Nova Pro | `amazon.nova-pro-v1:0` | Text, Image, Video | Text | Yes |
| Nova Lite | `amazon.nova-lite-v1:0` | Text, Image, Video | Text | Yes |
| Nova Micro | `amazon.nova-micro-v1:0` | Text | Text | Yes |
| Nova 2 Lite | `amazon.nova-2-lite-v1:0` | Text, Image, Video | Text | Yes |
| Nova 2 Sonic | `amazon.nova-2-sonic-v1:0` | Speech | Speech, Text | Yes |
| Nova Sonic | `amazon.nova-sonic-v1:0` | Speech | Speech, Text | Yes |
| Nova Canvas | `amazon.nova-canvas-v1:0` | Text | Image | No |
| Nova Reel | `amazon.nova-reel-v1:0` | Text, Image | Video | No |

### Titan Series

| Model | Model ID | Input | Output | Streaming |
|-------|----------|-------|--------|-----------|
| Titan Text Large | `amazon.titan-tg1-large` | Text | Text | Yes |
| Titan Text Embeddings V2 | `amazon.titan-embed-text-v2:0` | Text | Embedding | No |
| Titan Embeddings G1 - Text | `amazon.titan-embed-text-v1` | Text | Embedding | No |
| Titan Multimodal Embeddings G1 | `amazon.titan-embed-image-v1` | Text, Image | Embedding | No |
| Titan Image Generator G1 v2 | `amazon.titan-image-generator-v2:0` | Text | Image | No |

### Amazon Rerank

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| Rerank 1.0 | `amazon.rerank-v1:0` | Text | Text |

---

## Anthropic Claude Models

| Model | Model ID | Input | Output | Streaming |
|-------|----------|-------|--------|-----------|
| Claude Opus 4.5 | `anthropic.claude-opus-4-5-20251101-v1:0` | Text, Image | Text | Yes |
| Claude Opus 4.1 | `anthropic.claude-opus-4-1-20250805-v1:0` | Text, Image | Text | Yes |
| Claude Opus 4 | `anthropic.claude-opus-4-20250514-v1:0` | Text, Image | Text | Yes |
| Claude Sonnet 4.5 | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Text, Image | Text | Yes |
| Claude Sonnet 4 | `anthropic.claude-sonnet-4-20250514-v1:0` | Text, Image | Text | Yes |
| Claude 3.7 Sonnet | `anthropic.claude-3-7-sonnet-20250219-v1:0` | Text, Image | Text | Yes |
| Claude 3.5 Haiku | `anthropic.claude-3-5-haiku-20241022-v1:0` | Text | Text | Yes |
| Claude Haiku 4.5 | `anthropic.claude-haiku-4-5-20251001-v1:0` | Text, Image | Text | Yes |
| Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` | Text, Image | Text | Yes |

---

## AI21 Labs

| Model | Model ID | Input | Output | Streaming |
|-------|----------|-------|--------|-----------|
| Jamba 1.5 Large | `ai21.jamba-1-5-large-v1:0` | Text | Text | Yes |
| Jamba 1.5 Mini | `ai21.jamba-1-5-mini-v1:0` | Text | Text | Yes |

---

## Cohere Models

| Model | Model ID | Input | Output | Streaming |
|-------|----------|-------|--------|-----------|
| Command R+ | `cohere.command-r-plus-v1:0` | Text | Text | Yes |
| Command R | `cohere.command-r-v1:0` | Text | Text | Yes |
| Embed English | `cohere.embed-english-v3` | Text | Embedding | No |
| Embed Multilingual | `cohere.embed-multilingual-v3` | Text | Embedding | No |
| Embed v4 | `cohere.embed-v4:0` | Text, Image | Embedding | No |
| Rerank 3.5 | `cohere.rerank-v3-5:0` | Text | Text | No |

---

## Meta Llama Models

### Llama 3 Series

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| Llama 3 70B Instruct | `meta.llama3-70b-instruct-v1:0` | Text | Text |
| Llama 3 8B Instruct | `meta.llama3-8b-instruct-v1:0` | Text | Text |

### Llama 3.1 Series

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| Llama 3.1 405B Instruct | `meta.llama3-1-405b-instruct-v1:0` | Text | Text |
| Llama 3.1 70B Instruct | `meta.llama3-1-70b-instruct-v1:0` | Text | Text |
| Llama 3.1 8B Instruct | `meta.llama3-1-8b-instruct-v1:0` | Text | Text |

### Llama 3.2 Series (Multimodal)

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| Llama 3.2 90B Instruct | `meta.llama3-2-90b-instruct-v1:0` | Text, Image | Text |
| Llama 3.2 11B Instruct | `meta.llama3-2-11b-instruct-v1:0` | Text, Image | Text |
| Llama 3.2 3B Instruct | `meta.llama3-2-3b-instruct-v1:0` | Text | Text |
| Llama 3.2 1B Instruct | `meta.llama3-2-1b-instruct-v1:0` | Text | Text |

### Llama 3.3 Series

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| Llama 3.3 70B Instruct | `meta.llama3-3-70b-instruct-v1:0` | Text | Text |

### Llama 4 Series (Multimodal)

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| Llama 4 Maverick 17B | `meta.llama4-maverick-17b-instruct-v1:0` | Text, Image | Text |
| Llama 4 Scout 17B | `meta.llama4-scout-17b-instruct-v1:0` | Text, Image | Text |

---

## Mistral AI Models

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| Pixtral Large (25.02) | `mistral.pixtral-large-2502-v1:0` | Text, Image | Text |
| Mistral Large 3 | `mistral.mistral-large-3-675b-instruct` | Text, Image | Text |
| Mistral Large (24.07) | `mistral.mistral-large-2407-v1:0` | Text | Text |
| Mistral Large (24.02) | `mistral.mistral-large-2402-v1:0` | Text | Text |
| Magistral Small 2509 | `mistral.magistral-small-2509` | Text, Image | Text |
| Ministral 14B 3.0 | `mistral.ministral-3-14b-instruct` | Text | Text |
| Ministral 3 8B | `mistral.ministral-3-8b-instruct` | Text | Text |
| Ministral 3B | `mistral.ministral-3-3b-instruct` | Text | Text |
| Mistral Small (24.02) | `mistral.mistral-small-2402-v1:0` | Text | Text |
| Mistral 7B Instruct | `mistral.mistral-7b-instruct-v0:2` | Text | Text |
| Mixtral 8x7B Instruct | `mistral.mixtral-8x7b-instruct-v0:1` | Text | Text |
| Voxtral Small 24B | `mistral.voxtral-small-24b-2507` | Audio | Text |
| Voxtral Mini 3B | `mistral.voxtral-mini-3b-2507` | Audio | Text |

---

## DeepSeek Models

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| DeepSeek-V3.1 | `deepseek.v3-v1:0` | Text | Text |
| DeepSeek-R1 | `deepseek.r1-v1:0` | Text | Text |

---

## Qwen Models

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| Qwen3 235B A22B | `qwen.qwen3-235b-a22b-2507-v1:0` | Text | Text |
| Qwen3 Coder 480B | `qwen.qwen3-coder-480b-a35b-v1:0` | Text | Text |
| Qwen3 VL 235B | `qwen.qwen3-vl-235b-a22b` | Text, Image | Text |
| Qwen3 32B | `qwen.qwen3-32b-v1:0` | Text | Text |
| Qwen3-Coder-30B | `qwen.qwen3-coder-30b-a3b-v1:0` | Text | Text |

---

## Stability AI Image Generation

| Model | Model ID | Input | Output |
|-------|----------|-------|--------|
| Stable Diffusion 3.5 Large | `stability.sd3-5-large-v1:0` | Text | Image |
| Stable Image Ultra 1.0 | `stability.stable-image-ultra-v1:1` | Text, Image | Image |
| Stable Image Core 1.0 | `stability.stable-image-core-v1:1` | Text, Image | Image |

---

## Key Features

- **Model IDs**: AWS Region-agnostic identifiers used in inference operations
- **Input Modalities**: Text, Image, Video, Audio, or combinations
- **Output Modalities**: Text, Image, Video, or Embedding
- **Streaming Support**: Available for most text and multimodal models via `InvokeModelWithResponseStream` and `ConverseStream` APIs
- **Cross-Region Inference Profiles**: Support for inference calls across multiple regions