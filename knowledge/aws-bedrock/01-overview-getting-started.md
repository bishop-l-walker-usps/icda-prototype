---
title: Amazon Bedrock Overview and Getting Started
category: aws-bedrock
tags:
  - aws
  - bedrock
  - getting-started
  - setup
  - iam
  - foundation-models
---

# Amazon Bedrock Overview and Getting Started

## What is Amazon Bedrock?

Amazon Bedrock is a **fully managed service** that makes high-performing foundation models (FMs) from leading AI companies and Amazon available through a unified API. It enables you to:

- Choose from a wide range of foundation models best suited for your use case
- Build generative AI applications with built-in security, privacy, and responsible AI capabilities
- Experiment with and evaluate top foundation models privately
- Customize models with your data using fine-tuning and Retrieval Augmented Generation (RAG)
- Build agents that execute tasks using enterprise systems and data sources

### Key Architecture Benefits

**Serverless Experience:**
- Get started quickly without managing infrastructure
- Privately customize foundation models with your own data
- Easily and securely integrate and deploy using AWS tools

## What Can You Do With Amazon Bedrock?

### 1. Experiment with Prompts and Configurations
- Submit prompts and generate responses with model inference
- Use API or text, image, and chat playgrounds in the console
- Test different configurations and foundation models graphically
- Use `InvokeModel` APIs for production applications

### 2. Augment Response Generation (RAG)
- Create knowledge bases by uploading data sources
- Query knowledge bases to enhance foundation model responses

### 3. Create Reasoning Applications (Agents)
- Build agents using foundation models
- Make API calls and query knowledge bases
- Reason through and execute tasks for customers

### 4. Customize Models
- Fine-tune foundation models with training data
- Perform continued-pretraining
- Adjust model parameters for specific tasks and domains

### 5. Evaluate and Compare Models
- Evaluate outputs of different models
- Use built-in or custom prompt datasets
- Determine the best-suited model for your application

### 6. Content Safety and Guardrails
- Implement safeguards for generative AI applications
- Prevent inappropriate or unwanted content

---

## Prerequisites

Before using Amazon Bedrock, you must:

1. **Sign up for an AWS account** (if you don't already have one)
2. **Create an AWS Identity and Access Management (IAM) role** with necessary permissions

### Important Note on Model Access

> Beginning June 15, 2025 access to all Amazon Bedrock foundation models is enabled by default. For Anthropic models, first-time users may need to submit use case details before they can access the model.

---

## Setup Steps

### Step 1: Create an AWS Account

1. Open https://portal.aws.amazon.com/billing/signup
2. Follow the online instructions
3. Complete phone verification as part of the sign-up process
4. An **AWS account root user** is automatically created

**Security Best Practice:** Assign administrative access to a user and use the root user only for tasks requiring root-level access.

### Step 2: Secure Your AWS Account Root User

1. Sign in to the AWS Management Console as the account owner
2. **Enable Multi-Factor Authentication (MFA)** for your root user

### Step 3: Create a User with Administrative Access

1. **Enable IAM Identity Center**
2. **Grant administrative access to a user in IAM Identity Center**
3. **Sign in as the administrative user**

---

## IAM Setup for Amazon Bedrock

### Create an Amazon Bedrock Role

**Step 1:** Create a role with the AmazonBedrockFullAccess policy

**Step 2:** Create a policy for managing Amazon Bedrock model access:

```json
{
    "Version":"2012-10-17",
    "Statement": [
        {
            "Sid": "MarketplaceBedrock",
            "Effect": "Allow",
            "Action": [
                "aws-marketplace:ViewSubscriptions",
                "aws-marketplace:Unsubscribe",
                "aws-marketplace:Subscribe"
            ],
            "Resource": "*"
        }
    ]
}
```

**Step 3:** Attach the policy to your Amazon Bedrock role

### Restrict Access to Limited Time Window

Create inline policy to expire credentials:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "DateGreaterThan": {
          "aws:CurrentTime": "2024-01-01T00:00:000"
        }
      }
    }
  ]
}
```

---

## Security Best Practices

**Critical Warnings:**
- **DO NOT** use root credentials for AWS resources
- **DO NOT** put literal access keys in application files
- **DO NOT** include credential files in project areas
- Manage access keys securely
- Credentials in shared AWS credentials file are stored in plaintext

---

## Next Steps

After requesting access to foundation models:

1. **Use the Console Playgrounds** - Learn to run basic prompts
2. **Use the API** - Set up access to Amazon Bedrock operations
3. **Use SDKs** - Learn about supported Software Development Kits

---

## Additional Resources

- [Identity and access management for Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html)
- [IAM User Guide](https://docs.aws.amazon.com/IAM/latest/UserGuide/)
- [Access Amazon Bedrock foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
