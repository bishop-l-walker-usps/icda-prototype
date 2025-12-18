# ICDA Prototype - Claude Development Standards

## CRITICAL RULE - NO EXCEPTIONS
**NEVER COMMIT WITHOUT FULL VERIFICATION**
**RUSSIAN OLYMPIC JUDGE STANDARD - NEVER LIE TO ME, MENTION EVERY FLAW, CRITICIZE LESS THAN PERFECTION**
- **MANDATORY: Test everything live before committing** - run the application, verify it works
- **MANDATORY: Run all tests and ensure they pass** - unit tests, integration tests, build tests
- **MANDATORY: Verify app actually works** - test features, read output
- **FAILURE TO DO THIS IS UNACCEPTABLE** - broken commits waste everyone's time

## Project Overview
ICDA (Intelligent Customer Data Access) is an AI-powered customer data query system featuring:
- Natural language processing via AWS Bedrock Nova
- Semantic search with OpenSearch vector store
- 8-agent query orchestration pipeline
- Address verification with 5-agent pipeline
- Knowledge base RAG (Retrieval-Augmented Generation)
- React frontend with Material-UI

## Tech Stack
| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI, Python 3.10+, Pydantic 2.10, Uvicorn |
| **Frontend** | React 18, TypeScript 5.6, Material-UI 6.1, Vite |
| **AI/ML** | AWS Bedrock Nova (LLM), Titan Embeddings |
| **Data** | Redis (cache), OpenSearch (vector store) |
| **Infra** | Docker Compose |

## Key Architecture Patterns
1. **Graceful Degradation**: LITE mode (no AWS) vs FULL mode - zero hard dependencies
2. **8-Agent Query Pipeline**: Intent -> Context -> Parser -> Resolver -> Search/Knowledge -> Nova -> Enforcer
3. **Index Federation**: Unified search across customers, knowledge, and code indexes
4. **Auto-indexing**: File watcher on `/knowledge` folder for instant RAG updates

## Project Awareness & Context
- **Always read `PLANNING.md`** at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
- **Check `TASK.md`** before starting a new task. If the task isn't listed, add it with a brief description and today's date.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in this document.

## Code Structure & Modularity
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use environment variables** for configuration (never hardcode secrets or environment-specific values).

## Directory Structure
```
icda-prototype/
├── main.py                    # FastAPI entry point
├── frontend/                  # React SPA
│   ├── src/components/        # React components
│   ├── src/hooks/             # Custom React hooks
│   ├── src/services/          # API communication
│   └── src/types/             # TypeScript interfaces
├── icda/                      # Core backend modules
│   ├── agents/                # 8-agent query orchestrator
│   ├── indexes/               # Vector index federation
│   ├── address_*.py           # Address verification pipeline
│   ├── router.py              # Query routing
│   ├── nova.py                # Bedrock Nova client
│   └── knowledge.py           # RAG document indexing
├── knowledge/                 # Auto-indexed documents
└── docker-compose.yml         # Infrastructure
```

## Testing & Reliability
- **Always create unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
- Include at least: 1 test for expected use, 1 edge case, 1 failure case.

## Task Completion
- **Mark completed tasks in `TASK.md`** immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to `TASK.md` under a "Discovered During Work" section.

## Style & Conventions

### Python (Backend)
- Follow PEP8 style guidelines
- Use type hints for all function signatures
- Use async/await for all I/O operations
- Use Pydantic models for request/response validation
- Format with Black, lint with Ruff

### TypeScript/React (Frontend)
- Use functional components with hooks
- Use TypeScript strict mode
- Follow Airbnb style guide conventions
- Use Material-UI components consistently
- Format with Prettier

## AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** - only use known, verified packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.
- **Never act lazy or do half-ass work.** Always deliver complete, high-quality implementations.
- **Never reduce functionality to solve a problem** without human author's permission.
- **Never claim work is complete** without proper live testing and validation.

## Development Environment
- **Backend**: `uvicorn main:app --reload --port 8000`
- **Frontend**: `cd frontend && npm run dev` (port 5173)
- **Docker**: `docker-compose up -d` (starts Redis, OpenSearch)
- **ALWAYS check if ports are available first** before starting new dev servers.

## Git Workflow Standards
- **Each completed feature** must be added, committed with concise commit message, and pushed.
- **Features that don't work** break the pipeline and are unacceptable.
- **Commit messages** must be descriptive but concise.
- **No broken code** should ever be committed - all commits must maintain working state.

## API Endpoints (Core)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/query` | POST | Main query endpoint with guardrails |
| `/api/health` | GET | Health check + mode status |
| `/api/cache/stats` | GET | Cache hit/miss statistics |
| `/api/knowledge/search` | GET | Knowledge base search |
| `/api/address/*` | POST/GET | Address verification |
| `/docs` | GET | Interactive Swagger API docs |

## ABSOLUTE REQUIREMENT
**FULL VERIFICATION BEFORE ANY COMMIT/PUSH** - test live app, run tests, verify pipeline
