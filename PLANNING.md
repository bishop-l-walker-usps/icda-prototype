# PLANNING.md
# ICDA Prototype - Project Architecture & Planning

## Project Architecture

### Frontend Stack
- **Framework**: React 18.3 + TypeScript 5.6
- **UI Library**: Material-UI (MUI) 6.1 with custom theming
- **Build Tool**: Vite 5.4
- **State Management**: React hooks (useState, useCallback) - no external state library
- **Key Libraries**: Axios 1.7 (HTTP), UUID 10.0 (session IDs)

### Backend Stack
- **Framework**: FastAPI 0.115 (Python 3.11+)
- **Language**: Python 3.11+ with type hints throughout
- **Key Libraries**:
  - boto3 1.35 (AWS SDK - Bedrock, Titan)
  - redis[hiredis] 5.2 (async caching)
  - opensearch-py[async] 3.1 (vector search)
  - pydantic 2.10 (data validation)
- **Database**: JSON file (customer_data.json - 50K records), Redis (cache), OpenSearch (vectors)
- **Authentication**: AWS IAM credentials for Bedrock access

### Infrastructure
- **Containerization**: Docker multi-stage build + Docker Compose
- **CI/CD**: GitHub Actions (7-stage security scanning pipeline)
- **Cloud Provider**: AWS (Bedrock Nova, Titan Embeddings, OpenSearch)
- **Monitoring**: Structured logging with Python logging module
- **Security**: Guardrails (PII blocking), non-root Docker user, CORS configuration

## Development Workflow

### 1. Pre-Development Setup
1. Read `.claude/CLAUDE.md` for development standards
2. Check `TASK.md` for current sprint work
3. Reference `.claude/INITIAL.md` for technical specifications
4. Ensure AWS credentials configured (~/.aws/credentials or environment)

### 2. Feature Development Process
```bash
# 1. Start development environment
start-dev.bat  # Windows - starts backend + frontend concurrently

# Or manually:
# Backend (terminal 1)
python -m uvicorn main:app --reload --port 8000

# Frontend (terminal 2)
cd frontend && npm run dev  # Vite on port 5173

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Development cycle
npm run build  # Frontend build check
python -m pytest tests/  # Run tests
python -m py_compile main.py icda/*.py  # Syntax check

# 4. Before commit
npm run build && python -m pytest tests/
```

### 3. Testing Strategy
- **Unit Tests**: pytest for backend, located in `/tests`
- **Integration Tests**: API endpoint tests with TestClient
- **E2E Tests**: Manual testing via UI (no automated E2E yet)
- **Coverage Requirements**: Critical paths must have tests

## Naming Conventions

### Files & Directories
- **Components**: PascalCase (e.g., `ChatPanel.tsx`, `QueryInput.tsx`)
- **Utilities/Services**: camelCase (e.g., `api.ts`, `useQuery.ts`)
- **Python Modules**: snake_case (e.g., `vector_index.py`, `address_pipeline.py`)
- **Constants**: UPPER_SNAKE_CASE
- **Types/Interfaces**: PascalCase (e.g., `QueryResponse`, `HealthStatus`)
- **Test Files**: `test_*.py` for pytest discovery

### Code Structure
```
icda-prototype/
├── frontend/
│   ├── src/
│   │   ├── components/       # React components (PascalCase)
│   │   │   └── chat/         # Chat-specific components
│   │   ├── hooks/            # Custom React hooks (use*.ts)
│   │   ├── services/         # API client (api.ts)
│   │   ├── theme/            # MUI theme configuration
│   │   ├── types/            # TypeScript interfaces
│   │   └── utils/            # Helper utilities
│   ├── public/               # Static assets, favicons
│   └── scripts/              # Build/generation scripts
├── icda/                     # Python backend modules
│   ├── router.py             # Query routing pipeline
│   ├── nova.py               # AWS Bedrock Nova client
│   ├── database.py           # Customer DB operations
│   ├── cache.py              # Redis + fallback cache
│   ├── vector_index.py       # OpenSearch kNN routing
│   ├── guardrails.py         # PII/content blocking
│   ├── session.py            # Session management
│   ├── address_*.py          # 6-stage address verification
│   └── config.py             # Configuration management
├── tests/                    # Test suite
├── templates/                # Jinja2 templates (legacy)
├── main.py                   # FastAPI application entry
├── customer_data.json        # 50K customer dataset
├── Dockerfile                # Production container build
└── docker-compose.yml        # Local dev orchestration
```

## Key Architectural Decisions

### 1. Query Processing Pipeline
- **Decision**: Multi-stage pipeline with guardrails → cache → routing → execution
- **Rationale**: Separation of concerns, fail-fast on blocked content, cache optimization
- **Implementation**: `icda/router.py` orchestrates: guardrails.check() → cache.get() → vector_index.route() → execute()

### 2. AWS Bedrock Nova for NLP
- **Decision**: Use Nova Micro model with tool calling for customer queries
- **Rationale**: Cost-effective, low latency, native tool support for structured data access
- **Implementation**: `icda/nova.py` defines tools (lookup_crid, search_customers, get_stats) and handles tool execution loop

### 3. Graceful Degradation
- **Decision**: All external services (Redis, OpenSearch, Bedrock) have fallbacks
- **Rationale**: Prototype must work in varied environments without full AWS setup
- **Implementation**:
  - Redis → In-memory dict cache
  - OpenSearch → Keyword-based routing
  - Bedrock → Demo mode with mock responses

### 4. State Management (Frontend)
- **Approach**: React hooks only (no Redux/Zustand)
- **Rationale**: Lightweight for prototype, sufficient for current complexity
- **Patterns**: useState for local state, custom hooks (useQuery, useHealth) for shared logic

### 5. Error Handling
- **Frontend**: Try-catch with user-friendly error messages in chat
- **Backend**: FastAPI exception handlers, structured error responses with detail field
- **Logging**: Python logging with DEBUG/INFO/WARN/ERROR levels, no sensitive data logged

## Performance Requirements

### Query Response Standards
- **Cache Hit**: <1ms target (~0.5ms actual)
- **LOOKUP (CRID)**: <5ms target (~3ms actual)
- **DATABASE (search)**: <200ms target (~50-150ms actual)
- **NOVA (LLM)**: <2000ms target (~1000-1500ms actual)
- **Address Verify**: <100ms target (~20-80ms actual)

### System Performance
- **Build Size**: Frontend bundle <500KB gzipped
- **Response Time**: P95 <2s for LLM queries
- **Load Time**: Initial page load <3s
- **Memory Usage**: Backend <512MB under normal load

## Security & Compliance

### Data Protection
- **Data Storage**: Customer data in JSON file (no PII beyond names/addresses)
- **Authentication**: AWS IAM for Bedrock, no user auth in prototype
- **Authorization**: N/A for prototype (single user)
- **Encryption**: HTTPS in production, AWS SDK handles Bedrock encryption

### Access Control
- **Guardrails**: Block SSN, credit card, bank account, password queries
- **Input Validation**: Pydantic models for all API inputs
- **API Security**: CORS configured, rate limiting via AWS (future)
- **Infrastructure**: Non-root Docker user (UID >10000), minimal base image

## Deployment Strategy

### Environments
- **Development**: Local with `start-dev.bat`, Vite HMR, uvicorn reload
- **Staging**: N/A for prototype
- **Production**: Docker container via `docker-compose up`

### Release Process
1. Ensure all tests pass locally
2. Build frontend: `cd frontend && npm run build`
3. Build Docker image: `docker-compose build`
4. Test container: `docker-compose up` and verify health endpoint
5. Push to registry (future): `docker push`

---

**Reference Files:**
- Technical Specifications: `.claude/INITIAL.md`
- Development Standards: `.claude/CLAUDE.md`
- Current Tasks: `TASK.md`
- API Documentation: `TECHNICAL_DOCUMENTATION.md`
