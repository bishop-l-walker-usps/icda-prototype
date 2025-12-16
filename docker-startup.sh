#!/bin/bash
# ============================================
# ICDA Docker Startup Script
# ============================================
# Handles service initialization, Code RAG indexing, and server startup
# ============================================

set -e

echo "============================================"
echo "  ICDA Unified Container Starting"
echo "============================================"
echo ""

# ============================================
# Wait for Redis
# ============================================
echo "[1/4] Waiting for Redis..."
REDIS_HOST="${REDIS_URL#redis://}"
REDIS_HOST="${REDIS_HOST%%:*}"
REDIS_PORT="${REDIS_URL##*:}"
REDIS_PORT="${REDIS_PORT%%/*}"

for i in {1..30}; do
    if curl -sf "http://${REDIS_HOST:-redis}:${REDIS_PORT:-6379}" >/dev/null 2>&1 || \
       python -c "import redis; r=redis.Redis(host='${REDIS_HOST:-redis}', port=${REDIS_PORT:-6379}); r.ping()" 2>/dev/null; then
        echo "  ✓ Redis is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ⚠ Redis not responding after 30 attempts, continuing anyway..."
    fi
    sleep 1
done

# ============================================
# Wait for OpenSearch
# ============================================
echo "[2/4] Waiting for OpenSearch..."
OPENSEARCH_URL="${OPENSEARCH_HOST:-http://opensearch:9200}"

for i in {1..60}; do
    if curl -sf "${OPENSEARCH_URL}/_cluster/health" | grep -qE '(green|yellow)'; then
        echo "  ✓ OpenSearch is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ⚠ OpenSearch not ready after 60 attempts, continuing anyway..."
    fi
    sleep 2
done

# ============================================
# Initialize Indexes (via FastAPI startup)
# ============================================
echo "[3/4] Starting server and initializing indexes..."
echo "      (Knowledge documents will auto-index from /knowledge folder)"

# ============================================
# Optional: Index Customer Data
# ============================================
if [ "${AUTO_INDEX_CUSTOMERS:-false}" = "true" ]; then
    echo "[3.5/4] Auto-indexing customer data..."
    python index_customers.py --force --batch-size 100 2>/dev/null || echo "  ⚠ Customer indexing skipped (may already exist)"
fi

# ============================================
# Start the Server
# ============================================
echo "[4/4] Starting FastAPI server..."
echo ""
echo "============================================"
echo "  ICDA is starting on http://0.0.0.0:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "============================================"
echo ""

# Start uvicorn with the app
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000
