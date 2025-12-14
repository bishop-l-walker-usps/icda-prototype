"""
Knowledge Document Indexing for ICDA RAG
========================================

Handles document upload, chunking, embedding, and indexing into OpenSearch.
Reuses the existing Titan embeddings and OpenSearch infrastructure.

Usage:
    knowledge = KnowledgeManager(embedder, vector_index)
    result = await knowledge.index_document(file_path, tags=["api", "design"])
    results = await knowledge.search("how does address verification work")
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingClient


class DocumentProcessor:
    """Extract and chunk content from various file formats."""

    CHUNK_SIZE = 512  # tokens (approximate)
    CHUNK_OVERLAP = 50

    def process_file(self, path: Path) -> list[dict]:
        """Process a file into chunks."""
        suffix = path.suffix.lower()

        try:
            if suffix == ".txt":
                content = path.read_text(encoding="utf-8", errors="ignore")
            elif suffix == ".md":
                content = path.read_text(encoding="utf-8", errors="ignore")
            elif suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                content = self._json_to_text(data)
            elif suffix == ".pdf":
                content = self._read_pdf(path)
            elif suffix == ".docx":
                content = self._read_docx(path)
            else:
                # Try as plain text
                content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return []

        if not content:
            return []

        return self.chunk_text(content, path.name)

    def process_text(self, content: str, source_name: str = "direct") -> list[dict]:
        """Process raw text content into chunks."""
        return self.chunk_text(content, source_name)

    def chunk_text(self, content: str, source_name: str) -> list[dict]:
        """Split content into overlapping chunks at paragraph boundaries."""
        content = self._clean_text(content)
        if not content:
            return []

        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para.split())  # Rough token estimate

            if para_size > self.CHUNK_SIZE:
                # Flush current
                if current_chunk:
                    chunks.append(self._make_chunk(current_chunk, source_name, len(chunks)))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_size = len(sent.split())
                    if current_size + sent_size > self.CHUNK_SIZE:
                        if current_chunk:
                            chunks.append(self._make_chunk(current_chunk, source_name, len(chunks)))
                        current_chunk = [current_chunk[-1]] if current_chunk else []
                        current_size = len(current_chunk[0].split()) if current_chunk else 0
                    current_chunk.append(sent)
                    current_size += sent_size

            elif current_size + para_size > self.CHUNK_SIZE:
                chunks.append(self._make_chunk(current_chunk, source_name, len(chunks)))
                # Overlap: keep last item
                current_chunk = [current_chunk[-1], para] if current_chunk else [para]
                current_size = sum(len(p.split()) for p in current_chunk)
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append(self._make_chunk(current_chunk, source_name, len(chunks)))

        return chunks

    def _make_chunk(self, parts: list[str], source: str, index: int) -> dict:
        text = "\n\n".join(p for p in parts if p)
        return {
            "text": text,
            "source": source,
            "chunk_index": index,
            "char_count": len(text)
        }

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()

    def _json_to_text(self, data: Any, prefix: str = "") -> str:
        if isinstance(data, dict):
            parts = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    parts.append(f"{prefix}{key}:")
                    parts.append(self._json_to_text(value, "  "))
                else:
                    parts.append(f"{prefix}{key}: {value}")
            return "\n".join(parts)
        elif isinstance(data, list):
            parts = []
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    parts.append(f"{prefix}[{i}]:")
                    parts.append(self._json_to_text(item, prefix + "  "))
                else:
                    parts.append(f"{prefix}- {item}")
            return "\n".join(parts)
        return str(data)

    def _read_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            print("pypdf not installed")
            return ""

    def _read_docx(self, path: Path) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            print("python-docx not installed")
            return ""


class KnowledgeManager:
    """
    Manages RAG knowledge documents in OpenSearch.

    Index schema (icda-knowledge):
      - doc_id: unique document identifier
      - chunk_id: unique chunk identifier
      - filename: original filename
      - chunk_index: position in document
      - text: chunk content
      - embedding: Titan 1024-dim vector
      - tags: list of tags
      - category: document category
      - indexed_at: timestamp
    """

    INDEX_NAME = "icda-knowledge"

    INDEX_MAPPING = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "filename": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "chunk_index": {"type": "integer"},
                "text": {"type": "text"},
                "tags": {"type": "keyword"},
                "category": {"type": "keyword"},
                "indexed_at": {"type": "date"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib"
                    }
                }
            }
        }
    }

    def __init__(self, embedder: EmbeddingClient, opensearch_client):
        self.embedder = embedder
        self.client = opensearch_client  # AsyncOpenSearch from vector_index
        self.processor = DocumentProcessor()
        self.available = False

    async def ensure_index(self) -> bool:
        """Create knowledge index if it doesn't exist."""
        if not self.client:
            return False
        try:
            if not await self.client.indices.exists(index=self.INDEX_NAME):
                await self.client.indices.create(index=self.INDEX_NAME, body=self.INDEX_MAPPING)
                print(f"Created knowledge index: {self.INDEX_NAME}")
            self.available = True
            return True
        except Exception as e:
            print(f"Failed to create knowledge index: {e}")
            return False

    async def index_document(
        self,
        content: str | Path,
        filename: str = None,
        tags: list[str] = None,
        category: str = "general"
    ) -> dict:
        """
        Index a document into the knowledge base.

        Args:
            content: File path or raw text content
            filename: Display name (required if content is text)
            tags: List of tags for filtering
            category: Category (e.g., "api", "architecture", "meeting-notes")

        Returns:
            {"success": bool, "doc_id": str, "chunks_indexed": int}
        """
        if not self.available:
            return {"success": False, "error": "Knowledge index not available"}

        # Process content
        if isinstance(content, Path) or (isinstance(content, str) and Path(content).exists()):
            path = Path(content)
            chunks = self.processor.process_file(path)
            filename = filename or path.name
        else:
            if not filename:
                return {"success": False, "error": "filename required for text content"}
            chunks = self.processor.process_text(content, filename)

        if not chunks:
            return {"success": False, "error": "No content extracted"}

        # Generate doc_id from content hash
        content_hash = hashlib.sha256(
            "".join(c["text"] for c in chunks).encode()
        ).hexdigest()[:12]
        doc_id = f"{Path(filename).stem}_{content_hash}"

        # Delete existing (for re-uploads)
        await self.delete_document(doc_id)

        # Index chunks
        indexed = 0
        errors = 0
        now = datetime.utcnow().isoformat()

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embedder.embed(chunk["text"])

            if not embedding:
                errors += 1
                continue

            doc = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk["text"],
                "tags": tags or [],
                "category": category,
                "indexed_at": now,
                "embedding": embedding
            }

            try:
                await self.client.index(index=self.INDEX_NAME, id=chunk_id, body=doc)
                indexed += 1
            except Exception as e:
                print(f"Index error: {e}")
                errors += 1

        await self.client.indices.refresh(index=self.INDEX_NAME)

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "chunks_indexed": indexed,
            "errors": errors,
            "category": category,
            "tags": tags or []
        }

    async def search(
        self,
        query: str,
        limit: int = 5,
        tags: list[str] = None,
        category: str = None
    ) -> dict:
        """Semantic search over knowledge base."""
        if not self.available:
            return {"success": False, "hits": [], "error": "Not available"}

        embedding = self.embedder.embed(query)
        if not embedding:
            return {"success": False, "hits": [], "error": "Embedding failed"}

        # Build filter
        filters = []
        if tags:
            filters.append({"terms": {"tags": tags}})
        if category:
            filters.append({"term": {"category": category}})

        if filters:
            search_body = {
                "size": limit,
                "query": {
                    "bool": {
                        "filter": filters,
                        "must": [{"knn": {"embedding": {"vector": embedding, "k": limit * 2}}}]
                    }
                }
            }
        else:
            search_body = {
                "size": limit,
                "query": {"knn": {"embedding": {"vector": embedding, "k": limit}}}
            }

        try:
            resp = await self.client.search(index=self.INDEX_NAME, body=search_body)
            hits = []
            for hit in resp["hits"]["hits"]:
                src = hit["_source"]
                hits.append({
                    "doc_id": src["doc_id"],
                    "filename": src["filename"],
                    "chunk_index": src["chunk_index"],
                    "text": src["text"],
                    "category": src["category"],
                    "tags": src["tags"],
                    "score": round(hit["_score"], 4)
                })
            return {"success": True, "query": query, "hits": hits}
        except Exception as e:
            return {"success": False, "hits": [], "error": str(e)}

    async def list_documents(self, category: str = None, limit: int = 50) -> list[dict]:
        """List unique documents in the knowledge base."""
        if not self.available:
            return []

        agg_body: dict = {
            "size": 0,
            "aggs": {
                "documents": {
                    "terms": {"field": "doc_id", "size": limit},
                    "aggs": {
                        "doc_info": {
                            "top_hits": {
                                "size": 1,
                                "_source": ["filename", "category", "tags", "indexed_at"]
                            }
                        },
                        "chunk_count": {"value_count": {"field": "chunk_id"}}
                    }
                }
            }
        }

        if category:
            agg_body["query"] = {"term": {"category": category}}

        try:
            resp = await self.client.search(index=self.INDEX_NAME, body=agg_body)
            docs = []
            for bucket in resp["aggregations"]["documents"]["buckets"]:
                info = bucket["doc_info"]["hits"]["hits"][0]["_source"]
                docs.append({
                    "doc_id": bucket["key"],
                    "filename": info["filename"],
                    "category": info["category"],
                    "tags": info["tags"],
                    "indexed_at": info["indexed_at"],
                    "chunk_count": int(bucket["chunk_count"]["value"])
                })
            return docs
        except Exception as e:
            print(f"List error: {e}")
            return []

    async def delete_document(self, doc_id: str) -> dict:
        """Delete all chunks for a document."""
        if not self.available:
            return {"deleted": 0}

        try:
            resp = await self.client.delete_by_query(
                index=self.INDEX_NAME,
                body={"query": {"term": {"doc_id": doc_id}}}
            )
            return {"deleted": resp.get("deleted", 0)}
        except Exception as e:
            return {"deleted": 0, "error": str(e)}

    async def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        if not self.available:
            return {"available": False}

        try:
            count = await self.client.count(index=self.INDEX_NAME)

            agg_resp = await self.client.search(
                index=self.INDEX_NAME,
                body={
                    "size": 0,
                    "aggs": {
                        "unique_docs": {"cardinality": {"field": "doc_id"}},
                        "categories": {"terms": {"field": "category", "size": 100}},
                        "tags": {"terms": {"field": "tags", "size": 100}}
                    }
                }
            )

            return {
                "available": True,
                "total_chunks": count["count"],
                "unique_documents": int(agg_resp["aggregations"]["unique_docs"]["value"]),
                "categories": {
                    b["key"]: b["doc_count"]
                    for b in agg_resp["aggregations"]["categories"]["buckets"]
                },
                "tags": {
                    b["key"]: b["doc_count"]
                    for b in agg_resp["aggregations"]["tags"]["buckets"]
                }
            }
        except Exception as e:
            return {"available": True, "error": str(e)}
