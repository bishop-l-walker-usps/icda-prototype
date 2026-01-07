#!/usr/bin/env python3
"""
Knowledge Base Reindex CLI
==========================
Standalone script for reindexing the knowledge base without requiring the server.

Usage:
    python reindex_knowledge.py [OPTIONS]

Options:
    --stats           Show index statistics only
    --list            List all indexed documents
    --incremental     Only index new/modified files (default)
    --full            Force full reindex (delete all, reindex everything)
    --verify          Verify index integrity, report orphans
    --cleanup         Remove orphaned entries from index
    --dry-run         Show what would be done without doing it
    --verbose         Detailed output
    --dir PATH        Knowledge directory (default: ./knowledge)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from icda.config import Config
from icda.embedding_client import EmbeddingClient
from icda.knowledge import KnowledgeManager
from icda.knowledge_index_state import (
    compute_file_hash, load_index_state, save_index_state,
    update_file_state, remove_file_state, needs_reindex,
    get_orphaned_entries, create_empty_state, mark_full_reindex, get_stats as get_state_stats,
    SUPPORTED_EXTENSIONS
)

# Default directories
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
INDEX_STATE_FILE = KNOWLEDGE_DIR / ".index_state.json"


async def get_opensearch_client(cfg: Config):
    """Connect to OpenSearch and return client."""
    try:
        from opensearchpy import AsyncOpenSearch
        client = AsyncOpenSearch(
            hosts=[cfg.opensearch_host],
            http_compress=True,
            use_ssl=False,
            verify_certs=False
        )
        # Test connection
        info = await client.info()
        print(f"Connected to OpenSearch: {info['version']['number']}")
        return client
    except Exception as e:
        print(f"ERROR: Cannot connect to OpenSearch: {e}")
        print(f"       Host: {cfg.opensearch_host}")
        print("       Start with: docker-compose up -d opensearch")
        return None


async def show_stats(knowledge_manager: KnowledgeManager, state: dict):
    """Show knowledge base statistics."""
    print("\n" + "=" * 50)
    print("  Knowledge Base Statistics")
    print("=" * 50)

    # OpenSearch stats
    kb_stats = await knowledge_manager.get_stats()
    print(f"\nOpenSearch Index:")
    print(f"  Backend: {kb_stats.get('backend', 'unknown')}")
    print(f"  Documents: {kb_stats.get('unique_documents', 0)}")
    print(f"  Chunks: {kb_stats.get('total_chunks', 0)}")

    if kb_stats.get("categories"):
        print(f"  Categories: {kb_stats['categories']}")

    # Index state stats
    state_stats = get_state_stats(state)
    print(f"\nIndex State Tracking:")
    print(f"  Tracked files: {state_stats['tracked_files']}")
    print(f"  Total chunks: {state_stats['total_chunks']}")
    print(f"  Last full index: {state_stats['last_full_index'] or 'Never'}")

    # Check for orphans
    orphans = get_orphaned_entries(state, KNOWLEDGE_DIR)
    if orphans:
        print(f"\n  WARNING: {len(orphans)} orphaned entries detected")
        print("           Run with --cleanup to remove them")

    print()


async def list_documents(knowledge_manager: KnowledgeManager):
    """List all indexed documents."""
    print("\n" + "=" * 50)
    print("  Indexed Documents")
    print("=" * 50 + "\n")

    docs = await knowledge_manager.list_documents(limit=1000)
    if not docs:
        print("No documents indexed.")
        return

    for doc in docs:
        print(f"  [{doc.get('category', 'general')}] {doc.get('filename', 'unknown')}")
        print(f"      ID: {doc.get('doc_id', 'unknown')[:20]}...")
        print(f"      Chunks: {doc.get('chunk_count', 0)}")
        if doc.get('tags'):
            print(f"      Tags: {', '.join(doc['tags'])}")
        print()

    print(f"Total: {len(docs)} documents")


async def verify_index(state: dict, verbose: bool = False):
    """Verify index integrity and report issues."""
    print("\n" + "=" * 50)
    print("  Index Integrity Check")
    print("=" * 50 + "\n")

    issues = []

    # Check for orphaned entries
    orphans = get_orphaned_entries(state, KNOWLEDGE_DIR)
    if orphans:
        issues.append(f"{len(orphans)} orphaned entries (indexed but file deleted)")
        if verbose:
            for path, doc_id in orphans:
                print(f"  ORPHAN: {path}")

    # Check for untracked files
    untracked = []
    for filepath in KNOWLEDGE_DIR.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if filepath.name.startswith(".") or filepath.name.lower() == "readme.md":
            continue

        relative_path = str(filepath.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")
        if relative_path not in state.get("files", {}):
            untracked.append(relative_path)

    if untracked:
        issues.append(f"{len(untracked)} untracked files (not indexed)")
        if verbose:
            for path in untracked:
                print(f"  UNTRACKED: {path}")

    # Check for modified files
    modified = []
    for relative_path, file_state in state.get("files", {}).items():
        filepath = KNOWLEDGE_DIR / relative_path
        if not filepath.exists():
            continue
        try:
            current_hash = compute_file_hash(filepath)
            if file_state.get("content_hash") != current_hash:
                modified.append(relative_path)
        except Exception:
            pass

    if modified:
        issues.append(f"{len(modified)} modified files (need reindex)")
        if verbose:
            for path in modified:
                print(f"  MODIFIED: {path}")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues found. Index is healthy.")

    print()
    return len(issues) == 0


async def cleanup_orphans(knowledge_manager: KnowledgeManager, state: dict, dry_run: bool = False):
    """Remove orphaned entries from the index."""
    print("\n" + "=" * 50)
    print("  Orphan Cleanup")
    print("=" * 50 + "\n")

    orphans = get_orphaned_entries(state, KNOWLEDGE_DIR)
    if not orphans:
        print("No orphaned entries found.")
        return

    print(f"Found {len(orphans)} orphaned entries.")

    if dry_run:
        print("\n[DRY RUN] Would remove:")
        for path, doc_id in orphans:
            print(f"  - {path} (doc_id: {doc_id[:20] if doc_id else 'none'}...)")
        return

    removed = 0
    for path, doc_id in orphans:
        try:
            if doc_id:
                await knowledge_manager.delete_document(doc_id)
            remove_file_state(state, path)
            print(f"  Removed: {path}")
            removed += 1
        except Exception as e:
            print(f"  ERROR: {path} - {e}")

    save_index_state(INDEX_STATE_FILE, state)
    print(f"\nRemoved {removed} orphaned entries.")


async def reindex_incremental(
    knowledge_manager: KnowledgeManager,
    state: dict,
    dry_run: bool = False,
    verbose: bool = False
):
    """Incrementally reindex new/modified files."""
    print("\n" + "=" * 50)
    print("  Incremental Reindex")
    print("=" * 50 + "\n")

    indexed = 0
    skipped = 0
    failed = 0

    for filepath in KNOWLEDGE_DIR.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if filepath.name.startswith(".") or filepath.name.lower() == "readme.md":
            continue

        relative_path = str(filepath.relative_to(KNOWLEDGE_DIR)).replace("\\", "/")

        try:
            current_hash = compute_file_hash(filepath)
        except Exception as e:
            if verbose:
                print(f"  ERROR (hash): {relative_path} - {e}")
            failed += 1
            continue

        if not needs_reindex(state, relative_path, current_hash):
            if verbose:
                print(f"  SKIP: {relative_path} (unchanged)")
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] Would index: {relative_path}")
            indexed += 1
            continue

        category = filepath.parent.name if filepath.parent != KNOWLEDGE_DIR else "general"

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            tags = _extract_tags(content, filepath)

            result = await knowledge_manager.index_document(
                content=filepath,
                filename=relative_path,
                tags=tags,
                category=category
            )

            if result.get("success"):
                update_file_state(
                    state,
                    relative_path,
                    result.get("doc_id", ""),
                    current_hash,
                    result.get("chunks_indexed", 0)
                )
                print(f"  OK: {relative_path} ({result.get('chunks_indexed', 0)} chunks)")
                indexed += 1
            else:
                print(f"  FAIL: {relative_path} - {result.get('error')}")
                failed += 1
        except Exception as e:
            print(f"  ERROR: {relative_path} - {e}")
            failed += 1

    if not dry_run:
        save_index_state(INDEX_STATE_FILE, state)

    print(f"\nResults: {indexed} indexed, {skipped} skipped, {failed} failed")


async def reindex_full(
    knowledge_manager: KnowledgeManager,
    dry_run: bool = False,
    verbose: bool = False
):
    """Full reindex: delete all documents and reindex everything."""
    print("\n" + "=" * 50)
    print("  Full Reindex")
    print("=" * 50 + "\n")

    # Delete all existing documents
    docs = await knowledge_manager.list_documents(limit=1000)
    print(f"Deleting {len(docs)} existing documents...")

    if not dry_run:
        for doc in docs:
            await knowledge_manager.delete_document(doc["doc_id"])

    # Create fresh state
    state = create_empty_state()
    mark_full_reindex(state)

    # Reindex everything
    await reindex_incremental(knowledge_manager, state, dry_run, verbose)


def _extract_tags(content: str, filepath: Path) -> list[str]:
    """Extract tags from file content or infer from filename/path."""
    tags = []

    # Extract from YAML frontmatter if present
    if content.startswith("---"):
        lines = content.split("\n")
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                if not in_frontmatter:
                    in_frontmatter = True
                    continue
                else:
                    break
            if in_frontmatter and line.startswith("tags:"):
                tag_part = line.split(":", 1)[1].strip()
                if tag_part.startswith("["):
                    tags.extend([t.strip().strip('"\'') for t in tag_part.strip("[]").split(",")])

    # Infer from filepath
    filename_lower = filepath.stem.lower()
    if "puerto" in filename_lower or "pr-" in filename_lower:
        tags.extend(["puerto-rico", "urbanization"])
    if "address" in filename_lower:
        tags.append("addressing")
    if "example" in filename_lower:
        tags.append("examples")

    return list(set(tags))


async def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Base Reindex CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reindex_knowledge.py --stats          # Show statistics
  python reindex_knowledge.py --list           # List all documents
  python reindex_knowledge.py --incremental    # Index new/modified files
  python reindex_knowledge.py --full           # Full reindex
  python reindex_knowledge.py --verify         # Check integrity
  python reindex_knowledge.py --cleanup        # Remove orphans
"""
    )

    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--list", action="store_true", help="List all indexed documents")
    parser.add_argument("--incremental", action="store_true", help="Index new/modified files (default)")
    parser.add_argument("--full", action="store_true", help="Force full reindex")
    parser.add_argument("--verify", action="store_true", help="Verify index integrity")
    parser.add_argument("--cleanup", action="store_true", help="Remove orphaned entries")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")
    parser.add_argument("--dir", type=str, help="Knowledge directory path")

    args = parser.parse_args()

    # Use custom directory if specified
    global KNOWLEDGE_DIR, INDEX_STATE_FILE
    if args.dir:
        KNOWLEDGE_DIR = Path(args.dir)
        INDEX_STATE_FILE = KNOWLEDGE_DIR / ".index_state.json"

    if not KNOWLEDGE_DIR.exists():
        print(f"ERROR: Knowledge directory not found: {KNOWLEDGE_DIR}")
        sys.exit(1)

    # Initialize config and connections
    cfg = Config()

    print("Connecting to OpenSearch...")
    opensearch_client = await get_opensearch_client(cfg)
    if not opensearch_client:
        sys.exit(1)

    print("Initializing embeddings client...")
    embedder = EmbeddingClient(cfg.aws_region, cfg.titan_embed_model, cfg.embed_dimensions)

    print("Initializing knowledge manager...")
    knowledge_manager = KnowledgeManager(embedder, opensearch_client)
    await knowledge_manager.ensure_index()

    # Load index state
    state = load_index_state(INDEX_STATE_FILE)

    try:
        # Execute requested operation
        if args.stats:
            await show_stats(knowledge_manager, state)
        elif args.list:
            await list_documents(knowledge_manager)
        elif args.verify:
            await verify_index(state, args.verbose)
        elif args.cleanup:
            await cleanup_orphans(knowledge_manager, state, args.dry_run)
        elif args.full:
            await reindex_full(knowledge_manager, args.dry_run, args.verbose)
        else:
            # Default: incremental reindex
            await reindex_incremental(knowledge_manager, state, args.dry_run, args.verbose)

    finally:
        await opensearch_client.close()


if __name__ == "__main__":
    asyncio.run(main())
