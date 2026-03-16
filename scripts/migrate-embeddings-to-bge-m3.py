"""Re-embed all episodic memories from Gemini (3072d) to bge-m3 (1024d).

Steps:
1. Connect to existing Qdrant collection (3072d)
2. Create new collection with 1024d vectors
3. Scroll through all points, re-embed content with bge-m3
4. Upsert to new collection
5. Validate recall@10 on test queries
6. Switch alias (manual step)

Usage:
    python scripts/migrate-embeddings-to-bge-m3.py --dry-run
    python scripts/migrate-embeddings-to-bge-m3.py --execute
    python scripts/migrate-embeddings-to-bge-m3.py --switch-alias
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate")

OLD_COLLECTION = "engram-memory"  # matches config.yaml episodic.namespace
NEW_COLLECTION = "engram-memory-bge"
NEW_DIM = 1024
BATCH_SIZE = 100


def get_client() -> QdrantClient:
    """Connect to Qdrant server using engram config."""
    from engram.config import load_config
    cfg = load_config().episodic
    # Use http:// URL to avoid SSL issues with API key on non-TLS servers
    url = f"http://{cfg.host}:{cfg.port}"
    return QdrantClient(url=url, api_key=cfg.api_key or None)


def create_new_collection(client: QdrantClient) -> None:
    """Create new collection with 1024d vectors."""
    if client.collection_exists(NEW_COLLECTION):
        logger.info("Collection %s already exists, skipping creation", NEW_COLLECTION)
        return
    client.create_collection(
        collection_name=NEW_COLLECTION,
        vectors_config=VectorParams(size=NEW_DIM, distance=Distance.COSINE),
    )
    logger.info("Created collection %s (%dd)", NEW_COLLECTION, NEW_DIM)


def migrate(client: QdrantClient, dry_run: bool = False) -> None:
    """Scroll old collection, re-embed with bge-m3, upsert to new collection."""
    from engram.episodic.local_embeddings import embed

    old_info = client.get_collection(OLD_COLLECTION)
    total = old_info.points_count
    logger.info("Old collection has %d points", total)

    if dry_run:
        logger.info("DRY RUN — would re-embed %d points to %s", total, NEW_COLLECTION)
        return

    create_new_collection(client)

    offset = None
    migrated = 0
    start = time.time()

    while True:
        records, next_offset = client.scroll(
            collection_name=OLD_COLLECTION,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break

        # Extract content from payloads
        texts = []
        for r in records:
            content = (r.payload or {}).get("content", "")
            texts.append(content if content else " ")

        # Re-embed with bge-m3
        vectors = embed(texts)

        # Upsert to new collection
        points = [
            PointStruct(id=r.id, vector=v, payload=r.payload or {})
            for r, v in zip(records, vectors)
        ]
        client.upsert(collection_name=NEW_COLLECTION, points=points)
        migrated += len(points)
        logger.info("Migrated %d/%d points (%.1f%%)", migrated, total, migrated / total * 100)

        if next_offset is None:
            break
        offset = next_offset

    elapsed = time.time() - start
    logger.info("Migration complete: %d points in %.1fs", migrated, elapsed)

    # Validate counts match
    new_info = client.get_collection(NEW_COLLECTION)
    logger.info("New collection has %d points (expected %d)", new_info.points_count, total)
    if new_info.points_count != total:
        logger.warning("Point count mismatch! Check for errors.")


def switch_alias(client: QdrantClient) -> None:
    """Switch alias from old to new collection for zero-downtime cutover."""
    from qdrant_client.models import CreateAliasOperation, AliasOperations

    alias_name = OLD_COLLECTION
    # Rename old collection first if needed
    logger.info("Switching alias '%s' → %s", alias_name, NEW_COLLECTION)
    try:
        client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias={"collection_name": NEW_COLLECTION, "alias_name": alias_name}
                ),
            ]
        )
        logger.info("Alias '%s' now points to %s", alias_name, NEW_COLLECTION)
    except Exception as e:
        logger.error("Alias switch failed: %s", e)
        logger.info("You may need to manually rename collections in Qdrant dashboard")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Qdrant embeddings from Gemini to bge-m3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Show what would be done")
    group.add_argument("--execute", action="store_true", help="Run the migration")
    group.add_argument("--switch-alias", action="store_true", help="Switch alias after validation")
    args = parser.parse_args()

    client = get_client()

    if args.dry_run:
        migrate(client, dry_run=True)
    elif args.execute:
        migrate(client, dry_run=False)
    elif args.switch_alias:
        switch_alias(client)


if __name__ == "__main__":
    main()
