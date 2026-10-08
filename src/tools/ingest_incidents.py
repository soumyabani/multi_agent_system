#!/usr/bin/env python
"""Ingest incident documents from JSON into Qdrant vector store."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import settings
from src.tools.embeddings import get_embedding_model
from src.tools.qdrant_client import QdrantKnowledgeClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ingestion")


def load_incidents_from_json(file_path: Path) -> list[dict[str, Any]]:
    """Load incident documents from a JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Incident file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of incident objects")

    logger.info(f"Loaded {len(data)} incidents from {file_path}")
    return data


def validate_incident(incident: dict[str, Any]) -> bool:
    """Validate that an incident has required fields."""
    required_fields = ["id", "title", "content"]
    for field in required_fields:
        if field not in incident or not incident[field]:
            logger.warning(f"Skipping incident with missing '{field}': {incident.get('id', 'unknown')}")
            return False
    return True


def ingest_incidents(file_path: Path, collection_name: str | None = None) -> int:
    """Ingest incidents from JSON file into Qdrant.

    Returns:
        int: Number of successfully ingested incidents.
    """
    logger.info(f"Starting incident ingestion from {file_path}")

    # Load incidents
    incidents = load_incidents_from_json(file_path)
    valid_incidents = [inc for inc in incidents if validate_incident(inc)]
    logger.info(f"Validated {len(valid_incidents)} incidents (skipped {len(incidents) - len(valid_incidents)})")

    if not valid_incidents:
        logger.error("No valid incidents to ingest")
        return 0

    # Initialize embedding and Qdrant clients
    embedding_model = get_embedding_model()
    qdrant_client = QdrantKnowledgeClient(collection_name=collection_name)

    # Ensure collection exists with correct vector size
    logger.info(f"Ensuring Qdrant collection exists with vector size {embedding_model.dimension}")
    qdrant_client.ensure_collection(vector_size=embedding_model.dimension)

    # Ingest each incident
    ingested_count = 0
    failed_count = 0

    for idx, incident in enumerate(valid_incidents):
        try:
            incident_id_str = incident["id"]
            content = incident["content"]
            title = incident.get("title", "")

            # Convert string ID to integer for Qdrant compatibility
            # Qdrant requires unsigned integers or UUIDs, not strings
            incident_id = idx  # Use sequential index (0, 1, 2, ...)

            # Embed the incident content
            logger.debug(f"Embedding incident {incident_id_str}: {title}")
            embedding = embedding_model.embed_query(content)

            # Build metadata, excluding content but keeping original ID
            metadata = {k: v for k, v in incident.items() if k not in ("content",)}

            # Upsert into Qdrant
            qdrant_client.upsert_document(
                document_id=incident_id,
                content=content,
                embedding=embedding,
                metadata=metadata,
            )

            logger.info(f"Ingested incident {incident_id_str}: {title}")
            ingested_count += 1

        except Exception as exc:
            logger.error(f"Failed to ingest incident {incident.get('id', 'unknown')}: {exc}")
            failed_count += 1

    # Summary
    logger.info(f"Ingestion complete: {ingested_count} succeeded, {failed_count} failed")

    if ingested_count > 0:
        logger.info(f"Incidents are now searchable in Qdrant collection '{qdrant_client.collection_name}'")

    return ingested_count


def main() -> int:
    """CLI entry point for incident ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest incident documents from JSON into Qdrant vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.tools.ingest_incidents data/sample_incidents.json
  python -m src.tools.ingest_incidents data/incidents.json --collection custom_collection
  python -m src.tools.ingest_incidents data/incidents.json --debug
        """,
    )

    parser.add_argument(
        "file",
        help="Path to JSON file containing incident documents",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Optional Qdrant collection name (defaults to config value)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    file_path = Path(args.file)
    ingested = ingest_incidents(file_path, collection_name=args.collection)

    if ingested == 0:
        logger.error("No incidents were ingested successfully")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
