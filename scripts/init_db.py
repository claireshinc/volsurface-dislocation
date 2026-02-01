#!/usr/bin/env python3
"""Initialize database tables.

Usage:
    python -m scripts.init_db
    python -m scripts.init_db --url sqlite:///./custom.db
"""

import argparse
import logging

from data.database import init_database, Base, get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize database tables."""
    parser = argparse.ArgumentParser(description="Initialize database")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Database URL (default: from settings)",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables before creating",
    )

    args = parser.parse_args()

    if args.drop:
        logger.warning("Dropping existing tables...")
        engine = get_engine(args.url)
        Base.metadata.drop_all(engine)
        logger.info("Tables dropped")

    logger.info("Creating database tables...")
    init_database(args.url)
    logger.info("Database initialization complete")

    # Print table info
    engine = get_engine(args.url)
    logger.info("Created tables:")
    for table_name in Base.metadata.tables:
        logger.info(f"  - {table_name}")


if __name__ == "__main__":
    main()
