"""
build_qh_kg_from_excel.py

Minimal, reproducible KG construction script for the Qin–Hang metallogenic belt.

This script:
- Reads an Excel file containing geological triples (with Chinese column names).
- Creates nodes and relationships in a Neo4j database.
- Uses a simple, language-agnostic schema suitable for open-source publication.

Expected columns in the Excel file (sheet can be configured):
- 实体1       : source entity name (e.g., a deposit, belt, or unit)
- 属性类型     : attribute category (e.g., 地理位置 / "geographic", 构造 / "tectonic")
- 关系        : relation surface name (e.g., 位置, 大地构造, 控矿构造)
- 实体2       : target entity name (e.g., region, tectonic unit)
- 实体类型     : entity type/category for the source entity (e.g., 矿床, 成矿带, 构造单元)

Neo4j schema used:
- Nodes: (:GeoEntity {name, entity_type})
- Relationships: (:GeoEntity)-[:RELATION {relation, attr_type}]->(:GeoEntity)

Author: QHGeoGPT team
License: MIT
"""

import os
import sys
import argparse
import logging

import pandas as pd
from neo4j import GraphDatabase


# ------------------ Logging ------------------ #

def setup_logger() -> logging.Logger:
    """Configure UTF-8 logging to console."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        # Older Python versions do not support reconfigure; ignore.
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


logger = setup_logger()


# ------------------ KG Builder ------------------ #

class QHKGBuilder:
    """
    Simple knowledge graph (KG) builder for the Qin–Hang metallogenic belt.

    Schema:
        (s:GeoEntity {name, entity_type})
            -[r:RELATION {relation, attr_type}]->
        (t:GeoEntity {name})

    Where:
        - 'name' is the canonical entity name (from 实体1 / 实体2).
        - 'entity_type' is taken from 实体类型 (if available).
        - 'relation' is the surface relation string (from 关系).
        - 'attr_type' is the attribute category (from 属性类型).
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_from_excel(
        self,
        xlsx_path: str,
        sheet_name: str = "Sheet1",
        limit: int | None = None,
    ):
        """
        Load triples from an Excel file and write them into Neo4j.

        Parameters
        ----------
        xlsx_path : str
            Path to the Excel file.
        sheet_name : str
            Sheet name to read (default: "Sheet1").
        limit : int or None
            If provided, only the first `limit` rows are imported (useful for demo/testing).
        """
        logger.info(f"Loading KG data from: {xlsx_path} (sheet={sheet_name})")
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

        # Required columns (in Chinese, as in the original data source)
        required_cols = ["实体1", "关系", "实体2"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in Excel: {col}")

        # Drop rows missing core fields
        df = df.dropna(subset=["实体1", "关系", "实体2"])

        if limit is not None:
            df = df.head(limit)
            logger.info(f"Only importing first {limit} rows (demo mode).")

        records = df.to_dict(orient="records")
        logger.info(f"Triples to import: {len(records)}")

        with self.driver.session() as session:
            for idx, row in enumerate(records, start=1):
                source = str(row["实体1"]).strip()
                target = str(row["实体2"]).strip()
                relation = str(row["关系"]).strip()

                attr_type = str(row.get("属性类型", "")).strip() or None
                entity_type = str(row.get("实体类型", "")).strip() or None

                if not source or not target or not relation:
                    logger.warning(f"Skipping row {idx}: missing source/target/relation")
                    continue

                session.execute_write(
                    self._merge_triple,
                    source,
                    target,
                    relation,
                    attr_type,
                    entity_type,
                )

                if idx % 100 == 0:
                    logger.info(f"Imported {idx} triples...")

        logger.info("KG import finished.")

    @staticmethod
    def _merge_triple(
        tx,
        source: str,
        target: str,
        relation: str,
        attr_type: str | None,
        entity_type: str | None,
    ):
        """
        Merge a triple into Neo4j with the unified schema:

            (s:GeoEntity {name: source, entity_type})
                -[r:RELATION {relation, attr_type}]->
            (t:GeoEntity {name: target})

        If the node already exists, we keep its existing properties
        and only set entity_type if it is not already defined.
        """
        cypher = """
        MERGE (s:GeoEntity {name: $source})
        MERGE (t:GeoEntity {name: $target})
        MERGE (s)-[r:RELATION {relation: $relation}]->(t)
        SET
            r.attr_type = $attr_type,
            s.entity_type = coalesce(s.entity_type, $entity_type)
        """
        tx.run(
            cypher,
            source=source,
            target=target,
            relation=relation,
            attr_type=attr_type,
            entity_type=entity_type,
        )


# ------------------ CLI ------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Qin–Hang metallogenic belt KG in Neo4j from an Excel file."
    )
    parser.add_argument(
        "--xlsx",
        type=str,
        required=True,
        help="Path to the Excel file containing KG triples.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Sheet1",
        help="Sheet name to read from (default: Sheet1).",
    )
    parser.add_argument(
        "--kg-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687).",
    )
    parser.add_argument(
        "--kg-user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j).",
    )
    parser.add_argument(
        "--kg-password",
        type=str,
        default=None,
        help="Neo4j password. If not provided, NEO4J_PASSWORD env var will be used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on rows to import (for demo / testing).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    password = args.kg_password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError(
            "Neo4j password not provided. Use --kg-password or set NEO4J_PASSWORD."
        )

    builder = QHKGBuilder(args.kg_uri, args.kg_user, password)
    try:
        builder.import_from_excel(
            xlsx_path=args.xlsx,
            sheet_name=args.sheet,
            limit=args.limit,
        )
    finally:
        builder.close()


if __name__ == "__main__":
    main()
