"""pgvector-backed semantic memory store using the same PostgreSQL as the ledger."""

import json
from typing import Any

from sqlalchemy import Column, String, Text, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class SupplierEmbeddingModel(Base):
    __tablename__ = "supplier_embeddings"

    supplier_id = Column(String(64), primary_key=True)
    embedding = Column(Text, nullable=False)  # Stored as JSON array string
    metadata_json = Column(Text, nullable=False)


class PgVectorMemoryStore:
    """
    Async semantic memory backed by pgvector in the same PostgreSQL.
    Embeddings are simple JSON arrays (placeholder for real vectors).
    Uses pgvector's vector type and <=> operator for similarity search.
    """

    def __init__(self, db_url: str) -> None:
        self.engine = create_async_engine(db_url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_schema(self) -> None:
        """Create pgvector extension and tables."""
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
            # Ensure vector column exists with proper type
            await conn.execute(
                text("""
                ALTER TABLE supplier_embeddings
                ALTER COLUMN embedding TYPE vector(384)
                USING embedding::vector(384)
            """)
            )

    def _embed_profile(self, profile: dict[str, Any]) -> list[float]:
        """
        Deterministic embedding from profile fields.
        In production, replace with sentence-transformers or OpenAI.
        """
        # Simple 384-dim hash-based embedding (deterministic, fast)
        import hashlib
        import struct

        text_repr = json.dumps(profile, sort_keys=True)
        hash_bytes = hashlib.sha256(text_repr.encode()).digest()

        # Expand 32 bytes to 384 floats via repeated hashing
        embedding = []
        seed = hash_bytes
        for _ in range(48):  # 48 * 32 / 4 = 384 floats
            seed = hashlib.sha256(seed).digest()
            for i in range(0, 32, 4):
                f = struct.unpack("f", seed[i : i + 4])[0]
                embedding.append(float(f))

        # Normalize to [-1, 1]
        embedding = [max(-1.0, min(1.0, e)) for e in embedding]
        return embedding

    async def index_supplier(self, supplier_id: str, profile: dict[str, Any]) -> None:
        """Store or update supplier profile embedding."""
        embedding = self._embed_profile(profile)
        embedding_str = "[" + ",".join(str(round(e, 6)) for e in embedding) + "]"
        metadata = json.dumps(profile, sort_keys=True)

        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    text("""
                    INSERT INTO supplier_embeddings (supplier_id, embedding, metadata_json)
                    VALUES (:sid, CAST(:emb AS vector), :meta)
                    ON CONFLICT (supplier_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        metadata_json = EXCLUDED.metadata_json
                """),
                    {
                        "sid": supplier_id,
                        "emb": embedding_str,
                        "meta": metadata,
                    },
                )
            await session.commit()

    async def query_similar_suppliers(
        self,
        query_profile: dict[str, Any],
        n_results: int = 3,
    ) -> list[dict[str, Any]]:
        """Find suppliers with similar behavioral profiles."""
        query_vec = self._embed_profile(query_profile)
        query_str = "[" + ",".join(str(round(e, 6)) for e in query_vec) + "]"

        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                SELECT supplier_id, metadata_json,
                       embedding <=> CAST(:qv AS vector) as distance
                FROM supplier_embeddings
                ORDER BY embedding <=> CAST(:qv AS vector)
                LIMIT :n
            """),
                {"qv": query_str, "n": n_results},
            )

            rows = result.mappings().all()
            return [
                {
                    "supplier_id": r["supplier_id"],
                    "distance": float(r["distance"]),
                    "metadata": json.loads(r["metadata_json"]),
                }
                for r in rows
            ]

    async def get_profile(self, supplier_id: str) -> dict[str, Any] | None:
        """Retrieve stored profile for a supplier."""
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                SELECT metadata_json FROM supplier_embeddings
                WHERE supplier_id = :sid
            """),
                {"sid": supplier_id},
            )
            row = result.scalar_one_or_none()
            return json.loads(row) if row else None
