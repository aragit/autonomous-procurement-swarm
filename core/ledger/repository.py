"""Async PostgreSQL ledger repository with hash-chain verification."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LedgerEventModel(Base):
    __tablename__ = "ledger_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    sender_id: Mapped[str] = mapped_column(String(64), nullable=False)
    message_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    current_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class PostgresLedgerRepository:
    """Append-only, hash-chained event ledger backed by PostgreSQL."""

    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_schema(self) -> None:
        """Create tables. Call once at startup."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_last_hash(self, session_id: str) -> str:
        """Retrieve last hash for chain continuity."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(LedgerEventModel.current_hash)
                .where(LedgerEventModel.session_id == session_id)
                .order_by(LedgerEventModel.id.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            return row if row else "0" * 64

    async def append_event(
        self,
        session_id: str,
        turn: int,
        sender: str,
        message_type: str,
        payload: dict[str, Any],
    ) -> str:
        """Append event with full SHA-256 hash chaining."""
        last_hash = await self.get_last_hash(session_id)
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        raw = f"{session_id}:{turn}:{sender}:{message_type}:{payload_str}:{last_hash}"
        current_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        async with self.session_factory() as session:
            async with session.begin():
                event = LedgerEventModel(
                    session_id=session_id,
                    turn_number=turn,
                    sender_id=sender,
                    message_type=message_type,
                    payload_json=payload_str,
                    prev_hash=last_hash,
                    current_hash=current_hash,
                )
                session.add(event)
            await session.commit()
        return current_hash

    async def get_events(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all events for a session in order."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(LedgerEventModel)
                .where(LedgerEventModel.session_id == session_id)
                .order_by(LedgerEventModel.id.asc())
            )
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "turn_number": r.turn_number,
                    "sender_id": r.sender_id,
                    "message_type": r.message_type,
                    "payload": json.loads(r.payload_json),
                    "prev_hash": r.prev_hash,
                    "current_hash": r.current_hash,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in rows
            ]

    async def verify_chain(self, session_id: str) -> bool:
        """Walk the chain and verify every hash link."""
        events = await self.get_events(session_id)
        if not events:
            return True  # Empty chain is valid

        for i, event in enumerate(events):
            if i == 0:
                if event["prev_hash"] != "0" * 64:
                    return False
            else:
                if event["prev_hash"] != events[i - 1]["current_hash"]:
                    return False

            # Recompute hash
            raw = (
                f"{event['session_id']}:{event['turn_number']}:"
                f"{event['sender_id']}:{event['message_type']}:"
                f"{json.dumps(event['payload'], sort_keys=True, default=str)}:"
                f"{event['prev_hash']}"
            )
            expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            if event["current_hash"] != expected:
                return False
        return True

    async def get_stats(self) -> dict[str, Any]:
        """Global ledger statistics."""
        async with self.session_factory() as session:
            from sqlalchemy import func

            total = await session.scalar(select(func.count()).select_from(LedgerEventModel))
            sessions = await session.scalar(
                select(func.count(func.distinct(LedgerEventModel.session_id)))
            )
            deals = await session.scalar(
                select(func.count()).where(LedgerEventModel.message_type == "award")
            )
        return {
            "total_events": total or 0,
            "total_sessions": sessions or 0,
            "deals_awarded": deals or 0,
        }

    async def close(self) -> None:
        await self.engine.dispose()
