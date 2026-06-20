"""
Contract ledger for negotiation audit trail.
Immutable append-only log with hash chaining.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class LedgerEntry:
    timestamp: str
    episode_id: str
    turn_number: int
    agent_name: str
    agent_role: str
    message_type: str
    material: str
    quantity: Optional[int] = None
    price: Optional[float] = None
    delivery_date: Optional[str] = None
    payment_terms: Optional[str] = None
    justification: Optional[str] = None
    hash: str = ""
    prev_hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of entry content."""
        content = json.dumps({
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "turn_number": self.turn_number,
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "message_type": self.message_type,
            "material": self.material,
            "quantity": self.quantity,
            "price": self.price,
            "delivery_date": self.delivery_date,
            "payment_terms": self.payment_terms,
            "justification": self.justification,
            "prev_hash": self.prev_hash,
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        return asdict(self)


class ContractLedger:
    """Immutable negotiation ledger with hash chain."""

    def __init__(self):
        self.entries: List[LedgerEntry] = []
        self.last_hash = "0" * 16

    def append(self, entry: LedgerEntry):
        """Add entry with hash chaining."""
        entry.prev_hash = self.last_hash
        entry.hash = entry._compute_hash()
        self.entries.append(entry)
        self.last_hash = entry.hash

    def get_episode(self, episode_id: str) -> List[LedgerEntry]:
        """Get all entries for an episode."""
        return [e for e in self.entries if e.episode_id == episode_id]

    def verify_integrity(self) -> bool:
        """Verify hash chain integrity."""
        for i, entry in enumerate(self.entries):
            if i == 0:
                if entry.prev_hash != "0" * 16:
                    return False
            else:
                if entry.prev_hash != self.entries[i-1].hash:
                    return False
            # Recompute hash
            expected = entry._compute_hash()
            if entry.hash != expected:
                return False
        return True

    def export_json(self, filepath: str):
        """Export ledger to JSON."""
        with open(filepath, 'w') as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def get_stats(self) -> Dict:
        """Compute ledger statistics."""
        if not self.entries:
            return {}
        return {
            "total_entries": len(self.entries),
            "total_episodes": len(set(e.episode_id for e in self.entries)),
            "deals_closed": sum(1 for e in self.entries if e.message_type == "accept"),
            "deadlocks": sum(1 for e in self.entries if e.message_type == "reject"),
            "avg_price": sum(e.price for e in self.entries if e.price) /
                        max(sum(1 for e in self.entries if e.price), 1),
        }
