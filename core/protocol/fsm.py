"""Finite State Machine for global auction lifecycle and bilateral bartering."""

from enum import Enum, auto
from typing import Any

from core.protocol.schema import MessageType


class GlobalAuctionState(Enum):
    INIT = auto()
    RFQ_BROADCAST = auto()
    BID_COLLECTION = auto()
    EVALUATION = auto()
    SHORTLIST_BARTER = auto()
    AWARDED = auto()
    TERMINATED = auto()


class GlobalAuctionFSM:
    """Deterministic state machine for the 1×N auction."""

    VALID_TRANSITIONS: dict[GlobalAuctionState, set[GlobalAuctionState]] = {
        GlobalAuctionState.INIT: {GlobalAuctionState.RFQ_BROADCAST},
        GlobalAuctionState.RFQ_BROADCAST: {GlobalAuctionState.BID_COLLECTION},
        GlobalAuctionState.BID_COLLECTION: {
            GlobalAuctionState.EVALUATION,
            GlobalAuctionState.TERMINATED,
        },
        GlobalAuctionState.EVALUATION: {
            GlobalAuctionState.SHORTLIST_BARTER,
            GlobalAuctionState.AWARDED,
            GlobalAuctionState.TERMINATED,
        },
        GlobalAuctionState.SHORTLIST_BARTER: {
            GlobalAuctionState.AWARDED,
            GlobalAuctionState.TERMINATED,
        },
        GlobalAuctionState.AWARDED: set(),
        GlobalAuctionState.TERMINATED: set(),
    }

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = GlobalAuctionState.INIT
        self.turn_count: int = 0

    def transition(self, new_state: GlobalAuctionState) -> bool:
        if new_state not in self.VALID_TRANSITIONS[self.state]:
            return False
        self.state = new_state
        self.turn_count += 1
        return True

    def can_transition(self, new_state: GlobalAuctionState) -> bool:
        return new_state in self.VALID_TRANSITIONS[self.state]

    def is_terminal(self) -> bool:
        return self.state in {GlobalAuctionState.AWARDED, GlobalAuctionState.TERMINATED}


# ─── BILATERAL FSM ───────────────────────────────────────────────────────────


class BilateralFSM:
    """
    Per-pair state tracker for bilateral buyer-supplier bartering.
    Enforces: no ACCEPT without prior offer, no exceeding max_turns.
    """

    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self.turn_count = 0
        self.has_offer = False
        self.is_terminal = False
        self.history: list[dict[str, Any]] = []

    def validate_message(self, msg_type: MessageType) -> bool:
        """Check if message type is legal in current state."""
        if self.is_terminal:
            return False

        # Cannot ACCEPT or COUNTER without a prior offer on the table
        if msg_type in {MessageType.ACCEPT, MessageType.COUNTER} and not self.has_offer:
            return False

        # Turn limit
        if self.turn_count >= self.max_turns:
            self.is_terminal = True
            return False

        return True

    def record_message(self, msg_type: MessageType, sender: str, payload: dict[str, Any]) -> bool:
        """
        Record message and advance state.
        Returns True if message was accepted into history.
        """
        if not self.validate_message(msg_type):
            return False

        self.history.append(
            {
                "turn": self.turn_count,
                "sender": sender,
                "type": msg_type.value,
                "payload": dict(payload),
            }
        )

        if msg_type in {MessageType.OFFER, MessageType.COUNTER}:
            self.has_offer = True

        if msg_type in {MessageType.ACCEPT, MessageType.REJECT}:
            self.is_terminal = True

        self.turn_count += 1
        return True

    def get_last_price(self) -> float | None:
        """Extract most recent price/counter_price from history."""
        for entry in reversed(self.history):
            p = (
                entry["payload"].get("unit_price")
                or entry["payload"].get("counter_price")
                or entry["payload"].get("final_price")
            )
            if p is not None:
                return float(p)
        return None

    def get_summary(self) -> dict[str, Any]:
        return {
            "turns_taken": self.turn_count,
            "is_terminal": self.is_terminal,
            "has_offer": self.has_offer,
            "history_length": len(self.history),
        }
