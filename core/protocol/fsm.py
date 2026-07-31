"""Finite State Machine for global auction lifecycle."""

from enum import Enum, auto
from typing import Dict, Optional, Set

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
    
    VALID_TRANSITIONS: Dict[GlobalAuctionState, Set[GlobalAuctionState]] = {
        GlobalAuctionState.INIT: {GlobalAuctionState.RFQ_BROADCAST},
        GlobalAuctionState.RFQ_BROADCAST: {GlobalAuctionState.BID_COLLECTION},
        GlobalAuctionState.BID_COLLECTION: {
            GlobalAuctionState.EVALUATION, 
            GlobalAuctionState.TERMINATED
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
