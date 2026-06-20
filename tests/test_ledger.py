"""
Tests for contract ledger integrity.
"""

import pytest
from core.contract_ledger import ContractLedger, LedgerEntry


def test_ledger_append():
    ledger = ContractLedger()
    entry = LedgerEntry(
        timestamp="2026-01-01T00:00:00",
        episode_id="test-1",
        turn_number=1,
        agent_name="Buyer",
        agent_role="buyer",
        message_type="offer",
        material="steel",
        quantity=100,
        price=450.0,
        delivery_date="2026-07-01",
        payment_terms="net_30",
    )
    ledger.append(entry)
    assert len(ledger.entries) == 1
    assert ledger.last_hash != "0" * 16


def test_ledger_integrity():
    ledger = ContractLedger()
    for i in range(3):
        entry = LedgerEntry(
            timestamp=f"2026-01-0{i+1}T00:00:00",
            episode_id="test-1",
            turn_number=i+1,
            agent_name="Buyer",
            agent_role="buyer",
            message_type="offer",
            material="steel",
            quantity=100,
            price=450.0,
            delivery_date="2026-07-01",
            payment_terms="net_30",
        )
        ledger.append(entry)

    assert ledger.verify_integrity()


def test_ledger_tamper_detection():
    ledger = ContractLedger()
    entry = LedgerEntry(
        timestamp="2026-01-01T00:00:00",
        episode_id="test-1",
        turn_number=1,
        agent_name="Buyer",
        agent_role="buyer",
        message_type="offer",
        material="steel",
        quantity=100,
        price=450.0,
        delivery_date="2026-07-01",
        payment_terms="net_30",
    )
    ledger.append(entry)

    # Tamper
    ledger.entries[0].price = 999.0
    assert not ledger.verify_integrity()
