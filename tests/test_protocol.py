"""
Tests for negotiation protocol and message validation.
"""

import pytest
from core.negotiation_protocol import NegotiationMessage, MessageType, ProtocolValidator


def test_parse_offer():
    text = '{"type": "offer", "material": "steel", "quantity": 1000, "unit_price": 450.50, "delivery_date": "2026-07-15", "payment_terms": "net_30"}'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.OFFER
    assert msg.material == "steel"
    assert msg.quantity == 1000
    assert msg.unit_price == 450.50


def test_parse_counter():
    text = '{"type": "counter", "material": "aluminum", "quantity": 500, "counter_price": 2100.0, "justification": "Market oversupply", "deadline": "2026-06-30"}'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.COUNTER
    assert msg.counter_price == 2100.0


def test_parse_accept():
    text = '{"type": "accept", "material": "copper", "quantity": 200, "final_price": 8200.0, "delivery_date": "2026-08-01"}'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.ACCEPT
    assert ProtocolValidator.is_terminal(msg)


def test_parse_reject():
    text = '{"type": "reject", "material": "plastic", "reason": "Price too high"}'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.REJECT
    assert ProtocolValidator.is_terminal(msg)


def test_invalid_material():
    text = '{"type": "offer", "material": "unobtainium", "quantity": 100, "unit_price": 100.0}'
    msg = NegotiationMessage.from_llm_output(text)
    errors = ProtocolValidator.validate(msg)
    assert len(errors) > 0
    assert "Invalid material" in errors[0]


def test_negative_price():
    text = '{"type": "offer", "material": "steel", "quantity": 100, "unit_price": -50.0}'
    msg = NegotiationMessage.from_llm_output(text)
    errors = ProtocolValidator.validate(msg)
    assert any("positive" in e for e in errors)


def test_invalid_date():
    text = '{"type": "offer", "material": "steel", "quantity": 100, "unit_price": 450.0, "delivery_date": "15-07-2026"}'
    msg = NegotiationMessage.from_llm_output(text)
    errors = ProtocolValidator.validate(msg)
    assert any("Invalid date" in e for e in errors)


def test_markdown_wrapper():
    text = '```json\n{"type": "offer", "material": "steel", "quantity": 100, "unit_price": 450.0}\n```'
    msg = NegotiationMessage.from_llm_output(text)
    assert msg.type == MessageType.OFFER
