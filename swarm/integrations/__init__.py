"""Connector integration layer for the autonomous procurement swarm.

Public surface: :class:`BaseConnector` (the deterministic port every
ERP/supplier adapter satisfies), :class:`MockConnector` (default in-memory
adapter), and the enterprise adapters (:class:`SupplierAPIConnector`,
``SAPConnector``, ``OracleConnector``, ``CoupaConnector``).
"""
