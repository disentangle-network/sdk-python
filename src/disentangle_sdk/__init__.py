"""Disentangle SDK - Python client for the Disentangle Protocol.

Example:
    >>> from disentangle_sdk import DisentangleAgent
    >>> agent = DisentangleAgent("http://localhost:8000")
    >>> identity = agent.register(agent_type="agi")
    >>> print(f"Registered as {identity.did}")
"""

from .client import DisentangleAgent
from .types import AgentIdentity, CapabilityHandle, CoherenceReport
from .exceptions import (
    DisentangleError,
    DIDNotFoundError,
    CapabilityDeniedError,
    NodeConnectionError,
    NotRegisteredError,
)

__version__ = "0.1.0"

__all__ = [
    "DisentangleAgent",
    "AgentIdentity",
    "CapabilityHandle",
    "CoherenceReport",
    "DisentangleError",
    "DIDNotFoundError",
    "CapabilityDeniedError",
    "NodeConnectionError",
    "NotRegisteredError",
]
