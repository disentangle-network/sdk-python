"""Typed exceptions for the Disentangle SDK."""


class DisentangleError(Exception):
    """Base exception for Disentangle SDK."""


class DIDNotFoundError(DisentangleError):
    """DID not found on the network."""


class CapabilityDeniedError(DisentangleError):
    """Capability invocation denied (e.g., coherence too low).

    Attributes:
        coherence_score: The agent's coherence score if available
    """

    def __init__(self, message: str, coherence_score: float | None = None):
        super().__init__(message)
        self.coherence_score = coherence_score


class NodeConnectionError(DisentangleError):
    """Cannot connect to node."""


class NotRegisteredError(DisentangleError):
    """Agent is not registered. Call register() first."""
