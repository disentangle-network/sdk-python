"""Pydantic data models for the Disentangle SDK."""

from pydantic import BaseModel
from typing import Any


class AgentIdentity(BaseModel):
    """Agent identity information returned by registration."""

    did: str
    signing_key_hex: str
    document: dict


class CapabilityHandle(BaseModel):
    """Handle for a capability that can be delegated or invoked.

    Maps to the Rust CreateCapabilityResponse which returns:
      - capability_id_hex: hex-encoded 32-byte capability ID
      - capability: full capability JSON object
    """

    capability_id_hex: str
    capability: dict[str, Any]


class CoherenceReport(BaseModel):
    """Coherence profile for an agent on the network.

    The Rust coherence_profile_handler returns { profile: Value },
    and we construct this from the profile contents. Fields here
    must match the Rust CoherenceProfile serialization.

    Attributes:
        did: Agent's decentralized identifier
        topological_mass: Network position score
        mean_local_curvature: Average curvature with neighbors
        relational_diversity: Number of unique supporters
        temporal_depth: Historical depth of participation
        composite_score: Overall coherence score (0.0-1.0)
        decayed_mass: Time-decayed topological mass
    """

    did: str
    topological_mass: float
    mean_local_curvature: float
    relational_diversity: int
    temporal_depth: int
    composite_score: float
    decayed_mass: float
