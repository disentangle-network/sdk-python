"""DisentangleAgent client for interacting with a Disentangle node."""

import httpx
from typing import Any

from .types import AgentIdentity, CapabilityHandle, CoherenceReport
from .exceptions import (
    DisentangleError,
    DIDNotFoundError,
    CapabilityDeniedError,
    NodeConnectionError,
    NotRegisteredError,
)


class DisentangleAgent:
    """Client for interacting with a Disentangle node via HTTP.

    The DisentangleAgent wraps all RPC endpoints and provides typed access
    to identity, capabilities, social graph, and coherence metrics.

    All field names in request payloads and response deserialization are
    aligned with the Rust RPC handlers in identity_rpc.rs.

    Example:
        >>> agent = DisentangleAgent("http://localhost:8000")
        >>> identity = agent.register(agent_type="agi")
        >>> print(f"Registered as {identity.did}")
    """

    def __init__(self, node_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            node_url: Base URL of the Disentangle node
        """
        self._client = httpx.Client(base_url=node_url, timeout=30.0)
        self._identity: AgentIdentity | None = None
        self._signing_key_hex: str | None = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close HTTP client."""
        self.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request and handle errors.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Endpoint path
            json: JSON body for POST requests
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            NodeConnectionError: Cannot connect to node
            DIDNotFoundError: DID not found (404)
            CapabilityDeniedError: Capability denied (403)
            DisentangleError: Other errors
        """
        try:
            response = self._client.request(
                method=method,
                url=path,
                json=json,
                params=params,
            )

            # Handle error status codes
            if response.status_code == 404:
                error_msg = response.json().get("error", "Not found")
                raise DIDNotFoundError(error_msg)
            elif response.status_code == 403:
                error_data = response.json()
                error_msg = error_data.get("error", "Forbidden")
                coherence_score = error_data.get("coherence_score")
                raise CapabilityDeniedError(error_msg, coherence_score)
            elif response.status_code == 400:
                error_msg = response.json().get("error", "Bad request")
                raise DisentangleError(error_msg)
            elif not response.is_success:
                error_msg = response.json().get("error", f"HTTP {response.status_code}")
                raise DisentangleError(error_msg)

            return response.json()

        except httpx.ConnectError as e:
            raise NodeConnectionError(f"Cannot connect to node: {e}")
        except httpx.TimeoutException as e:
            raise NodeConnectionError(f"Request timeout: {e}")
        except httpx.HTTPError as e:
            raise NodeConnectionError(f"HTTP error: {e}")

    # -------------------------------------------------------------------------
    # Identity Methods
    # -------------------------------------------------------------------------

    def register(
        self,
        agent_type: str = "agi",
        runtime_attestation: str | None = None,
    ) -> AgentIdentity:
        """Register a new agent identity.

        Rust RPC: RegisterRequest { agent_type, runtime_attestation? }
        Rust RPC: RegisterResponse { did, signing_key_hex, document }

        Args:
            agent_type: Type of agent ("agi" or "human")
            runtime_attestation: Optional runtime attestation string for AGI agents

        Returns:
            AgentIdentity with DID, signing key, and document

        Raises:
            DisentangleError: Registration failed
        """
        payload: dict[str, Any] = {"agent_type": agent_type}

        if runtime_attestation is not None:
            payload["runtime_attestation"] = runtime_attestation

        response = self._request("POST", "/identity/register", json=payload)

        # Store identity locally
        self._identity = AgentIdentity(**response)
        self._signing_key_hex = response["signing_key_hex"]

        return self._identity

    @property
    def did(self) -> str:
        """Get the agent's DID.

        Returns:
            The DID string

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if self._identity is None:
            raise NotRegisteredError("Agent is not registered. Call register() first.")
        return self._identity.did

    @property
    def is_registered(self) -> bool:
        """Check if the agent is registered.

        Returns:
            True if registered, False otherwise
        """
        return self._identity is not None

    def get_identity(self, did: str) -> dict[str, Any]:
        """Look up an agent's identity document.

        Rust RPC: GET /identity/{did}
        Rust RPC: GetIdentityResponse { document }

        Args:
            did: The DID to look up

        Returns:
            Response dict with 'document' key containing the identity document

        Raises:
            DIDNotFoundError: DID not found on the network
        """
        return self._request("GET", f"/identity/{did}")

    # -------------------------------------------------------------------------
    # Capability Methods
    # -------------------------------------------------------------------------

    def create_capability(
        self,
        subject: dict[str, Any],
        constraints: list[dict[str, Any]] | None = None,
        delegatable: bool = True,
    ) -> CapabilityHandle:
        """Create a new capability.

        Rust RPC: CreateCapabilityRequest {
            issuer_did, signing_key_hex, subject, constraints, delegatable
        }
        Rust RPC: CreateCapabilityResponse { capability_id_hex, capability }

        Args:
            subject: Capability subject as a JSON-serializable dict
                     (must deserialize to CapabilitySubject in Rust)
            constraints: List of constraint dicts (default empty list)
            delegatable: Whether the capability can be delegated

        Returns:
            CapabilityHandle with capability_id_hex and full capability data

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "issuer_did": self.did,
            "signing_key_hex": self._signing_key_hex,
            "subject": subject,
            "constraints": constraints if constraints is not None else [],
            "delegatable": delegatable,
        }

        response = self._request("POST", "/capability/create", json=payload)
        return CapabilityHandle(**response)

    def delegate(self, capability_id_hex: str, to_did: str) -> dict[str, Any]:
        """Delegate a capability to another agent.

        Rust RPC: DelegateCapabilityRequest {
            capability_id_hex, delegator_did, delegator_sk_hex, delegatee_did
        }
        Rust RPC: DelegateCapabilityResponse { delegation }

        Args:
            capability_id_hex: Hex-encoded capability ID to delegate
            to_did: DID of the agent to delegate to

        Returns:
            Response dict with 'delegation' key containing delegation details

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Delegation failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "capability_id_hex": capability_id_hex,
            "delegator_did": self.did,
            "delegator_sk_hex": self._signing_key_hex,
            "delegatee_did": to_did,
        }

        return self._request("POST", "/capability/delegate", json=payload)

    def invoke(self, capability_id_hex: str) -> bool:
        """Invoke a capability.

        Rust RPC: InvokeCapabilityRequest { capability_id_hex, invoker_did }
        Rust RPC: InvokeCapabilityResponse { success, message }

        Args:
            capability_id_hex: Hex-encoded capability ID to invoke

        Returns:
            True if invocation succeeded

        Raises:
            NotRegisteredError: Agent is not registered
            CapabilityDeniedError: Invocation denied (e.g., coherence too low)
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "capability_id_hex": capability_id_hex,
            "invoker_did": self.did,
        }

        response = self._request("POST", "/capability/invoke", json=payload)
        return response.get("success", False)

    def revoke(self, capability_id_hex: str, scope: str = "single") -> bool:
        """Revoke a capability.

        Rust RPC: RevokeCapabilityRequest {
            capability_id_hex, revoker_did, scope
        }
        Rust RPC: SuccessResponse { success }

        Args:
            capability_id_hex: Hex-encoded capability ID to revoke
            scope: Revocation scope ("single", "subtree", or "all")

        Returns:
            True if revocation succeeded

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "capability_id_hex": capability_id_hex,
            "revoker_did": self.did,
            "scope": scope,
        }

        response = self._request("POST", "/capability/revoke", json=payload)
        return response.get("success", False)

    def get_capability(self, capability_id_hex: str) -> dict[str, Any]:
        """Get a capability by its hex-encoded ID.

        Rust RPC: GET /capability/{cap_id_hex}
        Rust RPC: GetCapabilityResponse { capability }

        Args:
            capability_id_hex: Hex-encoded capability ID

        Returns:
            Response dict with 'capability' key

        Raises:
            DIDNotFoundError: Capability not found
        """
        return self._request("GET", f"/capability/{capability_id_hex}")

    def list_capabilities(self) -> list[dict[str, Any]]:
        """List all capabilities held by this agent.

        Rust RPC: GET /capability/by-did/{did}
        Rust RPC: ListCapabilitiesResponse { capabilities }

        Returns:
            List of capability dictionaries

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        response = self._request("GET", f"/capability/by-did/{self.did}")
        return response.get("capabilities", [])

    # -------------------------------------------------------------------------
    # Social Graph Methods
    # -------------------------------------------------------------------------

    def introduce(self, other_did: str, edge_name: str = "collaborator") -> bool:
        """Introduce this agent to another agent.

        Rust RPC: IntroductionRequest {
            introducer_did, introducer_sk_hex, introduced_did, edge_name
        }
        Rust RPC: SuccessResponse { success }

        Args:
            other_did: DID of the agent to introduce to
            edge_name: Name for the relationship edge

        Returns:
            True if introduction succeeded

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "introducer_did": self.did,
            "introducer_sk_hex": self._signing_key_hex,
            "introduced_did": other_did,
            "edge_name": edge_name,
        }

        response = self._request("POST", "/introduction", json=payload)
        return response.get("success", False)

    def get_introduction_chain(self, to_did: str) -> list[str]:
        """Get the introduction chain from this agent to another.

        Rust RPC: GET /introduction/chain/{from_did}/{to_did}
        Rust RPC: IntroductionChainResponse { chain: Vec<String> }

        Args:
            to_did: Target DID

        Returns:
            List of DID strings forming the chain

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        response = self._request(
            "GET", f"/introduction/chain/{self.did}/{to_did}"
        )
        return response.get("chain", [])

    # -------------------------------------------------------------------------
    # Coherence Methods
    # -------------------------------------------------------------------------

    def coherence(self) -> CoherenceReport:
        """Get this agent's coherence metrics.

        Rust RPC: GET /coherence/{did}
        Rust RPC: CoherenceProfileResponse { profile }

        Returns:
            CoherenceReport with all metrics

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        response = self._request("GET", f"/coherence/{self.did}")
        profile = response.get("profile", response)
        return CoherenceReport(**profile)

    def peer_coherence(self, did: str) -> CoherenceReport:
        """Get another agent's coherence metrics.

        Rust RPC: GET /coherence/{did}
        Rust RPC: CoherenceProfileResponse { profile }

        Args:
            did: DID of the agent to query

        Returns:
            CoherenceReport for the target agent

        Raises:
            DIDNotFoundError: DID not found
        """
        response = self._request("GET", f"/coherence/{did}")
        profile = response.get("profile", response)
        return CoherenceReport(**profile)

    def curvature_with(self, other_did: str) -> float:
        """Calculate curvature between this agent and another.

        Rust RPC: GET /coherence/curvature/{did_a}/{did_b}
        Rust RPC: CurvatureResponse { curvature }

        Args:
            other_did: DID of the other agent

        Returns:
            Curvature score

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        response = self._request("GET", f"/coherence/curvature/{self.did}/{other_did}")
        return response.get("curvature", 0.0)

    def neighbors(self) -> list[str]:
        """Get this agent's neighbors in the social graph.

        Rust RPC: GET /coherence/neighbors/{did}
        Rust RPC: NeighborsResponse { neighbors }

        Returns:
            List of neighbor DIDs

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        response = self._request("GET", f"/coherence/neighbors/{self.did}")
        return response.get("neighbors", [])

    # -------------------------------------------------------------------------
    # Petname Methods
    # -------------------------------------------------------------------------

    def name(self, did: str, petname: str) -> None:
        """Assign a petname to a DID.

        Rust RPC: SetPetnameRequest { name, did }
        Rust RPC: SuccessResponse { success }

        Args:
            did: DID to name
            petname: Human-friendly name

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "name": petname,
            "did": did,
        }

        self._request("POST", "/petname", json=payload)

    def resolve_name(self, petname: str) -> str | None:
        """Resolve a petname to a DID.

        Rust RPC: GET /petname/{name}
        Rust RPC: ResolvePetnameResponse { did }

        Args:
            petname: Petname to resolve

        Returns:
            DID if found, None otherwise

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        try:
            response = self._request("GET", f"/petname/{petname}")
            return response.get("did")
        except DIDNotFoundError:
            return None

    # -------------------------------------------------------------------------
    # Node Info Methods
    # -------------------------------------------------------------------------

    def node_status(self) -> dict[str, Any]:
        """Get node status information.

        Returns:
            Dictionary with node status info
        """
        return self._request("GET", "/status")
