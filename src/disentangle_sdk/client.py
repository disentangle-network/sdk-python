"""DisentangleAgent client for interacting with a Disentangle node."""

import httpx
import json
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

        response = self._request("GET", f"/introduction/chain/{self.did}/{to_did}")
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
    # Gradient (Excitability) Methods
    # -------------------------------------------------------------------------

    def curvature_derivative(
        self, did_a: str, did_b: str, window: int = 100
    ) -> dict[str, Any]:
        """Get the curvature derivative for a specific edge.

        The curvature derivative measures how fast curvature is changing
        on the edge between two agents over a sliding depth window.
        Positive values indicate strengthening coherence; negative
        values indicate decay.

        Rust RPC: GET /coherence/gradient/{did_a}/{did_b}?window={window}
        Rust RPC Response: CurvatureDerivative { ... }

        Args:
            did_a: DID of the first agent
            did_b: DID of the second agent
            window: Depth window for derivative computation (default 100)

        Returns:
            CurvatureDerivative dict with edge gradient data

        Raises:
            DIDNotFoundError: One or both DIDs not found
        """
        return self._request(
            "GET",
            f"/coherence/gradient/{did_a}/{did_b}",
            params={"window": window},
        )

    def excitability(self, did: str, window: int = 100) -> dict[str, Any]:
        """Get the excitability profile for an agent.

        Excitability captures how responsive an agent's local topology
        is to perturbation -- essentially the second derivative of
        coherence. High excitability means small actions produce large
        curvature shifts.

        Rust RPC: GET /coherence/excitability/{did}?window={window}
        Rust RPC Response: ExcitabilityProfile { ... }

        Args:
            did: DID of the agent to query
            window: Depth window for excitability computation (default 100)

        Returns:
            ExcitabilityProfile dict with per-agent excitability data

        Raises:
            DIDNotFoundError: DID not found
        """
        return self._request(
            "GET",
            f"/coherence/excitability/{did}",
            params={"window": window},
        )

    def gradient_map(self, top_n: int = 20, window: int = 100) -> dict[str, Any]:
        """Get the network-level coherence gradient map.

        Returns the top-N edges by absolute curvature derivative,
        giving a snapshot of where coherence is changing fastest
        across the entire network.

        Rust RPC: GET /coherence/gradient/map?top_n={top_n}&window={window}
        Rust RPC Response: CoherenceGradientMap { ... }

        Args:
            top_n: Number of top edges to return (default 20)
            window: Depth window for derivative computation (default 100)

        Returns:
            CoherenceGradientMap dict with ranked edge gradients

        Raises:
            DisentangleError: Query failed
        """
        return self._request(
            "GET",
            "/coherence/gradient/map",
            params={"top_n": top_n, "window": window},
        )

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
    # Service Agreement Methods
    # -------------------------------------------------------------------------

    def propose_agreement(
        self,
        consumer_did: str,
        description: str,
        success_criteria: list[str],
        deadline_depth: int | None = None,
    ) -> dict[str, Any]:
        """Propose a service agreement where this agent is the provider.

        Rust RPC: POST /agreement/propose
        Rust RPC: { provider_did, consumer_did, terms, signing_key_hex }
        Rust RPC Response: { agreement_id, agreement }

        Args:
            consumer_did: DID of the consumer agent
            description: Human-readable description of the service
            success_criteria: List of success criteria strings
            deadline_depth: Optional deadline as DAG depth

        Returns:
            Response dict with 'agreement_id' and 'agreement' keys

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Agreement proposal failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "provider_did": self.did,
            "consumer_did": consumer_did,
            "terms": {
                "description": description,
                "success_criteria": success_criteria,
                "deadline_depth": deadline_depth,
            },
            "signing_key_hex": self._signing_key_hex,
        }

        return self._request("POST", "/agreement/propose", json=payload)

    def accept_agreement(self, agreement_id: str) -> bool:
        """Accept a proposed service agreement where this agent is the consumer.

        Rust RPC: POST /agreement/accept
        Rust RPC: { agreement_id, consumer_sk_hex }
        Rust RPC Response: { success }

        Args:
            agreement_id: Hex-encoded agreement ID

        Returns:
            True if acceptance succeeded

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Agreement acceptance failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "agreement_id": agreement_id,
            "consumer_sk_hex": self._signing_key_hex,
        }

        response = self._request("POST", "/agreement/accept", json=payload)
        return response.get("success", False)

    def complete_agreement(
        self, agreement_id: str, success: bool, outcome_hash: str
    ) -> bool:
        """Mark a service agreement as completed.

        Rust RPC: POST /agreement/complete
        Rust RPC: { agreement_id, success, outcome_hash, signing_key_hex }
        Rust RPC Response: { success }

        Args:
            agreement_id: Hex-encoded agreement ID
            success: Whether the agreement completed successfully
            outcome_hash: Hash of the outcome/deliverable

        Returns:
            True if completion succeeded

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Agreement completion failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "agreement_id": agreement_id,
            "success": success,
            "outcome_hash": outcome_hash,
            "signing_key_hex": self._signing_key_hex,
        }

        response = self._request("POST", "/agreement/complete", json=payload)
        return response.get("success", False)

    def list_agreements(self) -> list[dict[str, Any]]:
        """List all service agreements involving this agent.

        Rust RPC: GET /agreement/by-did/{did}
        Rust RPC Response: { agreements: [...] }

        Returns:
            List of agreement dictionaries

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        response = self._request("GET", f"/agreement/by-did/{self.did}")
        return response.get("agreements", [])

    # -------------------------------------------------------------------------
    # Event Watching Methods
    # -------------------------------------------------------------------------

    def watch(
        self,
        topics: list[str] | None = None,
        callback: Any = None,
    ):
        """Subscribe to Server-Sent Events (SSE) from the node.

        Rust RPC: GET /watch?topics=delegation,agreement,coherence,introduction&did={did}
        Rust RPC: Returns SSE stream with events as JSON

        Args:
            topics: Optional list of event topics to filter:
                   ['delegation', 'agreement', 'coherence', 'introduction', 'transaction']
                   If None, receives all event types.
            callback: Optional function to call for each event.
                     If provided, blocks and calls callback(event_dict).
                     If None, returns an iterator that yields events.

        Returns:
            Iterator[dict] if callback is None
            None if callback is provided (blocks until stream ends)

        Raises:
            NodeConnectionError: Cannot connect to node
            NotRegisteredError: Agent is not registered

        Example:
            # Iterator style
            for event in agent.watch(topics=['delegation']):
                print(f"Event: {event}")

            # Callback style
            def handle_event(event):
                print(f"Received: {event['type']}")
            agent.watch(topics=['agreement'], callback=handle_event)
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        # Build query parameters
        params: dict[str, str] = {"did": self.did}
        if topics:
            params["topics"] = ",".join(topics)

        try:
            # Use httpx.stream for Server-Sent Events
            with self._client.stream("GET", "/watch", params=params) as response:
                response.raise_for_status()

                # Process SSE stream line by line
                for line in response.iter_lines():
                    # SSE format: lines starting with "data: " contain JSON
                    if line.startswith("data: "):
                        event_json = line[6:]  # Strip "data: " prefix
                        try:
                            event = json.loads(event_json)
                            if callback:
                                callback(event)
                            else:
                                yield event
                        except json.JSONDecodeError:
                            # Skip malformed JSON (e.g., keepalive messages)
                            continue

        except httpx.ConnectError as e:
            raise NodeConnectionError(f"Cannot connect to node: {e}")
        except httpx.TimeoutException as e:
            raise NodeConnectionError(f"Request timeout: {e}")
        except httpx.HTTPError as e:
            raise NodeConnectionError(f"HTTP error: {e}")

    # -------------------------------------------------------------------------
    # Node Info Methods
    # -------------------------------------------------------------------------

    def network_health(self) -> dict[str, Any]:
        """Get enriched network health metrics.

        Rust RPC: GET /network/health
        Rust RPC Response: {
            node_id, peer_count, dag_size, current_depth, tips,
            registered_dids, active_capabilities, active_agreements,
            mean_network_curvature, identity_graph_edges, uptime_seconds
        }

        Returns:
            Dictionary with network health metrics

        Example:
            health = agent.network_health()
            print(f"Network has {health['registered_dids']} DIDs")
            print(f"Mean curvature: {health['mean_network_curvature']}")
        """
        return self._request("GET", "/network/health")

    def node_status(self) -> dict[str, Any]:
        """Get basic node status information.

        Returns:
            Dictionary with node status info
        """
        return self._request("GET", "/status")

    # -------------------------------------------------------------------------
    # Proposal Methods
    # -------------------------------------------------------------------------

    def create_proposal(
        self,
        description: str,
        activation_mass: float,
        min_participants: int,
        expiry_depth: int,
    ) -> dict[str, Any]:
        """Create a new coordination proposal.

        A Proposal is a potential SharedIntent waiting for sufficient
        topological mass. Approval is joining; no voting exists.

        Rust RPC: POST /proposal/create
        Rust RPC: { initiator_did, signing_key_hex, description,
                     activation_mass, min_participants, expiry_depth }
        Rust RPC Response: { proposal_id, proposal }

        Args:
            description: Human-readable description of the proposed coordination
            activation_mass: Topological mass threshold for activation
            min_participants: Minimum distinct joiners for diversity requirement
            expiry_depth: DAG depth at which this expires if not activated

        Returns:
            Response dict with 'proposal_id' and 'proposal' keys

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Proposal creation failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "initiator_did": self.did,
            "signing_key_hex": self._signing_key_hex,
            "description": description,
            "activation_mass": activation_mass,
            "min_participants": min_participants,
            "expiry_depth": expiry_depth,
        }

        return self._request("POST", "/proposal/create", json=payload)

    def join_proposal(self, proposal_id: str) -> dict[str, Any]:
        """Join an existing proposal by committing topological mass.

        Rust RPC: POST /proposal/join
        Rust RPC: { proposal_id, joiner_did, signing_key_hex }
        Rust RPC Response: { success, committed_mass, total_mass,
                             activated?, intent_id? }

        Args:
            proposal_id: Hex-encoded proposal ID to join

        Returns:
            Response dict with join result. If the join causes activation,
            the response includes 'activated': True and 'intent_id'.

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Join failed (e.g., proposal expired, coherence too low)
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "proposal_id": proposal_id,
            "joiner_did": self.did,
            "signing_key_hex": self._signing_key_hex,
        }

        return self._request("POST", "/proposal/join", json=payload)

    def list_proposals(self, status: str | None = None) -> list[dict[str, Any]]:
        """List proposals, optionally filtered by status.

        Rust RPC: GET /proposal/list?status={status}
        Rust RPC Response: { proposals: [...] }

        Args:
            status: Optional status filter ('attracting', 'activated',
                    'expired', 'archived'). If None, returns all proposals.

        Returns:
            List of proposal dictionaries

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status

        response = self._request("GET", "/proposal/list", params=params)
        return response.get("proposals", [])

    # -------------------------------------------------------------------------
    # SharedIntent Methods
    # -------------------------------------------------------------------------

    def create_intent(
        self,
        description: str,
        participant_dids: list[str],
        capability_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new SharedIntent for active collaboration.

        A SharedIntent is an active collaboration space with no
        provider/consumer distinction. The topology IS the outcome
        measurement.

        Rust RPC: POST /intent/create
        Rust RPC: { creator_did, signing_key_hex, description,
                     participant_dids, capability_ids }
        Rust RPC Response: { intent_id, intent }

        Args:
            description: Human-readable description of the collaboration
            participant_dids: List of DIDs for initial participants
            capability_ids: Optional list of capability hex IDs to contribute

        Returns:
            Response dict with 'intent_id' and 'intent' keys

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Intent creation failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload: dict[str, Any] = {
            "creator_did": self.did,
            "signing_key_hex": self._signing_key_hex,
            "description": description,
            "participant_dids": participant_dids,
        }

        if capability_ids is not None:
            payload["capability_ids"] = capability_ids

        return self._request("POST", "/intent/create", json=payload)

    def join_intent(
        self,
        intent_id: str,
        capability_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Join an active SharedIntent.

        Late joining is supported. Requires CoherenceMinimum and at
        least one existing participant to have a positive-curvature
        edge with the joiner.

        Rust RPC: POST /intent/join
        Rust RPC: { intent_id, joiner_did, signing_key_hex, capability_ids }
        Rust RPC Response: { success, participant }

        Args:
            intent_id: Hex-encoded SharedIntent ID to join
            capability_ids: Optional list of capability hex IDs to contribute

        Returns:
            Response dict with join result

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Join failed (e.g., coherence too low, no path)
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload: dict[str, Any] = {
            "intent_id": intent_id,
            "joiner_did": self.did,
            "signing_key_hex": self._signing_key_hex,
        }

        if capability_ids is not None:
            payload["capability_ids"] = capability_ids

        return self._request("POST", "/intent/join", json=payload)

    def archive_intent(self, intent_id: str) -> dict[str, Any]:
        """Archive a SharedIntent, snapshotting coherence deltas.

        On archival, the protocol snapshots mass delta and curvature
        delta among participants. These deltas ARE the outcome --
        no attestation needed.

        Rust RPC: POST /intent/archive
        Rust RPC: { intent_id, archiver_did, signing_key_hex }
        Rust RPC Response: { success, mass_delta, curvature_delta }

        Args:
            intent_id: Hex-encoded SharedIntent ID to archive

        Returns:
            Response dict with archive result including coherence deltas

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Archive failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "intent_id": intent_id,
            "archiver_did": self.did,
            "signing_key_hex": self._signing_key_hex,
        }

        return self._request("POST", "/intent/archive", json=payload)

    def intent_coherence(self, intent_id: str) -> dict[str, Any]:
        """Get a coherence snapshot for a SharedIntent.

        Returns the current mass delta, curvature delta, and per-participant
        metrics relative to the intent's baseline.

        Rust RPC: GET /intent/{id}/coherence
        Rust RPC Response: IntentCoherenceSnapshot {
            intent_id, participant_count, baseline_mass, current_mass,
            mass_delta, baseline_curvature, current_curvature,
            curvature_delta, depth
        }

        Args:
            intent_id: Hex-encoded SharedIntent ID

        Returns:
            Dict with coherence snapshot metrics

        Raises:
            NotRegisteredError: Agent is not registered
            DIDNotFoundError: Intent not found
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        return self._request("GET", f"/intent/{intent_id}/coherence")

    def list_intents(self, status: str | None = None) -> list[dict[str, Any]]:
        """List SharedIntents, optionally filtered by status.

        Rust RPC: GET /intent/list?status={status}
        Rust RPC Response: { intents: [...] }

        Args:
            status: Optional status filter ('active', 'archived').
                    If None, returns all intents.

        Returns:
            List of SharedIntent dictionaries

        Raises:
            NotRegisteredError: Agent is not registered
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status

        response = self._request("GET", "/intent/list", params=params)
        return response.get("intents", [])

    # -------------------------------------------------------------------------
    # Oracle Methods
    # -------------------------------------------------------------------------

    def query_oracle(
        self,
        region: dict[str, Any],
        depth_start: int,
        depth_end: int,
    ) -> dict[str, Any]:
        """Query the CoherenceOracle for a distribution computation.

        The oracle exposes the protocol's coherence computation as a
        deterministic, auditable external API. Anyone can recompute
        from DAG state.

        Rust RPC: POST /oracle/query
        Rust RPC: OracleQuery { region, depth_start, depth_end }
        Rust RPC Response: DistributionRoot {
            query_id, region, depth_window, weights, scores,
            merkle_root, computed_at_depth
        }

        Args:
            region: Region selector dict. Supported forms:
                    {"neighborhood": "<hash>"}
                    {"intent": "<intent_id_hex>"}
                    {"explicit": ["did:...", "did:..."]}
                    {"global": true}
            depth_start: Start of the depth window for evaluation
            depth_end: End of the depth window for evaluation

        Returns:
            DistributionRoot dict with per-agent weights and scores

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Oracle query failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "region": region,
            "depth_start": depth_start,
            "depth_end": depth_end,
        }

        return self._request("POST", "/oracle/query", json=payload)

    def get_distribution(self, distribution_id: str) -> dict[str, Any]:
        """Retrieve a previously computed distribution.

        Rust RPC: GET /oracle/distribution/{id}
        Rust RPC Response: DistributionRoot { ... }

        Args:
            distribution_id: Hex-encoded distribution/query ID

        Returns:
            DistributionRoot dict

        Raises:
            DIDNotFoundError: Distribution not found
        """
        return self._request("GET", f"/oracle/distribution/{distribution_id}")

    def list_distributions(self) -> list[dict[str, Any]]:
        """List all previously computed distributions.

        Rust RPC: GET /oracle/distributions
        Rust RPC Response: { distributions: [...] }

        Returns:
            List of DistributionRoot dictionaries
        """
        response = self._request("GET", "/oracle/distributions")
        return response.get("distributions", [])

    # -------------------------------------------------------------------------
    # Pool Methods (Demo)
    # -------------------------------------------------------------------------

    def pool_status(self, pool_id: str) -> dict[str, Any]:
        """Get the status of a CommonsPool.

        Rust RPC: GET /pool/{id}
        Rust RPC Response: CommonsPool { id, name, balance, min_coherence,
                           active_distribution, deposits, claims }

        Args:
            pool_id: Hex-encoded pool ID

        Returns:
            Pool state dictionary

        Raises:
            DIDNotFoundError: Pool not found
        """
        return self._request("GET", f"/pool/{pool_id}")

    def pool_claim(self, pool_id: str, distribution_id: str) -> dict[str, Any]:
        """Claim an allocation from a CommonsPool.

        Requires a valid distribution that includes this agent and
        CoherenceMinimum met.

        Rust RPC: POST /pool/claim
        Rust RPC: { pool_id, distribution_id, claimant_did, signing_key_hex }
        Rust RPC Response: { success, amount, claim }

        Args:
            pool_id: Hex-encoded pool ID
            distribution_id: Hex-encoded distribution ID (from oracle query)

        Returns:
            Response dict with claim result including amount

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Claim failed (e.g., not in distribution,
                              coherence too low)
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "pool_id": pool_id,
            "distribution_id": distribution_id,
            "claimant_did": self.did,
            "signing_key_hex": self._signing_key_hex,
        }

        return self._request("POST", "/pool/claim", json=payload)

    def create_pool(self, name: str, description: str = "") -> dict[str, Any]:
        """Create a new CommonsPool.

        A CommonsPool holds fungible value and distributes it to agents
        based on oracle-computed coherence weights.

        Rust RPC: POST /pool/create
        Rust RPC: { name, description, creator_did, signing_key_hex }
        Rust RPC Response: CommonsPool { id, name, balance, ... }

        Args:
            name: Human-readable pool name
            description: Optional description of the pool's purpose

        Returns:
            CommonsPool dict with pool metadata

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Pool creation failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "name": name,
            "description": description,
            "creator_did": self.did,
            "signing_key_hex": self._signing_key_hex,
        }

        return self._request("POST", "/pool/create", json=payload)

    def pool_deposit(
        self, pool_id: str, amount: float, source: str = ""
    ) -> dict[str, Any]:
        """Deposit value into a CommonsPool.

        Rust RPC: POST /pool/deposit
        Rust RPC: { pool_id, amount, source, depositor_did, signing_key_hex }
        Rust RPC Response: { success, new_balance, deposit }

        Args:
            pool_id: Hex-encoded pool ID
            amount: Amount to deposit
            source: Optional label for the deposit source

        Returns:
            Response dict with deposit confirmation and new balance

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Deposit failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "pool_id": pool_id,
            "amount": amount,
            "source": source,
            "depositor_did": self.did,
            "signing_key_hex": self._signing_key_hex,
        }

        return self._request("POST", "/pool/deposit", json=payload)

    def pool_distribute(self, pool_id: str, distribution_id: str) -> dict[str, Any]:
        """Trigger distribution from a pool using oracle results.

        Applies a previously computed DistributionRoot to allocate
        pool funds to participants proportional to their coherence
        weights.

        Rust RPC: POST /pool/distribute
        Rust RPC: { pool_id, distribution_id, initiator_did, signing_key_hex }
        Rust RPC Response: { success, allocations, remaining_balance }

        Args:
            pool_id: Hex-encoded pool ID
            distribution_id: Hex-encoded distribution ID from oracle query

        Returns:
            Response dict with distribution result and allocations

        Raises:
            NotRegisteredError: Agent is not registered
            DisentangleError: Distribution failed
        """
        if not self.is_registered:
            raise NotRegisteredError("Agent is not registered. Call register() first.")

        payload = {
            "pool_id": pool_id,
            "distribution_id": distribution_id,
            "initiator_did": self.did,
            "signing_key_hex": self._signing_key_hex,
        }

        return self._request("POST", "/pool/distribute", json=payload)

    def pool_claims(self, pool_id: str) -> list[dict[str, Any]]:
        """List all claims for a CommonsPool.

        Rust RPC: GET /pool/{id}/claims
        Rust RPC Response: { claims: [...] }

        Args:
            pool_id: Hex-encoded pool ID

        Returns:
            List of claim dictionaries

        Raises:
            DIDNotFoundError: Pool not found
        """
        response = self._request("GET", f"/pool/{pool_id}/claims")
        return response.get("claims", [])

    # -------------------------------------------------------------------------
    # Topology Methods
    # -------------------------------------------------------------------------

    def neighborhoods(self) -> list[dict[str, Any]]:
        """List current neighborhoods with mass and curvature summaries.

        Neighborhoods are connected components of the identity graph
        where all edges have weight >= W_MIN. They are ephemeral
        computed views, not stored entities.

        Rust RPC: GET /topology/neighborhoods
        Rust RPC Response: { neighborhoods: [...] }

        Returns:
            List of neighborhood summary dictionaries, each containing
            cluster hash, member count, total mass, mean curvature, etc.
        """
        response = self._request("GET", "/topology/neighborhoods")
        return response.get("neighborhoods", [])
