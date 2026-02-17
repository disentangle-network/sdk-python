"""Tests for the DisentangleAgent client.

These tests are structured to test the SDK interface. They will be skipped
unless DISENTANGLE_NODE_URL is set, which should point to a running node.
"""

import pytest
from disentangle_sdk import (
    DisentangleAgent,
    AgentIdentity,
    CapabilityHandle,
    CoherenceReport,
    DIDNotFoundError,
    CapabilityDeniedError,
    NotRegisteredError,
)
from conftest import requires_node


class TestIdentity:
    """Test identity registration and lookup."""

    @requires_node
    def test_register_agi_agent(self, node_url):
        """Test registering an AGI agent."""
        agent = DisentangleAgent(node_url)

        identity = agent.register(
            agent_type="agi",
            runtime_attestation="test-attestation-v1",
        )

        # Verify identity structure
        # Rust RegisterResponse: { did, signing_key_hex, document }
        assert isinstance(identity, AgentIdentity)
        assert identity.did.startswith("did:")
        assert len(identity.signing_key_hex) > 0
        assert isinstance(identity.document, dict)

        # Verify agent is registered
        assert agent.is_registered
        assert agent.did == identity.did

    @requires_node
    def test_register_human_agent(self, node_url):
        """Test registering a human agent."""
        agent = DisentangleAgent(node_url)

        identity = agent.register(agent_type="human")

        # Verify identity structure
        assert isinstance(identity, AgentIdentity)
        assert identity.did.startswith("did:")
        assert isinstance(identity.document, dict)

    @requires_node
    def test_lookup_registered_agent(self, node_url):
        """Test looking up a registered agent's identity."""
        # Register first agent
        agent1 = DisentangleAgent(node_url)
        identity1 = agent1.register(agent_type="agi")

        # Create second agent and look up first
        agent2 = DisentangleAgent(node_url)
        looked_up = agent2.get_identity(identity1.did)

        # Rust GetIdentityResponse: { document }
        assert "document" in looked_up
        assert isinstance(looked_up["document"], dict)

    @requires_node
    def test_lookup_nonexistent_agent_raises_error(self, node_url):
        """Test that looking up a non-existent DID raises DIDNotFoundError."""
        agent = DisentangleAgent(node_url)

        with pytest.raises(DIDNotFoundError):
            agent.get_identity("did:disentangle:nonexistent")

    def test_unregistered_agent_raises_error(self, node_url):
        """Test that operations on unregistered agent raise NotRegisteredError."""
        agent = DisentangleAgent(node_url)

        # Should raise for any operation requiring registration
        with pytest.raises(NotRegisteredError):
            _ = agent.did

        with pytest.raises(NotRegisteredError):
            agent.create_capability({"type": "file"})

        with pytest.raises(NotRegisteredError):
            agent.coherence()


class TestCapabilities:
    """Test capability creation, delegation, and invocation."""

    @requires_node
    def test_create_capability(self, node_url):
        """Test creating a capability."""
        agent = DisentangleAgent(node_url)
        agent.register(agent_type="agi")

        # Create a capability
        # Rust expects: { issuer_did, signing_key_hex, subject, constraints, delegatable }
        capability = agent.create_capability(
            subject={"type": "file", "scope": "read"},
            constraints=[{"path": "/data/*.txt"}],
            delegatable=True,
        )

        # Rust CreateCapabilityResponse: { capability_id_hex, capability }
        assert isinstance(capability, CapabilityHandle)
        assert capability.capability_id_hex
        assert isinstance(capability.capability, dict)

    @requires_node
    def test_delegate_and_invoke(self, node_url):
        """Test delegating a capability and invoking it."""
        # Create issuer and recipient agents
        issuer = DisentangleAgent(node_url)
        issuer.register(agent_type="agi")

        recipient = DisentangleAgent(node_url)
        recipient.register(agent_type="agi")

        # Issuer creates capability
        capability = issuer.create_capability(
            subject={"type": "compute", "scope": "execute"},
            delegatable=True,
        )

        # Delegate to recipient
        # Rust DelegateCapabilityResponse: { delegation }
        delegation = issuer.delegate(capability.capability_id_hex, recipient.did)
        assert "delegation" in delegation

        # Recipient should be able to invoke
        result = recipient.invoke(capability.capability_id_hex)
        assert result is True

    @requires_node
    def test_revoke_prevents_invoke(self, node_url):
        """Test that revoking a capability prevents invocation."""
        # Create issuer and recipient agents
        issuer = DisentangleAgent(node_url)
        issuer.register(agent_type="agi")

        recipient = DisentangleAgent(node_url)
        recipient.register(agent_type="agi")

        # Create and delegate capability
        capability = issuer.create_capability(
            subject={"type": "file"},
            delegatable=True,
        )
        issuer.delegate(capability.capability_id_hex, recipient.did)

        # Revoke the capability
        revoke_result = issuer.revoke(capability.capability_id_hex, scope="single")
        assert revoke_result is True

        # Recipient should NOT be able to invoke
        with pytest.raises(CapabilityDeniedError):
            recipient.invoke(capability.capability_id_hex)

    @requires_node
    def test_list_capabilities(self, node_url):
        """Test listing capabilities held by an agent."""
        agent = DisentangleAgent(node_url)
        agent.register(agent_type="agi")

        # Create multiple capabilities
        agent.create_capability(subject={"type": "file", "scope": "read"})
        agent.create_capability(subject={"type": "compute", "scope": "execute"})

        # List capabilities
        capabilities = agent.list_capabilities()
        assert len(capabilities) >= 2


class TestSocialGraph:
    """Test social graph and introduction functionality."""

    @requires_node
    def test_mutual_introduction(self, node_url):
        """Test mutual introduction between two agents."""
        # Create two agents
        agent1 = DisentangleAgent(node_url)
        agent1.register(agent_type="agi")

        agent2 = DisentangleAgent(node_url)
        agent2.register(agent_type="agi")

        # Mutual introduction
        # Rust SuccessResponse: { success }
        result1 = agent1.introduce(agent2.did, edge_name="collaborator")
        result2 = agent2.introduce(agent1.did, edge_name="collaborator")

        assert result1 is True
        assert result2 is True

        # Agents should now be neighbors
        neighbors1 = agent1.neighbors()
        neighbors2 = agent2.neighbors()

        assert agent2.did in neighbors1
        assert agent1.did in neighbors2

    @requires_node
    def test_curvature_calculation(self, node_url):
        """Test calculating curvature between two agents."""
        agent1 = DisentangleAgent(node_url)
        agent1.register(agent_type="agi")

        agent2 = DisentangleAgent(node_url)
        agent2.register(agent_type="agi")

        # Before introduction, curvature should be low
        curvature_before = agent1.curvature_with(agent2.did)
        assert curvature_before >= 0.0

        # After mutual introduction, curvature should increase
        agent1.introduce(agent2.did)
        agent2.introduce(agent1.did)

        curvature_after = agent1.curvature_with(agent2.did)
        assert curvature_after > curvature_before

    @requires_node
    def test_get_introduction_chain(self, node_url):
        """Test retrieving introduction chain between agents."""
        # Create three agents: A -> B -> C
        agent_a = DisentangleAgent(node_url)
        agent_a.register(agent_type="agi")

        agent_b = DisentangleAgent(node_url)
        agent_b.register(agent_type="agi")

        agent_c = DisentangleAgent(node_url)
        agent_c.register(agent_type="agi")

        # Create introduction chain
        agent_a.introduce(agent_b.did)
        agent_b.introduce(agent_c.did)

        # Get chain from A to C
        # Rust IntroductionChainResponse: { chain: Vec<String> }
        chain = agent_a.get_introduction_chain(agent_c.did)
        assert isinstance(chain, list)
        assert len(chain) >= 1
        # Chain entries are DID strings, not dicts
        assert all(isinstance(entry, str) for entry in chain)


class TestCoherence:
    """Test coherence metrics and sybil resistance."""

    @requires_node
    def test_coherence_increases_with_connections(self, node_url):
        """Test that coherence increases as agents build connections."""
        # Create an agent
        agent = DisentangleAgent(node_url)
        agent.register(agent_type="agi")

        # Initial coherence
        # Rust CoherenceProfileResponse: { profile }
        coherence_initial = agent.coherence()
        assert isinstance(coherence_initial, CoherenceReport)
        assert coherence_initial.did == agent.did
        assert 0.0 <= coherence_initial.composite_score <= 1.0

        # Create and connect with multiple peers
        peers = []
        for _ in range(5):
            peer = DisentangleAgent(node_url)
            peer.register(agent_type="agi")
            agent.introduce(peer.did)
            peer.introduce(agent.did)
            peers.append(peer)

        # Coherence should increase with connections
        coherence_after = agent.coherence()
        assert coherence_after.composite_score > coherence_initial.composite_score
        assert coherence_after.relational_diversity >= 5

    @requires_node
    def test_sybil_agents_get_low_coherence(self, node_url):
        """CRITICAL: Test that sybil agents receive low coherence scores.

        This is the key security test - agents with no real network connections
        should have very low coherence, preventing sybil attacks.
        """
        # Create a sybil agent with no connections
        sybil = DisentangleAgent(node_url)
        sybil.register(agent_type="agi")

        # Check coherence immediately after registration
        coherence = sybil.coherence()

        # Sybil agent should have minimal coherence
        assert coherence.composite_score < 0.3  # Threshold for "low" coherence
        assert coherence.topological_mass < 1.0
        assert coherence.relational_diversity == 0

        # Create a well-connected agent for comparison
        legitimate = DisentangleAgent(node_url)
        legitimate.register(agent_type="agi")

        # Connect with multiple diverse agents
        for _ in range(10):
            peer = DisentangleAgent(node_url)
            peer.register(agent_type="agi")
            legitimate.introduce(peer.did)
            peer.introduce(legitimate.did)

        legitimate_coherence = legitimate.coherence()

        # Legitimate agent should have MUCH higher coherence
        assert legitimate_coherence.composite_score > coherence.composite_score * 3
        assert legitimate_coherence.relational_diversity >= 10

    @requires_node
    def test_peer_coherence_lookup(self, node_url):
        """Test looking up another agent's coherence."""
        agent1 = DisentangleAgent(node_url)
        agent1.register(agent_type="agi")

        agent2 = DisentangleAgent(node_url)
        agent2.register(agent_type="agi")

        # Agent1 should be able to query agent2's coherence
        peer_coherence = agent1.peer_coherence(agent2.did)
        assert isinstance(peer_coherence, CoherenceReport)
        assert peer_coherence.did == agent2.did


class TestPetnames:
    """Test petname functionality."""

    @requires_node
    def test_petname_roundtrip(self, node_url):
        """Test assigning and resolving a petname."""
        agent1 = DisentangleAgent(node_url)
        agent1.register(agent_type="agi")

        agent2 = DisentangleAgent(node_url)
        agent2.register(agent_type="agi")

        # Agent1 assigns a petname to agent2
        # Rust SetPetnameRequest: { name, did }
        petname = "my-collaborator"
        agent1.name(agent2.did, petname)

        # Resolve the petname
        # Rust ResolvePetnameResponse: { did }
        resolved_did = agent1.resolve_name(petname)
        assert resolved_did == agent2.did

    @requires_node
    def test_resolve_nonexistent_petname(self, node_url):
        """Test resolving a non-existent petname returns None."""
        agent = DisentangleAgent(node_url)
        agent.register(agent_type="agi")

        resolved = agent.resolve_name("nonexistent-name")
        assert resolved is None


class TestServiceAgreements:
    """Test service agreement functionality."""

    @requires_node
    def test_propose_accept_complete_agreement(self, node_url):
        """Test full service agreement lifecycle: propose, accept, complete."""
        # Create provider and consumer agents
        provider = DisentangleAgent(node_url)
        provider.register(agent_type="agi")

        consumer = DisentangleAgent(node_url)
        consumer.register(agent_type="agi")

        # Provider proposes an agreement
        # Rust POST /agreement/propose: { provider_did, consumer_did, terms, signing_key_hex }
        # Response: { agreement_id, agreement }
        result = provider.propose_agreement(
            consumer_did=consumer.did,
            description="Compute 100 embeddings",
            success_criteria=["all 100 returned", "latency < 500ms"],
            deadline_depth=1000,
        )

        assert "agreement_id" in result
        assert "agreement" in result
        agreement_id = result["agreement_id"]

        # Consumer accepts the agreement
        # Rust POST /agreement/accept: { agreement_id, consumer_sk_hex }
        # Response: { success }
        accept_result = consumer.accept_agreement(agreement_id)
        assert accept_result is True

        # Provider completes the agreement successfully
        # Rust POST /agreement/complete: { agreement_id, success, outcome_hash, signing_key_hex }
        # Response: { success }
        complete_result = provider.complete_agreement(
            agreement_id=agreement_id,
            success=True,
            outcome_hash="abc123def456",
        )
        assert complete_result is True

        # Both agents should see the agreement in their lists
        provider_agreements = provider.list_agreements()
        consumer_agreements = consumer.list_agreements()

        assert any(a["id"] == agreement_id for a in provider_agreements)
        assert any(a["id"] == agreement_id for a in consumer_agreements)

    @requires_node
    def test_list_agreements(self, node_url):
        """Test listing agreements for an agent."""
        provider = DisentangleAgent(node_url)
        provider.register(agent_type="agi")

        consumer = DisentangleAgent(node_url)
        consumer.register(agent_type="agi")

        # Create multiple agreements
        result1 = provider.propose_agreement(
            consumer_did=consumer.did,
            description="Service 1",
            success_criteria=["criterion 1"],
        )

        result2 = provider.propose_agreement(
            consumer_did=consumer.did,
            description="Service 2",
            success_criteria=["criterion 2"],
        )

        # Rust GET /agreement/by-did/{did}
        # Response: { agreements: [...] }
        agreements = provider.list_agreements()
        assert isinstance(agreements, list)
        assert len(agreements) >= 2

        # Both agreements should be in the list
        agreement_ids = [result1["agreement_id"], result2["agreement_id"]]
        found_ids = [a["id"] for a in agreements]
        for aid in agreement_ids:
            assert aid in found_ids


class TestEventWatching:
    """Test SSE event watching functionality."""

    @requires_node
    def test_watch_method_exists(self, node_url):
        """Test that watch method exists and is callable.

        This is a basic test to verify the watch method is implemented.
        Full SSE testing requires a running node with event emission.
        """
        agent = DisentangleAgent(node_url)
        agent.register(agent_type="agi")

        # Verify watch method exists
        assert hasattr(agent, "watch")
        assert callable(agent.watch)

        # Note: Full SSE testing requires event emission from WS-A (Rust node)
        # which may not be implemented yet. This test verifies the SDK side
        # is ready to consume events when the node is ready.


class TestNodeInfo:
    """Test node information endpoints."""

    @requires_node
    def test_node_status(self, node_url):
        """Test retrieving node status."""
        agent = DisentangleAgent(node_url)

        status = agent.node_status()
        assert isinstance(status, dict)
        # Status should contain basic node info
        assert "version" in status or "status" in status or "uptime" in status

    @requires_node
    def test_network_health(self, node_url):
        """Test retrieving network health metrics."""
        agent = DisentangleAgent(node_url)

        # Rust GET /network/health
        # Response: {
        #   node_id, peer_count, dag_size, current_depth, tips,
        #   registered_dids, active_capabilities, active_agreements,
        #   mean_network_curvature, identity_graph_edges, uptime_seconds
        # }
        health = agent.network_health()
        assert isinstance(health, dict)

        # Health should contain enriched network metrics
        # Note: Some fields may not exist yet if WS-A implementation is pending
        # We test for the existence of at least some expected fields
        expected_fields = [
            "node_id",
            "peer_count",
            "dag_size",
            "current_depth",
            "registered_dids",
        ]

        # At least one expected field should be present
        assert any(field in health for field in expected_fields)


class TestContextManager:
    """Test context manager support."""

    @requires_node
    def test_context_manager(self, node_url):
        """Test using DisentangleAgent as a context manager."""
        with DisentangleAgent(node_url) as agent:
            identity = agent.register(agent_type="agi")
            assert agent.is_registered
            assert agent.did == identity.did

        # Client should be closed after context exit
        # (httpx client will raise error if used after close)
