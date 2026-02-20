"""Mock-based tests for DisentangleAgent client."""

from unittest.mock import MagicMock
import httpx
import pytest

from disentangle_sdk import DisentangleAgent
from disentangle_sdk.exceptions import (
    DIDNotFoundError,
    CapabilityDeniedError,
    DisentangleError,
    NodeConnectionError,
    NotRegisteredError,
)
from disentangle_sdk.types import AgentIdentity, CapabilityHandle, CoherenceReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_DID = "did:disentangle:abc123"
MOCK_SK_HEX = "deadbeef" * 8
MOCK_DOCUMENT = {"id": MOCK_DID, "type": "agi"}

MOCK_REGISTER_RESPONSE = {
    "did": MOCK_DID,
    "signing_key_hex": MOCK_SK_HEX,
    "document": MOCK_DOCUMENT,
}

MOCK_COHERENCE_PROFILE = {
    "did": MOCK_DID,
    "topological_mass": 1.5,
    "mean_local_curvature": 0.3,
    "relational_diversity": 4,
    "temporal_depth": 10,
    "composite_score": 0.72,
    "decayed_mass": 1.2,
}


def _mock_response(status_code=200, json_data=None):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = json_data or {}
    return resp


def _registered_agent():
    """Return an agent with mocked registration."""
    agent = DisentangleAgent("http://localhost:8000")
    agent._identity = AgentIdentity(**MOCK_REGISTER_RESPONSE)
    agent._signing_key_hex = MOCK_SK_HEX
    return agent


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_success(self):
        agent = DisentangleAgent("http://localhost:8000")
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, MOCK_REGISTER_RESPONSE)

        identity = agent.register(agent_type="agi")

        assert identity.did == MOCK_DID
        assert identity.signing_key_hex == MOCK_SK_HEX
        assert agent.is_registered
        assert agent.did == MOCK_DID

    def test_register_with_attestation(self):
        agent = DisentangleAgent("http://localhost:8000")
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, MOCK_REGISTER_RESPONSE)

        agent.register(agent_type="agi", runtime_attestation="sha256:abc")

        call_args = agent._client.request.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["runtime_attestation"] == "sha256:abc"

    def test_not_registered_raises(self):
        agent = DisentangleAgent("http://localhost:8000")
        with pytest.raises(NotRegisteredError):
            _ = agent.did


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_404_raises_did_not_found(self):
        agent = DisentangleAgent("http://localhost:8000")
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            404, {"error": "DID not found"}
        )

        with pytest.raises(DIDNotFoundError, match="DID not found"):
            agent.get_identity("did:disentangle:nonexistent")

    def test_403_raises_capability_denied(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            403, {"error": "Coherence too low", "coherence_score": 0.1}
        )

        with pytest.raises(CapabilityDeniedError) as exc_info:
            agent.invoke("cafebabe" * 8)
        assert exc_info.value.coherence_score == 0.1

    def test_400_raises_disentangle_error(self):
        agent = DisentangleAgent("http://localhost:8000")
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            400, {"error": "Bad request"}
        )

        with pytest.raises(DisentangleError, match="Bad request"):
            agent.node_status()

    def test_connect_error_raises_node_connection(self):
        agent = DisentangleAgent("http://localhost:8000")
        agent._client = MagicMock()
        agent._client.request.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(NodeConnectionError):
            agent.node_status()

    def test_timeout_raises_node_connection(self):
        agent = DisentangleAgent("http://localhost:8000")
        agent._client = MagicMock()
        agent._client.request.side_effect = httpx.ReadTimeout("Timeout")

        with pytest.raises(NodeConnectionError):
            agent.node_status()


# ---------------------------------------------------------------------------
# Coherence methods
# ---------------------------------------------------------------------------


class TestCoherence:
    def test_coherence_success(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            200, {"profile": MOCK_COHERENCE_PROFILE}
        )

        report = agent.coherence()

        assert isinstance(report, CoherenceReport)
        assert report.did == MOCK_DID
        assert report.composite_score == 0.72
        assert report.relational_diversity == 4

    def test_peer_coherence(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        other_did = "did:disentangle:other"
        other_profile = {**MOCK_COHERENCE_PROFILE, "did": other_did}
        agent._client.request.return_value = _mock_response(
            200, {"profile": other_profile}
        )

        report = agent.peer_coherence(other_did)
        assert report.did == other_did

    def test_curvature_with(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, {"curvature": 0.42})

        result = agent.curvature_with("did:disentangle:other")
        assert result == 0.42

    def test_neighbors(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        neighbor_dids = ["did:disentangle:n1", "did:disentangle:n2"]
        agent._client.request.return_value = _mock_response(
            200, {"neighbors": neighbor_dids}
        )

        result = agent.neighbors()
        assert result == neighbor_dids

    def test_coherence_not_registered(self):
        agent = DisentangleAgent("http://localhost:8000")
        with pytest.raises(NotRegisteredError):
            agent.coherence()


# ---------------------------------------------------------------------------
# Excitability / Gradient methods
# ---------------------------------------------------------------------------


class TestGradient:
    def test_curvature_derivative(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"did_a": MOCK_DID, "did_b": "did:other", "derivative": 0.05}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.curvature_derivative(MOCK_DID, "did:other", window=50)
        assert result["derivative"] == 0.05

    def test_excitability(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"did": MOCK_DID, "excitability": 0.8, "edge_count": 3}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.excitability(MOCK_DID, window=200)
        assert result["excitability"] == 0.8

    def test_gradient_map(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"edges": [{"a": "d1", "b": "d2", "derivative": 0.1}], "count": 1}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.gradient_map(top_n=10, window=50)
        assert len(result["edges"]) == 1


# ---------------------------------------------------------------------------
# Oracle methods
# ---------------------------------------------------------------------------


class TestOracle:
    def test_query_oracle(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {
            "query_id": "abc123",
            "region": {"global": True},
            "weights": {"did:a": 0.5, "did:b": 0.5},
            "merkle_root": "deadbeef",
        }
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.query_oracle(
            region={"global": True}, depth_start=0, depth_end=100
        )
        assert result["query_id"] == "abc123"
        assert "weights" in result

    def test_get_distribution(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"query_id": "abc123", "weights": {}}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.get_distribution("abc123")
        assert result["query_id"] == "abc123"

    def test_list_distributions(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            200, {"distributions": [{"id": "a"}]}
        )

        result = agent.list_distributions()
        assert len(result) == 1

    def test_query_oracle_not_registered(self):
        agent = DisentangleAgent("http://localhost:8000")
        with pytest.raises(NotRegisteredError):
            agent.query_oracle(region={"global": True}, depth_start=0, depth_end=100)


# ---------------------------------------------------------------------------
# Pool methods
# ---------------------------------------------------------------------------


class TestPool:
    def test_create_pool(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"id": "pool123", "name": "Test Pool", "balance": 0.0}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.create_pool("Test Pool", description="A test pool")
        assert result["name"] == "Test Pool"

    def test_pool_deposit(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"success": True, "new_balance": 100.0}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.pool_deposit("pool123", amount=100.0, source="funding")
        assert result["success"] is True
        assert result["new_balance"] == 100.0

    def test_pool_claim(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"success": True, "amount": 25.0}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.pool_claim("pool123", "dist456")
        assert result["amount"] == 25.0

    def test_pool_status(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"id": "pool123", "balance": 500.0, "min_coherence": 0.3}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.pool_status("pool123")
        assert result["balance"] == 500.0

    def test_pool_distribute(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {
            "success": True,
            "allocations": {"did:a": 50.0},
            "remaining_balance": 450.0,
        }
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.pool_distribute("pool123", "dist456")
        assert result["success"] is True

    def test_pool_claims_list(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            200, {"claims": [{"amount": 10}]}
        )

        result = agent.pool_claims("pool123")
        assert len(result) == 1

    def test_pool_not_registered(self):
        agent = DisentangleAgent("http://localhost:8000")
        with pytest.raises(NotRegisteredError):
            agent.create_pool("test")


# ---------------------------------------------------------------------------
# Capability methods
# ---------------------------------------------------------------------------


class TestCapability:
    def test_create_capability(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {
            "capability_id_hex": "cap123",
            "capability": {"subject": {"type": "file"}, "delegatable": True},
        }
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.create_capability(
            subject={"type": "file", "scope": "read"}, delegatable=True
        )
        assert isinstance(result, CapabilityHandle)
        assert result.capability_id_hex == "cap123"

    def test_invoke_success(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, {"success": True})

        result = agent.invoke("cap123")
        assert result is True

    def test_revoke(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, {"success": True})

        result = agent.revoke("cap123", scope="subtree")
        assert result is True

    def test_delegate(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"delegation": {"from": MOCK_DID, "to": "did:other"}}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.delegate("cap123", "did:other")
        assert "delegation" in result

    def test_list_capabilities(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            200, {"capabilities": [{"id": "c1"}, {"id": "c2"}]}
        )

        result = agent.list_capabilities()
        assert len(result) == 2

    def test_get_capability(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"capability": {"id": "cap123", "subject": {}}}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.get_capability("cap123")
        assert "capability" in result


# ---------------------------------------------------------------------------
# Social graph methods
# ---------------------------------------------------------------------------


class TestSocialGraph:
    def test_introduce(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, {"success": True})

        result = agent.introduce("did:other", edge_name="collaborator")
        assert result is True

    def test_get_introduction_chain(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        chain = [MOCK_DID, "did:mid", "did:target"]
        agent._client.request.return_value = _mock_response(200, {"chain": chain})

        result = agent.get_introduction_chain("did:target")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Petname methods
# ---------------------------------------------------------------------------


class TestPetname:
    def test_name(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, {"success": True})

        agent.name("did:other", "alice")
        agent._client.request.assert_called_once()

    def test_resolve_name_found(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, {"did": "did:other"})

        result = agent.resolve_name("alice")
        assert result == "did:other"

    def test_resolve_name_not_found(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(404, {"error": "Not found"})

        result = agent.resolve_name("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager(self):
        with DisentangleAgent("http://localhost:8000") as agent:
            assert agent is not None
        # After exit, client should be closed (no error)


# ---------------------------------------------------------------------------
# Agreement methods
# ---------------------------------------------------------------------------


class TestAgreement:
    def test_propose_agreement(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"agreement_id": "agr123", "agreement": {"status": "proposed"}}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.propose_agreement(
            consumer_did="did:consumer",
            description="Test service",
            success_criteria=["criterion1"],
            deadline_depth=1000,
        )
        assert result["agreement_id"] == "agr123"

    def test_accept_agreement(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, {"success": True})

        result = agent.accept_agreement("agr123")
        assert result is True

    def test_complete_agreement(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(200, {"success": True})

        result = agent.complete_agreement(
            "agr123", success=True, outcome_hash="sha256:xyz"
        )
        assert result is True

    def test_list_agreements(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            200, {"agreements": [{"id": "a1"}]}
        )

        result = agent.list_agreements()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Proposal methods
# ---------------------------------------------------------------------------


class TestProposal:
    def test_create_proposal(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"proposal_id": "prop123", "proposal": {"status": "attracting"}}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.create_proposal(
            description="Test proposal",
            activation_mass=10.0,
            min_participants=3,
            expiry_depth=500,
        )
        assert result["proposal_id"] == "prop123"

    def test_join_proposal(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"success": True, "committed_mass": 1.5, "total_mass": 5.0}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.join_proposal("prop123")
        assert result["committed_mass"] == 1.5

    def test_list_proposals(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            200, {"proposals": [{"id": "p1"}]}
        )

        result = agent.list_proposals(status="attracting")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Intent methods
# ---------------------------------------------------------------------------


class TestIntent:
    def test_create_intent(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"intent_id": "int123", "intent": {"status": "active"}}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.create_intent(
            description="Collaborate on X",
            participant_dids=["did:a", "did:b"],
            capability_ids=["cap1"],
        )
        assert result["intent_id"] == "int123"

    def test_join_intent(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"success": True, "participant": {"did": MOCK_DID}}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.join_intent("int123")
        assert result["success"] is True

    def test_archive_intent(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"success": True, "mass_delta": 0.5, "curvature_delta": 0.1}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.archive_intent("int123")
        assert result["mass_delta"] == 0.5

    def test_intent_coherence(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {
            "intent_id": "int123",
            "participant_count": 3,
            "mass_delta": 2.0,
            "curvature_delta": 0.5,
        }
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.intent_coherence("int123")
        assert result["participant_count"] == 3

    def test_list_intents(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        agent._client.request.return_value = _mock_response(
            200, {"intents": [{"id": "i1"}]}
        )

        result = agent.list_intents(status="active")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Topology methods
# ---------------------------------------------------------------------------


class TestTopology:
    def test_neighborhoods(self):
        agent = _registered_agent()
        agent._client = MagicMock()
        mock_data = {"neighborhoods": [{"hash": "n1", "members": 5, "mass": 12.0}]}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.neighborhoods()
        assert len(result) == 1
        assert result[0]["mass"] == 12.0


# ---------------------------------------------------------------------------
# Node methods
# ---------------------------------------------------------------------------


class TestNode:
    def test_node_status(self):
        agent = DisentangleAgent("http://localhost:8000")
        agent._client = MagicMock()
        mock_data = {"peer_id": "abc", "dag_size": 42}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.node_status()
        assert result["dag_size"] == 42

    def test_network_health(self):
        agent = DisentangleAgent("http://localhost:8000")
        agent._client = MagicMock()
        mock_data = {"registered_dids": 10, "mean_network_curvature": 0.5}
        agent._client.request.return_value = _mock_response(200, mock_data)

        result = agent.network_health()
        assert result["registered_dids"] == 10
