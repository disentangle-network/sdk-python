"""Tests for excitability gradient, oracle, pool, and topology SDK methods.

These tests mock HTTP calls to validate the SDK interface without
requiring a running Disentangle node.
"""

import pytest
from unittest.mock import patch, MagicMock

from disentangle_sdk import (
    DisentangleAgent,
    DIDNotFoundError,
    NotRegisteredError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registered_agent() -> DisentangleAgent:
    """Create a DisentangleAgent and fake-register it without HTTP."""
    agent = DisentangleAgent("http://localhost:8000")
    # Manually set internal state as if register() succeeded
    from disentangle_sdk.types import AgentIdentity

    agent._identity = AgentIdentity(
        did="did:disentangle:test_agent_abc",
        signing_key_hex="deadbeef" * 8,
        document={"type": "agi"},
    )
    agent._signing_key_hex = "deadbeef" * 8
    return agent


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Build a fake httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = json_data
    return resp


# =========================================================================
# Gradient (Excitability) Methods
# =========================================================================


class TestCurvatureDerivative:
    """Tests for curvature_derivative()."""

    def test_curvature_derivative_success(self):
        agent = _make_registered_agent()
        expected = {
            "did_a": "did:disentangle:alice",
            "did_b": "did:disentangle:bob",
            "derivative": 0.042,
            "window": 100,
        }
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.curvature_derivative(
                "did:disentangle:alice", "did:disentangle:bob"
            )
            assert result == expected
            mock_req.assert_called_once_with(
                method="GET",
                url="/coherence/gradient/did:disentangle:alice/did:disentangle:bob",
                json=None,
                params={"window": 100},
            )

    def test_curvature_derivative_custom_window(self):
        agent = _make_registered_agent()
        expected = {"derivative": 0.1, "window": 50}
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.curvature_derivative("did:a", "did:b", window=50)
            assert result == expected
            mock_req.assert_called_once_with(
                method="GET",
                url="/coherence/gradient/did:a/did:b",
                json=None,
                params={"window": 50},
            )

    def test_curvature_derivative_not_found(self):
        agent = _make_registered_agent()
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"error": "Not found"}, 404),
        ):
            with pytest.raises(DIDNotFoundError):
                agent.curvature_derivative("did:unknown:a", "did:unknown:b")


class TestExcitability:
    """Tests for excitability()."""

    def test_excitability_success(self):
        agent = _make_registered_agent()
        expected = {
            "did": "did:disentangle:alice",
            "excitability": 0.73,
            "mean_derivative": 0.02,
            "edge_count": 5,
        }
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.excitability("did:disentangle:alice")
            assert result == expected
            mock_req.assert_called_once_with(
                method="GET",
                url="/coherence/excitability/did:disentangle:alice",
                json=None,
                params={"window": 100},
            )

    def test_excitability_custom_window(self):
        agent = _make_registered_agent()
        expected = {"excitability": 0.5}
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ):
            result = agent.excitability("did:test", window=200)
            assert result == expected

    def test_excitability_not_found(self):
        agent = _make_registered_agent()
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"error": "Not found"}, 404),
        ):
            with pytest.raises(DIDNotFoundError):
                agent.excitability("did:nonexistent")


class TestGradientMap:
    """Tests for gradient_map()."""

    def test_gradient_map_defaults(self):
        agent = _make_registered_agent()
        expected = {
            "edges": [
                {"did_a": "did:a", "did_b": "did:b", "derivative": 0.1},
                {"did_a": "did:c", "did_b": "did:d", "derivative": -0.05},
            ],
            "top_n": 20,
            "window": 100,
        }
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.gradient_map()
            assert result == expected
            mock_req.assert_called_once_with(
                method="GET",
                url="/coherence/gradient/map",
                json=None,
                params={"top_n": 20, "window": 100},
            )

    def test_gradient_map_custom_params(self):
        agent = _make_registered_agent()
        expected = {"edges": [], "top_n": 5, "window": 50}
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.gradient_map(top_n=5, window=50)
            assert result == expected
            mock_req.assert_called_once_with(
                method="GET",
                url="/coherence/gradient/map",
                json=None,
                params={"top_n": 5, "window": 50},
            )


# =========================================================================
# Topology Methods
# =========================================================================


class TestNeighborhoods:
    """Tests for neighborhoods()."""

    def test_neighborhoods_success(self):
        agent = _make_registered_agent()
        expected_neighborhoods = [
            {"cluster_hash": "abc123", "member_count": 3, "mean_curvature": 0.5},
            {"cluster_hash": "def456", "member_count": 7, "mean_curvature": 0.8},
        ]
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"neighborhoods": expected_neighborhoods}),
        ):
            result = agent.neighborhoods()
            assert result == expected_neighborhoods
            assert len(result) == 2

    def test_neighborhoods_empty(self):
        agent = _make_registered_agent()
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"neighborhoods": []}),
        ):
            result = agent.neighborhoods()
            assert result == []


# =========================================================================
# Oracle Methods
# =========================================================================


class TestQueryOracle:
    """Tests for query_oracle()."""

    def test_query_oracle_global(self):
        agent = _make_registered_agent()
        expected = {
            "query_id": "abc123",
            "weights": {"did:a": 0.6, "did:b": 0.4},
            "merkle_root": "ff00ff",
        }
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.query_oracle(
                region={"global": True},
                depth_start=0,
                depth_end=1000,
            )
            assert result == expected
            mock_req.assert_called_once_with(
                method="POST",
                url="/oracle/query",
                json={
                    "region": {"global": True},
                    "depth_start": 0,
                    "depth_end": 1000,
                },
                params=None,
            )

    def test_query_oracle_requires_registration(self):
        agent = DisentangleAgent("http://localhost:8000")
        with pytest.raises(NotRegisteredError):
            agent.query_oracle({"global": True}, 0, 100)


class TestGetDistribution:
    """Tests for get_distribution()."""

    def test_get_distribution_success(self):
        agent = _make_registered_agent()
        expected = {
            "query_id": "abc123",
            "weights": {"did:a": 0.6},
            "merkle_root": "ff00ff",
        }
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.get_distribution("abc123")
            assert result == expected
            mock_req.assert_called_once_with(
                method="GET",
                url="/oracle/distribution/abc123",
                json=None,
                params=None,
            )

    def test_get_distribution_not_found(self):
        agent = _make_registered_agent()
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"error": "Not found"}, 404),
        ):
            with pytest.raises(DIDNotFoundError):
                agent.get_distribution("nonexistent")


class TestListDistributions:
    """Tests for list_distributions()."""

    def test_list_distributions_success(self):
        agent = _make_registered_agent()
        expected_dists = [
            {"query_id": "abc", "merkle_root": "ff00"},
            {"query_id": "def", "merkle_root": "00ff"},
        ]
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"distributions": expected_dists}),
        ) as mock_req:
            result = agent.list_distributions()
            assert result == expected_dists
            assert len(result) == 2
            mock_req.assert_called_once_with(
                method="GET",
                url="/oracle/distributions",
                json=None,
                params=None,
            )

    def test_list_distributions_empty(self):
        agent = _make_registered_agent()
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"distributions": []}),
        ):
            result = agent.list_distributions()
            assert result == []


# =========================================================================
# Commons Pool Methods
# =========================================================================


class TestCreatePool:
    """Tests for create_pool()."""

    def test_create_pool_success(self):
        agent = _make_registered_agent()
        expected = {
            "id": "pool_001",
            "name": "Research Fund",
            "balance": 0.0,
        }
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.create_pool("Research Fund", description="For research")
            assert result == expected
            mock_req.assert_called_once_with(
                method="POST",
                url="/pool/create",
                json={
                    "name": "Research Fund",
                    "description": "For research",
                    "creator_did": agent.did,
                    "signing_key_hex": agent._signing_key_hex,
                },
                params=None,
            )

    def test_create_pool_requires_registration(self):
        agent = DisentangleAgent("http://localhost:8000")
        with pytest.raises(NotRegisteredError):
            agent.create_pool("Test Pool")

    def test_create_pool_default_description(self):
        agent = _make_registered_agent()
        expected = {"id": "pool_002", "name": "Minimal"}
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.create_pool("Minimal")
            assert result == expected
            call_json = mock_req.call_args.kwargs["json"]
            assert call_json["description"] == ""


class TestPoolDeposit:
    """Tests for pool_deposit()."""

    def test_pool_deposit_success(self):
        agent = _make_registered_agent()
        expected = {"success": True, "new_balance": 100.0}
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.pool_deposit("pool_001", 100.0, source="grant")
            assert result == expected
            mock_req.assert_called_once_with(
                method="POST",
                url="/pool/deposit",
                json={
                    "pool_id": "pool_001",
                    "amount": 100.0,
                    "source": "grant",
                    "depositor_did": agent.did,
                    "signing_key_hex": agent._signing_key_hex,
                },
                params=None,
            )

    def test_pool_deposit_requires_registration(self):
        agent = DisentangleAgent("http://localhost:8000")
        with pytest.raises(NotRegisteredError):
            agent.pool_deposit("pool_001", 50.0)

    def test_pool_deposit_default_source(self):
        agent = _make_registered_agent()
        expected = {"success": True, "new_balance": 50.0}
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            agent.pool_deposit("pool_001", 50.0)
            call_json = mock_req.call_args.kwargs["json"]
            assert call_json["source"] == ""


class TestPoolDistribute:
    """Tests for pool_distribute()."""

    def test_pool_distribute_success(self):
        agent = _make_registered_agent()
        expected = {
            "success": True,
            "allocations": {"did:a": 60.0, "did:b": 40.0},
            "remaining_balance": 0.0,
        }
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.pool_distribute("pool_001", "dist_abc")
            assert result == expected
            mock_req.assert_called_once_with(
                method="POST",
                url="/pool/distribute",
                json={
                    "pool_id": "pool_001",
                    "distribution_id": "dist_abc",
                    "initiator_did": agent.did,
                    "signing_key_hex": agent._signing_key_hex,
                },
                params=None,
            )

    def test_pool_distribute_requires_registration(self):
        agent = DisentangleAgent("http://localhost:8000")
        with pytest.raises(NotRegisteredError):
            agent.pool_distribute("pool_001", "dist_abc")


class TestPoolStatus:
    """Tests for pool_status()."""

    def test_pool_status_success(self):
        agent = _make_registered_agent()
        expected = {
            "id": "pool_001",
            "name": "Research Fund",
            "balance": 100.0,
            "min_coherence": 0.3,
        }
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.pool_status("pool_001")
            assert result == expected
            mock_req.assert_called_once_with(
                method="GET",
                url="/pool/pool_001",
                json=None,
                params=None,
            )

    def test_pool_status_not_found(self):
        agent = _make_registered_agent()
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"error": "Not found"}, 404),
        ):
            with pytest.raises(DIDNotFoundError):
                agent.pool_status("nonexistent")


class TestPoolClaims:
    """Tests for pool_claims()."""

    def test_pool_claims_success(self):
        agent = _make_registered_agent()
        expected_claims = [
            {"claimant": "did:a", "amount": 60.0, "distribution_id": "dist_1"},
            {"claimant": "did:b", "amount": 40.0, "distribution_id": "dist_1"},
        ]
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"claims": expected_claims}),
        ) as mock_req:
            result = agent.pool_claims("pool_001")
            assert result == expected_claims
            assert len(result) == 2
            mock_req.assert_called_once_with(
                method="GET",
                url="/pool/pool_001/claims",
                json=None,
                params=None,
            )

    def test_pool_claims_empty(self):
        agent = _make_registered_agent()
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"claims": []}),
        ):
            result = agent.pool_claims("pool_001")
            assert result == []

    def test_pool_claims_not_found(self):
        agent = _make_registered_agent()
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response({"error": "Not found"}, 404),
        ):
            with pytest.raises(DIDNotFoundError):
                agent.pool_claims("nonexistent")


class TestPoolClaim:
    """Tests for pool_claim()."""

    def test_pool_claim_success(self):
        agent = _make_registered_agent()
        expected = {"success": True, "amount": 60.0}
        with patch.object(
            agent._client,
            "request",
            return_value=_mock_response(expected),
        ) as mock_req:
            result = agent.pool_claim("pool_001", "dist_abc")
            assert result == expected
            mock_req.assert_called_once_with(
                method="POST",
                url="/pool/claim",
                json={
                    "pool_id": "pool_001",
                    "distribution_id": "dist_abc",
                    "claimant_did": agent.did,
                    "signing_key_hex": agent._signing_key_hex,
                },
                params=None,
            )

    def test_pool_claim_requires_registration(self):
        agent = DisentangleAgent("http://localhost:8000")
        with pytest.raises(NotRegisteredError):
            agent.pool_claim("pool_001", "dist_abc")
