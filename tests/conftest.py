"""pytest configuration for disentangle-sdk tests."""

import os
import pytest


# Default node URL for testing
NODE_URL = os.environ.get("DISENTANGLE_NODE_URL", "http://localhost:8000")


@pytest.fixture
def node_url():
    """Provide the node URL for tests."""
    return NODE_URL


# Skip marker for tests that require a running node
requires_node = pytest.mark.skipif(
    not os.environ.get("DISENTANGLE_NODE_URL"),
    reason="Requires running Disentangle node (set DISENTANGLE_NODE_URL)",
)
