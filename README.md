# Disentangle SDK (Python)

Python client library for the Disentangle Protocol - a decentralized identity and capability system with sybil resistance through topological coherence.

## Installation

```bash
pip install disentangle-sdk
```

Or with uv:

```bash
uv pip install disentangle-sdk
```

## Quick Start

```python
from disentangle_sdk import DisentangleAgent

# Connect to a Disentangle node
agent = DisentangleAgent("http://localhost:8000")

# Register as an AGI agent
identity = agent.register(agent_type="agi")
print(f"Registered as {identity.did}")

# Create a capability
capability = agent.create_capability(
    subject_type="file",
    scope="read",
    delegatable=True
)

# Check your coherence score
coherence = agent.coherence()
print(f"Coherence score: {coherence.composite_score:.2f}")
```

## Features

### Identity Management
- Register agents (AGI, human, etc.)
- Lookup identity documents
- DID-based addressing

### Capabilities
- Create fine-grained capabilities
- Delegate capabilities to other agents
- Invoke capabilities with coherence-based authorization
- Revoke capabilities (single or chain)

### Social Graph
- Introduce agents to each other
- Build introduction chains
- Calculate curvature between agents
- Query neighbors

### Coherence Metrics
- Query your own coherence
- Look up peer coherence
- Sybil resistance through topological mass
- Relational diversity measurement

### Petnames
- Assign human-friendly names to DIDs
- Resolve petnames to DIDs
- Local namespace per agent

## Architecture

The SDK follows clean architecture principles:

- **Client Layer** (`DisentangleAgent`) - HTTP client wrapping all RPC endpoints
- **Type Layer** (`types.py`) - Pydantic models for data validation
- **Exception Layer** (`exceptions.py`) - Typed exceptions for error handling

All operations are synchronous for simplicity. The SDK stores the signing key in memory after registration.

## API Reference

### DisentangleAgent

#### Identity Methods
- `register(agent_type, model_hash, runtime_hash)` - Register a new agent
- `get_identity(did)` - Lookup an identity document
- `did` - Property to get current agent's DID
- `is_registered` - Check if registered

#### Capability Methods
- `create_capability(subject_type, scope, constraints, delegatable)` - Create capability
- `delegate(capability_id, to_did)` - Delegate to another agent
- `invoke(capability_id)` - Invoke a capability
- `revoke(capability_id, scope)` - Revoke a capability
- `list_capabilities()` - List all held capabilities

#### Social Graph Methods
- `introduce(other_did, edge_name)` - Introduce to another agent
- `get_introduction_chain(to_did)` - Get introduction path
- `curvature_with(other_did)` - Calculate curvature
- `neighbors()` - Get neighbor DIDs

#### Coherence Methods
- `coherence()` - Get own coherence metrics
- `peer_coherence(did)` - Get peer's coherence
- `curvature_with(other_did)` - Calculate curvature

#### Petname Methods
- `name(did, petname)` - Assign petname
- `resolve_name(petname)` - Resolve to DID

#### Node Methods
- `node_status()` - Get node information

## Error Handling

The SDK provides typed exceptions:

```python
from disentangle_sdk import (
    DisentangleError,         # Base exception
    DIDNotFoundError,         # DID not found (404)
    CapabilityDeniedError,    # Capability denied (403)
    NodeConnectionError,      # Cannot connect to node
    NotRegisteredError,       # Agent not registered
)

try:
    agent.invoke(capability_id)
except CapabilityDeniedError as e:
    print(f"Denied - coherence: {e.coherence_score}")
except DIDNotFoundError:
    print("DID not found on network")
```

## Testing

Run tests against a live Disentangle node:

```bash
# Set node URL
export DISENTANGLE_NODE_URL=http://localhost:8000

# Run tests
pytest tests/
```

Tests are skipped if `DISENTANGLE_NODE_URL` is not set.

## Development

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/

# Type check
mypy src/
```

## Context Manager Support

The SDK supports context managers for automatic cleanup:

```python
with DisentangleAgent("http://localhost:8000") as agent:
    agent.register(agent_type="agi")
    print(agent.did)
# HTTP client automatically closed
```

## License

MIT

## See Also

- [Disentangle Protocol Specification](../../../SPEC.md)
- [Agent Bridge Specification](../../../CC_SPEC_AGENT_BRIDGE.md)
- [Rust Implementation](../../../disentangle-core/)
