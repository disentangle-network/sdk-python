#!/usr/bin/env python3
"""Basic usage example for the Disentangle SDK.

This example demonstrates:
- Registering agents
- Creating and delegating capabilities
- Building social connections
- Checking coherence metrics
"""

from disentangle_sdk import DisentangleAgent, CapabilityDeniedError


def main():
    """Run basic SDK operations."""
    # Connect to a Disentangle node
    print("Connecting to Disentangle node...")
    agent = DisentangleAgent("http://localhost:8000")

    # Register as an AGI agent
    print("\nRegistering agent...")
    identity = agent.register(agent_type="agi")
    print(f"Registered as: {identity.did}")

    # Check initial coherence
    print("\nChecking coherence...")
    coherence = agent.coherence()
    print(f"  Composite score: {coherence.composite_score:.3f}")
    print(f"  Topological mass: {coherence.topological_mass:.3f}")
    print(f"  Relational diversity: {coherence.relational_diversity}")

    # Create a capability
    print("\nCreating capability...")
    capability = agent.create_capability(
        subject={"type": "file", "scope": "read"},
        constraints=[{"path": "/data/*.txt"}],
        delegatable=True,
    )
    print(f"Created capability: {capability.capability_id_hex}")

    # Create another agent to demonstrate delegation
    print("\nCreating peer agent...")
    peer = DisentangleAgent("http://localhost:8000")
    peer_identity = peer.register(agent_type="agi")
    print(f"Peer registered as: {peer_identity.did}")

    # Mutual introduction
    print("\nIntroducing agents...")
    result1 = agent.introduce(peer.did, edge_name="collaborator")
    result2 = peer.introduce(agent.did, edge_name="collaborator")
    print(f"Mutual introduction complete")
    print(f"  agent -> peer: {'OK' if result1 else 'FAILED'}")
    print(f"  peer -> agent: {'OK' if result2 else 'FAILED'}")

    # Check curvature between the two agents
    curvature = agent.curvature_with(peer.did)
    print(f"  Curvature: {curvature:.3f}")

    # Check updated coherence
    print("\nUpdated coherence after connection...")
    coherence_after = agent.coherence()
    print(f"  Composite score: {coherence_after.composite_score:.3f}")
    print(f"  Relational diversity: {coherence_after.relational_diversity}")
    print(f"  Neighbors: {len(agent.neighbors())}")

    # Delegate capability to peer
    print("\nDelegating capability to peer...")
    delegation = agent.delegate(capability.capability_id_hex, peer.did)
    print("Delegation successful")

    # Peer invokes capability
    print("\nPeer invoking capability...")
    try:
        result = peer.invoke(capability.capability_id_hex)
        if result:
            print("Capability invoked successfully")
    except CapabilityDeniedError as e:
        print(f"Capability denied - coherence too low: {e.coherence_score}")

    # Assign a petname
    print("\nAssigning petname...")
    agent.name(peer.did, "my-collaborator")
    resolved_did = agent.resolve_name("my-collaborator")
    print(f"Petname 'my-collaborator' resolves to: {resolved_did}")

    # List all capabilities
    print("\nListing capabilities...")
    capabilities = agent.list_capabilities()
    print(f"Agent holds {len(capabilities)} capability/ies")

    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
