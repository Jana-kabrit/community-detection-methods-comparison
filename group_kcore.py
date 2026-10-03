"""
Step 9 — Build the k-core hierarchy (SA).

NOTE: this reimplements the *general idea* of turning k-core numbers into a
nested hierarchy of size-bounded, connected communities. Cross-check the
actual "Core-based Hierarchies" paper's method section against this before
treating H2 as final, or use the authors' code if they released it — the
size-bounding rule below (deterministic BFS chunking) is a placeholder for
whatever exact recipe the paper specifies, not a copy of it.
"""
import json
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

SEED = 42
MAX_COMMUNITY_SIZE = 50  # placeholder bound — adjust once the paper's rule is confirmed

print("Loading entity graph...")
data = torch.load('entity_graph_full.pt', weights_only=False)
G = to_networkx(data, to_undirected=True)
G.remove_edges_from(nx.selfloop_edges(G))

print("Computing core numbers...")
core_numbers = nx.core_number(G)
k_values = sorted(set(core_numbers.values()))  # ascending
print(f"Core numbers range from {k_values[0]} to {k_values[-1]} ({len(k_values)} distinct shells)")


def split_oversized(nodes_in_component, max_size):
    """Deterministically chunk an oversized connected component via BFS order."""
    nodes_in_component = sorted(nodes_in_component)  # fixed starting order
    subG = G.subgraph(nodes_in_component)
    start = nodes_in_component[0]
    bfs_order = [start] + [v for _, v in nx.bfs_edges(subG, start)]
    remaining = set(nodes_in_component) - set(bfs_order)
    bfs_order += sorted(remaining)  # disconnected leftovers, shouldn't normally happen
    return [bfs_order[i:i + max_size] for i in range(0, len(bfs_order), max_size)]


assignments = {}
# level 0 = coarsest (lowest core threshold); increasing level = higher k
# threshold (tighter groups). Higher-k subgraphs are always subsets of
# lower-k ones, so this nests naturally like hierarchical Leiden's levels.
for level, k in enumerate(k_values):
    nodes_at_k = [n for n, c in core_numbers.items() if c >= k]
    subG = G.subgraph(nodes_at_k)
    components = list(nx.connected_components(subG))

    cluster_id = 0
    is_final_level = (level == len(k_values) - 1)
    for comp in components:
        chunks = split_oversized(comp, MAX_COMMUNITY_SIZE) if len(comp) > MAX_COMMUNITY_SIZE else [sorted(comp)]

        for chunk in chunks:
            for node in chunk:
                assignments.setdefault(str(node), []).append({
                    "level": level,
                    "cluster": cluster_id,
                    "is_final": is_final_level,
                })
            cluster_id += 1

with open('kcore_communities.json', 'w') as f:
    json.dump({"seed": SEED, "max_community_size": MAX_COMMUNITY_SIZE, "assignments": assignments}, f)

print(f"Nodes assigned: {len(assignments)}")
print("Sample node:", list(assignments.items())[0])

# Determinism check: rerunning should give byte-identical output — there's no
# randomness in core_number, connected_components (fixed graph), or the BFS
# chunking above. That's k-core's whole selling point vs. Leiden.

with open('kcore_run_config.json', 'w') as f:
    json.dump({
        "method": "kcore",
        "seed": SEED,
        "max_community_size": MAX_COMMUNITY_SIZE,
        "communities_file": "kcore_communities.json",
    }, f, indent=2)
