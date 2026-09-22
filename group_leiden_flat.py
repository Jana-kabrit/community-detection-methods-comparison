"""
Step 10 — Experiment H2b: single-level (flat) Leiden (SA).

Run directly via leidenalg (not graspologic's hierarchical_leiden) so we can
tell whether the *hierarchy* or the *algorithm itself* is doing the work in
H1's results.
"""
import json
import torch
import networkx as nx
import igraph as ig
import leidenalg
from torch_geometric.utils import to_networkx

SEED = 42

print("Loading entity graph...")
data = torch.load('entity_graph_full.pt', weights_only=False)
G = to_networkx(data, to_undirected=True)
G.remove_edges_from(nx.selfloop_edges(G))

print("Converting to igraph...")
node_list = list(G.nodes())
node_to_idx = {n: i for i, n in enumerate(node_list)}
edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

ig_graph = ig.Graph(n=len(node_list), edges=edges)

print("Running flat Leiden (leidenalg, RBConfigurationVertexPartition)...")
partition = leidenalg.find_partition(
    ig_graph,
    leidenalg.RBConfigurationVertexPartition,
    seed=SEED,
    n_iterations=-1,  # run to convergence
)

print(f"Flat Leiden found {len(partition)} communities")

assignments = {}
for cluster_id, community in enumerate(partition):
    for idx in community:
        node = node_list[idx]
        assignments[str(node)] = [{
            "level": 0,
            "cluster": cluster_id,
            "is_final": True,
        }]

with open('flat_leiden_communities.json', 'w') as f:
    json.dump({"seed": SEED, "assignments": assignments}, f)

print(f"Nodes assigned: {len(assignments)}")
print("Sample node:", list(assignments.items())[0])

with open('flat_leiden_run_config.json', 'w') as f:
    json.dump({
        "method": "flat_leiden",
        "seed": SEED,
        "communities_file": "flat_leiden_communities.json",
    }, f, indent=2)
