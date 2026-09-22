import torch, networkx as nx
from graspologic.partition import hierarchical_leiden
from torch_geometric.utils import to_networkx

data = torch.load('entity_graph_full.pt', weights_only=False)
G = to_networkx(data, to_undirected=True)

SEED = 42
communities = hierarchical_leiden(G, random_seed=SEED)

import json
assignments = {}
for c in communities:
    assignments.setdefault(str(c.node), []).append({
        "level": c.level, "cluster": c.cluster, "is_final": c.is_final_cluster
    })

with open('leiden_communities.json', 'w') as f:
    json.dump({"seed": SEED, "assignments": assignments}, f)

print(f"Nodes assigned: {len(assignments)}")
print("Sample node:", list(assignments.items())[0])