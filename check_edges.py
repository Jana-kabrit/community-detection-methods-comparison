import json, random, torch

data = torch.load('entity_graph_full.pt', weights_only=False)
N = data.x.size(0) if getattr(data, 'x', None) is not None else data.num_nodes
row, col = data.edge_index[0].tolist(), data.edge_index[1].tolist()
edges = [(a, b) for a, b in zip(row, col) if a != b]
print(f"nodes={N:,}  edges (non-self-loop entries)={len(edges):,}")

def frac_inside(labels):
    return sum(1 for a, b in edges if labels[a] == labels[b]) / len(edges)

def load(path, which):
    d = json.load(open(path))['assignments']
    out = []
    for i in range(N):
        v = d[str(i)]
        e = v[0] if which == 'top' else [x for x in v if x['is_final']][0]
        out.append((e['level'], e['cluster']))
    return out

random.seed(0)
rows = [
    ('DMoN top level (123 clusters)', load('dmon_communities.json', 'top')),
    ('DMoN leaves', load('dmon_communities.json', 'leaf')),
    ('flat Leiden (for comparison)', load('flat_leiden_communities.json', 'leaf')),
]
print(f"{'grouping':<32}{'edges inside a community':>26}{'shuffled baseline':>20}")
for name, labels in rows:
    shuf = labels[:]
    random.shuffle(shuf)
    print(f"{name:<32}{frac_inside(labels):>26.3f}{frac_inside(shuf):>20.4f}")
