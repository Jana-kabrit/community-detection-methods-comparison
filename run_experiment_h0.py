"""
Step 4 — Experiment H0: how friendly is our graph? (SB)

Measures homophily on the MuSiQue entity graph: average embedding similarity
of connected node pairs vs. random node pairs, plus a homophily ratio. High
gap -> the GNN should do well later. Small gap -> budget extra DMoN tuning
time in week 4 (see H0 discussion at the W1 Friday sync).
"""
import json, random
import torch
import networkx as nx

random.seed(42)
torch.manual_seed(42)

print("Loading node embeddings...")
emb = torch.load('entity_embeddings.pt', weights_only=False)
node_names = emb['nodes']
embeddings = emb['embeddings']
name_to_idx = {name: i for i, name in enumerate(node_names)}

print("Rebuilding entity graph from OpenIE extractions...")
with open('outputs/musique/openie_results_ner_gpt-4o-mini.json') as f:
    data = json.load(f)

G = nx.DiGraph()
for entry in data['docs']:
    for triple in entry.get('extracted_triples', []):
        if len(triple) == 3:
            subj, rel, obj = triple
            G.add_node(subj)
            G.add_node(obj)
            G.add_edge(subj, obj, relation=rel)

edges = list(G.edges())
print(f"Graph has {G.number_of_nodes()} nodes, {len(edges)} edges")


def cos_sim(i, j):
    a, b = embeddings[i], embeddings[j]
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)


# --- Connected-pair similarity ---
connected_sims = []
for u, v in edges:
    if u in name_to_idx and v in name_to_idx:
        connected_sims.append(cos_sim(name_to_idx[u], name_to_idx[v]).item())

avg_connected = sum(connected_sims) / len(connected_sims)
print(f"Connected-pair avg similarity: {avg_connected:.4f}  (n={len(connected_sims)})")

# --- Random-pair similarity (same sample size as connected pairs) ---
n_nodes = len(node_names)
random_sims = []
for _ in range(len(connected_sims)):
    i, j = random.randrange(n_nodes), random.randrange(n_nodes)
    while j == i:
        j = random.randrange(n_nodes)
    random_sims.append(cos_sim(i, j).item())

avg_random = sum(random_sims) / len(random_sims)
print(f"Random-pair avg similarity:    {avg_random:.4f}  (n={len(random_sims)})")

gap = avg_connected - avg_random
print(f"\nGap (connected - random): {gap:.4f}")

# --- Homophily ratio proxy ---
# There's no ground-truth node "class" on an LLM-extracted entity graph (the
# usual homophily-ratio ingredient), so this uses an embedding-based analog:
# an edge counts as "homophilous" if its endpoints are more similar than the
# average random pair. Flag this proxy choice at the sync — it's a stand-in,
# not the textbook definition.
homophilous_edges = sum(1 for s in connected_sims if s > avg_random)
homophily_ratio = homophilous_edges / len(connected_sims)
print(f"Homophily ratio (edges above random baseline): {homophily_ratio:.4f}")

results = {
    "avg_connected_similarity": avg_connected,
    "avg_random_similarity": avg_random,
    "similarity_gap": gap,
    "homophily_ratio": homophily_ratio,
    "n_edges_sampled": len(connected_sims),
}
with open('h0_homophily_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved h0_homophily_results.json")
print("One-line interpretation for the sync:")
if gap > 0.1:
    print(f"  Gap = {gap:.3f} -> meaningful homophily, GNN should have real signal to exploit.")
else:
    print(f"  Gap = {gap:.3f} -> weak homophily, budget extra DMoN tuning time in week 4.")
