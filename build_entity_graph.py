import json
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# Load pre-built MuSiQue extractions
with open('outputs/musique/openie_results_ner_gpt-4o-mini.json') as f:
    data = json.load(f)

docs = data['docs']

# Build NetworkX graph: entities = nodes, relations = edges
G = nx.DiGraph()

for entry in docs:
    triples = entry.get('extracted_triples', [])
    for triple in triples:
        if len(triple) == 3:
            subj, rel, obj = triple
            G.add_node(subj)
            G.add_node(obj)
            G.add_edge(subj, obj, relation=rel)

print(f"Node count: {G.number_of_nodes()}")
print(f"Edge count: {G.number_of_edges()}")

degrees = [d for _, d in G.degree()]
avg_degree = sum(degrees) / len(degrees) if degrees else 0
print(f"Average degree: {avg_degree:.2f}")

# Generate embeddings for each node (entity name) using sentence-transformers
model = SentenceTransformer('all-MiniLM-L6-v2')
node_list = list(G.nodes())
embeddings = model.encode(node_list, show_progress_bar=True)

# Save embeddings once (reused all semester)
torch.save({'nodes': node_list, 'embeddings': torch.tensor(embeddings)}, 'entity_embeddings.pt')
print("Saved embeddings to entity_embeddings.pt")

# Attach embeddings to graph nodes, then convert to PyG
for i, node in enumerate(node_list):
    G.nodes[node]['x'] = embeddings[i]

pyg_data = from_networkx(G)
print(pyg_data)
torch.save(pyg_data, 'entity_graph_full.pt')
print("Saved graph to entity_graph_full.pt")
# Plot degree distribution
plt.figure(figsize=(8, 5))
plt.hist(degrees, bins=30)
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.title('Degree Distribution — MuSiQue Entity Graph')
plt.savefig('degree_distribution.png')
print("Saved plot to degree_distribution.png")