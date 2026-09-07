import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Load the graph
data = torch.load('entity_graph_full.pt', weights_only=False)
x = data.x
edge_index = data.edge_index
num_nodes = data.num_nodes
num_edges = data.num_edges

# Normalize embeddings for cosine similarity
x_norm = F.normalize(x, p=2, dim=1)

# --- Connected-pair similarity ---
src, dst = edge_index[0], edge_index[1]
connected_sim = (x_norm[src] * x_norm[dst]).sum(dim=1)
connected_sim_mean = connected_sim.mean().item()

# --- Random-pair similarity ---
torch.manual_seed(42)
num_samples = num_edges
rand_src = torch.randint(0, num_nodes, (num_samples,))
rand_dst = torch.randint(0, num_nodes, (num_samples,))
random_sim = (x_norm[rand_src] * x_norm[rand_dst]).sum(dim=1)
random_sim_mean = random_sim.mean().item()

# --- Standardized gap (Cohen's d) ---
mean_c, std_c = connected_sim.mean().item(), connected_sim.std().item()
mean_r, std_r = random_sim.mean().item(), random_sim.std().item()
n_c, n_r = len(connected_sim), len(random_sim)
pooled_std = np.sqrt(((n_c - 1) * std_c**2 + (n_r - 1) * std_r**2) / (n_c + n_r - 2))
cohens_d = (mean_c - mean_r) / pooled_std

print(f"Nodes: {num_nodes}")
print(f"Edges: {num_edges}")
print()
print(f"Connected-pair mean similarity: {connected_sim_mean:.4f}")
print(f"Random-pair mean similarity:    {random_sim_mean:.4f}")
print(f"Standardized gap (Cohen's d):   {cohens_d:.4f}")

# --- Chart: overlapping histograms ---
connected_sim_np = connected_sim.numpy()
random_sim_np = random_sim.numpy()

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-0.2, 0.8, 41)
ax.hist(random_sim_np, bins=bins, alpha=0.6, label='Random pairs', color='#94a3b8', density=True)
ax.hist(connected_sim_np, bins=bins, alpha=0.6, label='Connected pairs', color='#2563eb', density=True)
ax.axvline(mean_c, color='#2563eb', linestyle='--', linewidth=1.5)
ax.axvline(mean_r, color='#64748b', linestyle='--', linewidth=1.5)
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Density')
ax.set_title(f"Embedding Similarity: Connected vs Random Node Pairs\n(Cohen's d = {cohens_d:.2f})")
ax.legend()
fig.tight_layout()
fig.savefig('homophily_chart.png', dpi=150)
print("\nChart saved as homophily_chart.png")