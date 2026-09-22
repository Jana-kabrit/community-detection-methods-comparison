"""
Task 11 - DMoN dry-run on a toy graph
Build a toy graph with two obvious cliques, run DMoN, confirm it finds
the two groups, then demonstrate the collapse failure mode.
"""
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, DMoNPooling
from torch_geometric.utils import to_dense_adj, to_dense_batch

# ===========================================================
# 1. Build a toy graph with two obvious cliques (8 nodes total)
# ===========================================================
edges = []
clique_a = [0, 1, 2, 3]
clique_b = [4, 5, 6, 7]

for i in clique_a:
    for j in clique_a:
        if i != j:
            edges.append((i, j))

for i in clique_b:
    for j in clique_b:
        if i != j:
            edges.append((i, j))

# One bridge edge connecting the two cliques, so it's not two fully
# disconnected islands, but still clearly two distinct groups.
edges.append((3, 4))
edges.append((4, 3))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
num_nodes = 8

torch.manual_seed(0)
x = torch.randn(num_nodes, 16)  # simple random node features
data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

print("Toy graph built:")
print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"  True groups -> Clique A: {clique_a}, Clique B: {clique_b}")
print()


# ===========================================================
# 2. GNN body + DMoN pooling model
# ===========================================================
class DMoNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters, zero_init=False):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool = DMoNPooling([hidden_channels, hidden_channels], num_clusters)

        if zero_init:
            # Force every node to start with an IDENTICAL cluster score
            # (a perfectly symmetric "collapsed" starting point).
            with torch.no_grad():
                last_layer = self.pool.mlp.lins[-1]
                last_layer.weight.zero_()
                if last_layer.bias is not None:
                    last_layer.bias.zero_()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x_dense, mask = to_dense_batch(x, batch=None)
        adj_dense = to_dense_adj(edge_index)
        s, x_pooled, adj_pooled, spectral_loss, ortho_loss, cluster_loss = self.pool(
            x_dense, adj_dense, mask
        )
        return s, spectral_loss, ortho_loss, cluster_loss


def train_and_report(use_ortho=True, use_cluster=True, zero_init=False,
                      epochs=300, lr=0.01, seed=0):
    torch.manual_seed(seed)
    model = DMoNModel(in_channels=16, hidden_channels=32, num_clusters=2,
                       zero_init=zero_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        s, spectral_loss, ortho_loss, cluster_loss = model(data.x, data.edge_index)
        loss = spectral_loss
        if use_ortho:
            loss = loss + ortho_loss
        if use_cluster:
            loss = loss + cluster_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        s, spectral_loss, ortho_loss, cluster_loss = model(data.x, data.edge_index)
        assignments = s.squeeze(0).argmax(dim=1).tolist()

    return assignments


# ===========================================================
# 3. Run WITH the full DMoN loss (spectral + orthogonality + collapse reg)
# ===========================================================
print("=== Run 1: Full DMoN loss ===")
assignments_full = train_and_report(use_ortho=True, use_cluster=True)
print(f"Cluster assignments: {assignments_full}")
print("Expected: [0,0,0,0,1,1,1,1] or the 0/1 labels swapped")
print()

# ===========================================================
# 4. Run WITHOUT the collapse regularizer, normal random init
# ===========================================================
print("=== Run 2: Collapse regularizer removed (normal random init) ===")
assignments_no_reg = train_and_report(use_ortho=True, use_cluster=False)
print(f"Cluster assignments: {assignments_no_reg}")
print("Note: on this cleanly-separated toy graph, modularity math already")
print("disfavors collapsing everyone into one cluster, so this often still")
print("finds the correct split -- collapse isn't guaranteed just by dropping")
print("this term when the community structure is this strong.")
print()

# ===========================================================
# 5. Force a genuine collapse: symmetric ("stuck") starting point
# ===========================================================
print("=== Run 3: Symmetric zero-init start, spectral loss ONLY ===")
assignments_collapsed = train_and_report(use_ortho=False, use_cluster=False,
                                          zero_init=True)
print(f"Cluster assignments: {assignments_collapsed}")
print(f"Unique clusters used: {len(set(assignments_collapsed))}")
print("This is the real collapse failure mode: every node starts with an")
print("identical score, so without the collapse regularizer's gradient to")
print("break the symmetry, training gets stuck and everything lands in")
print("one cluster.")
