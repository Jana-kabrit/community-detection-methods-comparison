"""
Train DMoN on the full entity graph (entity_graph_full.pt) — SB, week 1 step 4/5 -> week 4.

Why not PyG's built-in DMoNPooling:
  DMoNPooling expects a dense [B, N, N] adjacency. At N=103,056 that's ~1e10
  entries — won't fit in RAM. This script implements the DMoN loss directly
  on the sparse edge_index, so memory stays O(E * C) instead of O(N^2).

Storage footprint:
  - No dense adjacency is ever created.
  - Only the final model state_dict (a few MB) and the hard cluster
    assignment (one int32 per node, ~400KB) are saved. No per-epoch
    checkpoints, no soft assignment matrix on disk.

Usage:
  python train_dmon.py --graph entity_graph_full.pt --n-clusters 64
  (set --n-clusters to Leiden's leaf-community count once you have it)
"""
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected


class DMoNEncoder(nn.Module):
    """2-layer GraphSAGE encoder + soft cluster-assignment head."""

    def __init__(self, in_dim, hidden_dim, n_clusters, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.assign = nn.Linear(hidden_dim, n_clusters)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        s_logits = self.assign(h)
        s = F.softmax(s_logits, dim=1)
        return s


def sparse_dmon_loss(s, edge_index, num_nodes, n_clusters, collapse_weight=1.0):
    """
    DMoN loss computed without a dense adjacency matrix.

    modularity term:  -(1/2m) * [ Tr(S^T A S) - (1/2m) * ||d^T S||^2 ]
    collapse term:     (sqrt(C)/N) * ||sum_i S_i||_2 - 1

    edge_index must already be undirected (symmetric), each undirected
    edge appearing as both (i,j) and (j,i).
    """
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=s.dtype)
    two_m = row.size(0)  # each undirected edge counted twice -> sum(deg) == 2m
    m = two_m / 2.0

    # (A S)_i = sum_{j: (i,j) in E} S_j  -- scatter-add over edges, O(E*C)
    AS = s.new_zeros(num_nodes, n_clusters)
    AS.index_add_(0, row, s[col])

    tr_SAS = (s * AS).sum()

    dS = torch.matmul(deg, s)  # shape (C,)  == d^T S
    degree_term = (dS * dS).sum() / two_m

    modularity_loss = -(tr_SAS - degree_term) / two_m

    cluster_sizes = s.sum(dim=0)  # (C,)
    collapse_loss = (n_clusters ** 0.5) / num_nodes * torch.norm(cluster_sizes, p=2) - 1

    return modularity_loss + collapse_weight * collapse_loss, modularity_loss.item(), collapse_loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="entity_graph_full.pt")
    ap.add_argument("--n-clusters", type=int, default=64,
                     help="Set this to Leiden's leaf-community count for a fair comparison.")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--collapse-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20,
                     help="Stop early if loss doesn't improve for this many epochs.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.graph} ...")
    data = torch.load(args.graph, map_location="cpu", weights_only=False)
    x = data.x.to(device)
    edge_index = to_undirected(data.edge_index).to(device)  # symmetrize once, up front
    num_nodes = x.size(0)

    print(f"nodes={num_nodes:,}  undirected edges (directed entries)={edge_index.size(1):,}  "
          f"C={args.n_clusters}  device={device}")

    model = DMoNEncoder(x.size(1), args.hidden_dim, args.n_clusters, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        s = model(x, edge_index)
        loss, mod_loss, collapse_loss = sparse_dmon_loss(
            s, edge_index, num_nodes, args.n_clusters, args.collapse_weight
        )
        loss.backward()
        opt.step()

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                hard = s.argmax(dim=1)
                used_clusters = hard.unique().numel()
                # cheap collapse check without saving anything
                sizes = torch.bincount(hard, minlength=args.n_clusters).float()
                largest_frac = (sizes.max() / num_nodes).item()
            print(f"epoch {epoch:4d}  loss={loss.item(): .4f}  "
                  f"modularity={mod_loss: .4f}  collapse={collapse_loss: .4f}  "
                  f"clusters_used={used_clusters}/{args.n_clusters}  "
                  f"largest_cluster_frac={largest_frac:.3f}")

        if bad_epochs >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping early at epoch {epoch}.")
            break

    print(f"Training done in {time.time() - t0:.1f}s. Best loss: {best_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        s = model(x, edge_index)
        hard_assignment = s.argmax(dim=1).to(torch.int32).cpu()

    sizes = torch.bincount(hard_assignment, minlength=args.n_clusters)
    n_singletons = (sizes == 1).sum().item()
    n_empty = (sizes == 0).sum().item()
    largest_frac = (sizes.max().item() / num_nodes)

    print(f"Final: {args.n_clusters - n_empty} non-empty clusters, "
          f"{n_singletons} singletons, largest cluster = {largest_frac:.1%} of nodes.")
    if largest_frac > 0.5:
        print("WARNING: looks like a collapse (one giant cluster). "
              "Try raising --collapse-weight.")
    if n_singletons > 0.3 * args.n_clusters:
        print("WARNING: a lot of singleton clusters — try lowering --collapse-weight "
              "or reducing --n-clusters.")

    # Minimal-footprint outputs: small model state dict + int32 assignment vector.
    torch.save(best_state, f"{args.out_dir}/dmon_model.pt")
    torch.save(hard_assignment, f"{args.out_dir}/dmon_assignment.pt")
    with open(f"{args.out_dir}/dmon_run_config.json", "w") as f:
        json.dump(vars(args) | {"best_loss": best_loss, "num_nodes": num_nodes,
                                 "num_edges_undirected_entries": edge_index.size(1)}, f, indent=2)

    print(f"Saved: {args.out_dir}/dmon_model.pt, {args.out_dir}/dmon_assignment.pt, "
          f"{args.out_dir}/dmon_run_config.json")


if __name__ == "__main__":
    main()
