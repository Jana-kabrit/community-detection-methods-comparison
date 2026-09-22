"""
Recursive DMoN community detection — matches how hierarchical Leiden actually
builds its hierarchy, instead of one flat softmax over C=7,206 clusters.

WHY RECURSIVE, NOT FLAT C=7,206
--------------------------------
Hierarchical Leiden never finds 7,206 communities in one shot either. It finds
a small number of communities, re-runs inside each one, and keeps going —
7,206 is just where that recursion happens to bottom out (the leaf-level
count). Matching that raw number with a single flat DMoN softmax matches a
statistic, not the method: the fair comparison is to build DMoN's hierarchy
the same way, so both methods end up at their leaf counts through the same
kind of process.

There's a second, independent reason this is the right call: a single
softmax over 7,206 clusters is far outside DMoN's designed operating range
(the original paper works with tens of clusters). Even with unlimited GPU
memory, a flat C=7,206 run would likely train badly — the collapse
regularizer fighting thousands of near-empty clusters. Recursing with a
modest C at each level (tens to low hundreds) keeps every individual
training step inside DMoN's actual designed range.

As a side effect, this also fixes the memory problem: peak memory at any
point scales with the CURRENT subgraph's node count and the modest C, not
the full 103,056-node graph and C=7,206 — comfortably inside a free-tier
GPU, no more OOM fights.

HOW IT WORKS
------------
1. Start with the full graph. Train DMoN with a modest target cluster count
   (e.g. C=128) on it.
2. For each resulting cluster, if it's still bigger than a size threshold
   (and we haven't hit the depth cap), build the induced subgraph on just
   that cluster's nodes and recurse: train DMoN again, with C scaled down
   to something sensible for that subgraph's size.
3. Stop recursing into a cluster once it's small enough — that's a leaf
   community. The final leaf count and size distribution is what you
   compare against Leiden's own leaf-level output.

This reuses the exact same DMoN model/loss code (GCN encoder, sparse
modularity + collapse loss, chunked edge scatter-add, gradient checkpointing)
already verified correct and memory-safe in DMoNTrain_gpu_ready.py — the
only new part here is the recursion and subgraph bookkeeping.

USAGE
-----
Step 1 — pick good hyperparameters at a modest C on the full graph first
(reuses the existing sweep script, not this one):
    python DMoNTrain_gpu_ready.py --mode full --n-clusters 128
Read off the best (lr, hidden_dim, dropout, reg_weight) from
dmon_sweep_results.jsonl — or just use this script's defaults, which were
picked from configs that reliably avoided collapse in earlier testing
(lr=0.001, hidden_dim=128, dropout=0.0, reg_weight=0.5).

Step 2 — build the recursive hierarchy:
    python recursive_dmon.py --mode full --target-c 128

Quick pipeline sanity check first, no data file needed:
    python recursive_dmon.py --mode smoke

OUTPUT
------
- leaf_assignment.pt: (num_nodes,) LongTensor — entry i is node i's LEAF
  community id (globally unique across the whole tree). This is the
  thing to compare against Leiden's leaf-level assignment.
- recursion_tree.jsonl: one line per split performed (internal + leaf nodes
  of the recursion tree itself), so the hierarchy shape can be inspected —
  mirrors what "print the community at every level" means for Leiden.
- leaf_stats.json: leaf count + size distribution (min/max/mean/median),
  to put next to Leiden's own numbers in the lab notes / results sheet.
"""

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, remove_self_loops, degree, subgraph

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GRAPH_PATH = "entity_graph_full.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EDGE_CHUNK = 50_000  # same chunking fix as DMoNTrain_gpu_ready.py, same reason

SMOKE_N_NODES = 2000
SMOKE_AVG_DEGREE = 6


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph(path: str = GRAPH_PATH) -> Data:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"'{path}' not found. Run build_entity_graph.py first, or point "
            "--graph-path at wherever it landed. --mode smoke runs without "
            "this file if you just want to validate the pipeline."
        )
    data = torch.load(path, weights_only=False)
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=data.x.size(0))
    data.edge_index = edge_index
    if hasattr(data, "relation"):
        del data.relation
    return data


def make_smoke_graph(n_nodes: int = SMOKE_N_NODES, avg_degree: int = SMOKE_AVG_DEGREE,
                      in_dim: int = 384) -> Data:
    set_seed()
    n_edges = n_nodes * avg_degree // 2
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=n_nodes)
    x = torch.randn(n_nodes, in_dim)
    return Data(x=x, edge_index=edge_index, num_nodes=n_nodes)


# ---------------------------------------------------------------------------
# Model — identical to the verified flat-sweep script (encoder split from
# cluster head, memory fixes intact): edge-chunked scatter-add (the gather
# tensor is edge-sized, not node-sized, so chunk it), del logits right after
# softmax (its backward doesn't need the input), gradient checkpointing on
# the (N, C) segment. See DMoNTrain_gpu_ready.py for how each was verified.
# ---------------------------------------------------------------------------

class DMoNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_clusters: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.cluster_head = nn.Linear(hidden_dim, n_clusters)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return F.softmax(self.cluster_head(h), dim=-1)


def cluster_and_loss(h, cluster_head, row, col, deg, num_nodes, n_clusters, m, reg_weight):
    logits = cluster_head(h)
    s = F.softmax(logits, dim=-1)
    del logits

    as_ = torch.zeros_like(s)
    num_edges = row.size(0)
    for start in range(0, num_edges, EDGE_CHUNK):
        end = start + EDGE_CHUNK
        as_.index_add_(0, row[start:end], s[col[start:end]])

    tr_s_a_s = (s * as_).sum()
    d_dot_s = deg @ s
    null_term = (d_dot_s ** 2).sum() / (2.0 * m)
    spectral_loss = -(tr_s_a_s - null_term) / (2.0 * m)

    cluster_sizes = s.sum(dim=0)
    collapse_loss = (math.sqrt(n_clusters) / num_nodes) * torch.norm(cluster_sizes, p=2) - 1.0

    total = spectral_loss + reg_weight * collapse_loss
    return total, spectral_loss, collapse_loss


def pick_precision(requested: str):
    if requested == "fp32" or DEVICE.type != "cuda":
        return None, False
    if requested == "bf16":
        return torch.bfloat16, False
    if requested == "fp16":
        return torch.float16, True
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


@dataclass
class HParams:
    lr: float
    hidden_dim: int
    dropout: float
    reg_weight: float


def train_one_subgraph(
    x: torch.Tensor, edge_index: torch.Tensor, n_clusters: int, hp: HParams,
    autocast_dtype, use_scaler: bool, epochs: int, patience: int,
) -> torch.Tensor:
    """Trains DMoN on ONE subgraph (could be the full graph at depth 0, or a
    small induced subgraph deeper in the recursion) and returns the hard
    cluster assignment as a (num_nodes_in_this_subgraph,) LongTensor of
    LOCAL indices (0..n_clusters-1) — caller maps these back to global node
    ids and decides whether to recurse into each resulting group.
    """
    set_seed()
    num_nodes = x.size(0)
    model = DMoNEncoder(x.size(1), hp.hidden_dim, n_clusters, hp.dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_scaler)

    row, col = edge_index
    m = max(edge_index.size(1) / 2.0, 1.0)  # avoid div-by-zero on an edgeless subgraph
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)

    best_loss = float("inf")
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype,
                             enabled=autocast_dtype is not None):
            h = model.encode(x, edge_index)
            loss, _, _ = checkpoint(
                cluster_and_loss, h, model.cluster_head, row, col, deg,
                num_nodes, n_clusters, m, hp.reg_weight, use_reentrant=False,
            )
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss - 1e-4:
            best_loss = loss_val
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype,
                                          enabled=autocast_dtype is not None):
        s = model(x, edge_index)
    hard_assign = s.argmax(dim=-1).detach().cpu()
    del model, optimizer, scaler
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return hard_assign


# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------

@dataclass
class RecursionConfig:
    target_c: int
    min_size_to_split: int   # don't bother splitting a cluster this small or smaller
    max_depth: int
    epochs: int
    patience: int


def recursive_split(
    x_full: torch.Tensor, edge_index_full: torch.Tensor, node_ids: torch.Tensor,
    hp: HParams, cfg: RecursionConfig, autocast_dtype, use_scaler: bool,
    full_num_nodes: int, depth: int, path: tuple, tree_log: list, leaf_id_counter: list,
) -> list[tuple[int, torch.Tensor]]:
    """Returns a list of (leaf_id, global_node_ids) for every leaf under this node."""
    size = node_ids.numel()

    if size <= cfg.min_size_to_split or depth >= cfg.max_depth:
        leaf_id = leaf_id_counter[0]
        leaf_id_counter[0] += 1
        tree_log.append({"type": "leaf", "leaf_id": leaf_id, "depth": depth,
                          "path": list(path), "size": size})
        return [(leaf_id, node_ids)]

    # Cap C for this subgraph: don't ask for more clusters than makes sense
    # for its size (also avoids C > N, which is meaningless).
    effective_c = max(2, min(cfg.target_c, size // 4))

    # node_ids is kept on CPU throughout the recursion (cheap bookkeeping,
    # avoids pinning GPU memory across levels) — move a copy to the graph
    # tensors' device just for this extraction, so subgraph()/indexing don't
    # hit a CPU/CUDA mismatch.
    node_ids_dev = node_ids.to(edge_index_full.device)
    sub_edge_index = subgraph(node_ids_dev, edge_index_full, relabel_nodes=True,
                               num_nodes=full_num_nodes)[0]
    x_sub = x_full[node_ids_dev]

    hard_assign_local = train_one_subgraph(
        x_sub, sub_edge_index, effective_c, hp, autocast_dtype, use_scaler,
        cfg.epochs, cfg.patience,
    )
    del x_sub, sub_edge_index
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    used = hard_assign_local.unique().tolist()
    tree_log.append({"type": "split", "depth": depth, "path": list(path),
                      "size": size, "target_c": effective_c, "used_clusters": len(used)})

    if len(used) <= 1:
        # Didn't actually split (collapsed to one group) — treat as a leaf
        # rather than recursing forever into an identical-sized "child".
        leaf_id = leaf_id_counter[0]
        leaf_id_counter[0] += 1
        tree_log.append({"type": "leaf", "leaf_id": leaf_id, "depth": depth,
                          "path": list(path), "size": size,
                          "note": "no further split (single cluster)"})
        return [(leaf_id, node_ids)]

    leaves = []
    for c in used:
        members_local = (hard_assign_local == c).nonzero(as_tuple=True)[0]
        members_global = node_ids[members_local]
        leaves.extend(recursive_split(
            x_full, edge_index_full, members_global, hp, cfg, autocast_dtype,
            use_scaler, full_num_nodes, depth + 1, path + (int(c),), tree_log,
            leaf_id_counter,
        ))
    return leaves


def summarize_leaves(leaves: list[tuple[int, torch.Tensor]], num_nodes: int):
    sizes = [ids.numel() for _, ids in leaves]
    assignment = torch.full((num_nodes,), -1, dtype=torch.long)
    for leaf_id, ids in leaves:
        assignment[ids] = leaf_id

    unassigned = int((assignment == -1).sum())
    stats = {
        "leaf_count": len(leaves),
        "total_nodes_covered": num_nodes - unassigned,
        "unassigned_nodes": unassigned,  # should be 0 -- every node ends up in exactly one leaf
        "size_min": min(sizes) if sizes else None,
        "size_max": max(sizes) if sizes else None,
        "size_mean": round(statistics.mean(sizes), 2) if sizes else None,
        "size_median": statistics.median(sizes) if sizes else None,
        "size_stdev": round(statistics.stdev(sizes), 2) if len(sizes) > 1 else None,
    }
    return assignment, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "smoke"], default="smoke")
    parser.add_argument("--graph-path", default=GRAPH_PATH)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")

    # Hyperparameters -- defaults are a config that reliably avoided collapse
    # in earlier smoke testing (low lr). Override with whatever your modest-C
    # sweep (DMoNTrain_gpu_ready.py --n-clusters 128) finds is best.
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--reg-weight", type=float, default=0.5)

    # Recursion shape
    parser.add_argument("--target-c", type=int, default=128,
                         help="Cluster count to aim for at each split (capped down for small subgraphs).")
    parser.add_argument("--min-size-to-split", type=int, default=None,
                         help="Stop recursing into a group at or below this size. "
                              "Default: 4x target-c, so an average leaf has room to be meaningful.")
    parser.add_argument("--max-depth", type=int, default=6,
                         help="Safety cap on recursion depth.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--assignment-path", default="leaf_assignment.pt")
    parser.add_argument("--tree-path", default="recursion_tree.jsonl")
    parser.add_argument("--stats-path", default="leaf_stats.json")
    args = parser.parse_args()

    set_seed()

    if args.mode == "smoke":
        graph = make_smoke_graph()
        print(f"[smoke mode] synthetic graph: {graph.num_nodes} nodes, "
              f"{graph.edge_index.size(1)} directed edges")
    else:
        if DEVICE.type != "cuda" and not args.allow_cpu:
            raise SystemExit(
                "Refusing --mode full on CPU without --allow-cpu. Recursive "
                "training on small subgraphs is much lighter than the flat "
                "run, so CPU may actually work here -- but it'll still be "
                "slow at 103K nodes. Pass --allow-cpu to try anyway."
            )
        graph = load_graph(args.graph_path)
        print(f"Loaded graph: {graph.x.size(0)} nodes, {graph.edge_index.size(1)} "
              f"directed edges, {graph.x.size(1)}-dim features.")

    min_size_to_split = args.min_size_to_split or (4 * args.target_c)
    hp = HParams(lr=args.lr, hidden_dim=args.hidden_dim, dropout=args.dropout,
                 reg_weight=args.reg_weight)
    cfg = RecursionConfig(target_c=args.target_c, min_size_to_split=min_size_to_split,
                           max_depth=args.max_depth, epochs=args.epochs, patience=args.patience)
    autocast_dtype, use_scaler = pick_precision(args.precision)
    print(f"Device: {DEVICE} | precision: {autocast_dtype if autocast_dtype else 'fp32'} | "
          f"target_c={args.target_c} | min_size_to_split={min_size_to_split} | "
          f"max_depth={args.max_depth}")
    print(f"hparams: lr={hp.lr} hidden_dim={hp.hidden_dim} dropout={hp.dropout} "
          f"reg_weight={hp.reg_weight}")

    x_full = graph.x.to(DEVICE) if DEVICE.type == "cuda" else graph.x
    edge_index_full = graph.edge_index.to(DEVICE) if DEVICE.type == "cuda" else graph.edge_index
    all_node_ids = torch.arange(graph.num_nodes)

    tree_log: list = []
    leaf_id_counter = [0]
    t0 = time.time()
    leaves = recursive_split(
        x_full, edge_index_full, all_node_ids, hp, cfg, autocast_dtype, use_scaler,
        graph.num_nodes, depth=0, path=(), tree_log=tree_log, leaf_id_counter=leaf_id_counter,
    )
    elapsed = time.time() - t0

    assignment, stats = summarize_leaves(leaves, graph.num_nodes)
    stats["elapsed_sec"] = round(elapsed, 2)
    stats["max_depth_reached"] = max((entry["depth"] for entry in tree_log), default=0)

    torch.save(assignment, args.assignment_path)
    with open(args.tree_path, "w") as f:
        for entry in tree_log:
            f.write(json.dumps(entry) + "\n")
    with open(args.stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Leaf count: {stats['leaf_count']}")
    print(f"Leaf sizes: min={stats['size_min']} max={stats['size_max']} "
          f"mean={stats['size_mean']} median={stats['size_median']} stdev={stats['size_stdev']}")
    print(f"Every node assigned exactly once: {stats['unassigned_nodes'] == 0} "
          f"({stats['unassigned_nodes']} unassigned)")
    print(f"\nSaved: {args.assignment_path} (per-node leaf id), "
          f"{args.tree_path} (full recursion tree), {args.stats_path} (summary stats).")
    print("Compare leaf_count and the size distribution directly against Leiden's "
          "own leaf-level numbers for the lab notes.")


if __name__ == "__main__":
    main()
