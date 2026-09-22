"""
Train DMoN (Deep Modularity Networks) on the entity graph — GPU-ready revision.

WHY THIS VERSION EXISTS
------------------------
The original DMoNTrain.py is logically correct (loss math, model, sweep design
all check out). But it has no memory safety margin, and this graph is bigger
than it looks: with C = 7,206 target clusters and N = 103,056 nodes, the soft
assignment matrix S is N x C = ~743M floats = ~2.97 GB in fp32, and a single
training step needs several tensors of that same size alive at once (logits,
softmax(logits), the scatter-added A@S buffer, plus autograd's saved copies
for backward). That lands you at roughly 12-18 GB of peak memory for ONE
epoch of the *smallest* config in the grid (hidden_dim=128) — confirmed
empirically: a same-shape synthetic graph OOM-killed on a 7.8 GB-RAM,
CPU-only sandbox at that exact config. This will happen on any machine
without real headroom, GPU included, if you don't manage precision and
cleanup deliberately.

This revision changes HOW it runs, not WHAT it computes:
  1. Mixed precision (bf16 if the GPU supports it, else fp16 + GradScaler)
     roughly halves the big (N, C) tensors' memory footprint.
  2. A rough peak-memory estimate is printed before each config, so you can
     see trouble coming instead of finding out via a crash 40 configs in.
  3. Each config in the sweep is wrapped in a try/except for OOM: a failed
     config is logged and skipped, not fatal to the other 80.
  4. Results are appended to a .jsonl file after every config (not just at
     the end), so a crash or a killed Colab runtime doesn't lose completed
     work.
  5. GPU memory is explicitly freed (del + empty_cache) between configs,
     since this loop creates ~81 models/optimizers in one process.
  6. A --mode smoke option runs the exact same code path on a tiny synthetic
     graph (no entity_graph_full.pt needed) so you can validate the pipeline
     end-to-end in seconds before spending real GPU time on the full sweep.
  7. Running on CPU is still allowed but the script refuses by default for
     the real graph and tells you why (see point above) — pass --allow-cpu
     if you really want to try it anyway (not recommended for N=103,056).

HOW TO RUN
----------
Quick sanity check, no data file needed, runs in seconds on CPU or GPU:
    python DMoNTrain_gpu_ready.py --mode smoke

Real run, needs entity_graph_full.pt next to this script, needs a GPU:
    python DMoNTrain_gpu_ready.py --mode full

Limit the grid while you're still validating (e.g. first 3 configs only):
    python DMoNTrain_gpu_ready.py --mode full --max-configs 3

Force a specific precision if autodetection picks the wrong thing:
    python DMoNTrain_gpu_ready.py --mode full --precision fp16

Sizing a Colab/cluster GPU: budget roughly 6x the S-matrix size in fp32
(or 3x in mixed precision) as a rule of thumb for peak memory at a given
hidden_dim — the script prints this estimate per config so you don't have
to do the arithmetic by hand. If you're on a 16 GB GPU (e.g. Colab T4),
start with the smallest hidden_dim in the grid and watch `nvidia-smi` before
trusting the larger ones; a 40 GB+ GPU (A100) should clear the whole default
grid without issue.

Inputs
------
entity_graph_full.pt   PyG Data object: x (103,056 x 384), edge_index (2 x 122,417),
                        relation (list[str], per-edge type — not used by vanilla DMoN)

Cluster count C is fixed to 7,206 (Leiden's leaf-level community count from the
Week 2 run, seed 42) for a fair first comparison against Leiden. Vary C later
by editing N_CLUSTERS or looping the sweep over a list of C values.

This uses the *sparse* DMoN formulation (Tsitsulin et al., 2020, "Graph
Clustering with Graph Neural Networks"), not PyG's built-in DMoNPooling —
that layer needs a dense N x N adjacency, which is infeasible at N=103,056
(that's ~10^10 entries). Everything here works off edge_index directly, so
memory scales with edges (122k) and with N x C for the soft assignment
matrix S (103,056 x 7,206 ~= 743M floats ~= 3 GB in fp32 — this is the
memory bottleneck, not the graph itself; drop to fp16/bf16 or a smaller
hidden size first if you OOM).
"""

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Must be set before CUDA initializes. Reduces OOM from allocator fragmentation
# (won't rescue a genuine over-capacity need, but it's free and PyTorch's own
# error message recommends it — no reason not to have it on).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, remove_self_loops, degree

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GRAPH_PATH = "entity_graph_full.pt"
N_CLUSTERS = 7206          # Leiden leaf-level count, seed 42 — fixed for the fair-comparison pass
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
PATIENCE = 20               # early stopping on total loss plateau
SEED = 42
RESULTS_PATH = "dmon_sweep_results.jsonl"   # incremental checkpoint, one JSON line per config
ASSIGNMENTS_DIR = "cluster_assignments"     # per-config saved community assignments (node -> cluster id)

# The real memory spike turned out NOT to be the (N, C) S matrix itself — it's
# `s[col]` inside the A@S scatter-add: col has one entry per EDGE, and with
# 241,602 (undirected) edges vs 103,056 nodes, that gathered intermediate is
# shape (num_edges, C), bigger than S. Doing the scatter-add in edge chunks
# instead of one call caps that transient at (EDGE_CHUNK, C) regardless of
# graph size. Verified numerically identical to the unchunked version
# (forward value, loss, and gradients all match to float precision) before
# shipping this. Lower this further (e.g. 20_000) if still tight on memory.
EDGE_CHUNK = 50_000

# Smoke-test graph shape (no relation to the real graph's numbers — just
# small enough to run instantly anywhere).
SMOKE_N_NODES = 2000
SMOKE_N_CLUSTERS = 40
SMOKE_AVG_DEGREE = 6


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graph(path: str = GRAPH_PATH) -> Data:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"'{path}' not found. This is produced by build_entity_graph.py "
            "(task 6/step 3) — run that first, or point --graph-path at wherever "
            "it landed. --mode smoke runs without this file if you just want to "
            "validate the pipeline."
        )
    data = torch.load(path, weights_only=False)

    # Community detection is on the undirected graph (matches how Leiden saw it).
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=data.x.size(0))

    data.edge_index = edge_index
    # `relation` (edge type strings) isn't used by vanilla DMoN — dropped here.
    if hasattr(data, "relation"):
        del data.relation
    return data


def make_smoke_graph(
    n_nodes: int = SMOKE_N_NODES,
    avg_degree: int = SMOKE_AVG_DEGREE,
    in_dim: int = 384,
) -> Data:
    """Small synthetic stand-in with the real graph's shape (features, undirected
    edges) but tiny enough to run the full code path in seconds. NOT a substitute
    for the real training run — only for validating that the pipeline itself works
    (no crashes, sane loss trajectory, no accidental collapse) before spending real
    GPU time on the actual 103K-node graph.
    """
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
# Model: GCN encoder -> soft cluster assignment head
# ---------------------------------------------------------------------------

class DMoNEncoder(nn.Module):
    """GCN encoder only — produces node embeddings h, shape (N, hidden_dim).
    Cheap: hidden_dim is at most 512, so h itself is never the memory problem.
    The cluster head (hidden_dim -> n_clusters=7,206) is kept separate and
    called through `cluster_and_loss` below, under gradient checkpointing —
    that's where the expensive (N, C) tensors live.
    """
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
        """Full soft assignment s = softmax(head(encode(x))). Used for eval only —
        training goes through cluster_and_loss (checkpointed) instead."""
        h = self.encode(x, edge_index)
        return F.softmax(self.cluster_head(h), dim=-1)


# ---------------------------------------------------------------------------
# Sparse DMoN loss: modularity term (via edge_index) + collapse regularizer
#
# cluster_and_loss is the memory-heavy part: it materializes several (N, C)
# tensors (logits, softmax output s, the scatter-added A@S buffer) that
# autograd would otherwise all keep alive simultaneously for backward. It's
# called through torch.utils.checkpoint in train_one_config, which discards
# these after the forward pass and recomputes them on demand during backward
# instead of holding them the whole time — trades some extra compute (cheap;
# this model is tiny) for meaningfully lower peak memory. This is what turns
# an OOM at hidden=128/C=7,206 on a 14.56GB GPU into (hopefully) a fit.
# ---------------------------------------------------------------------------

def cluster_and_loss(
    h: torch.Tensor,
    cluster_head: nn.Linear,
    row: torch.Tensor,
    col: torch.Tensor,
    deg: torch.Tensor,
    num_nodes: int,
    n_clusters: int,
    m: float,
    reg_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = cluster_head(h)
    s = F.softmax(logits, dim=-1)          # (N, C) soft cluster assignment
    del logits  # softmax's backward only needs its OUTPUT (s), not its input —
    # verified this frees ~1.5GB (bf16) / ~3GB (fp32) immediately rather than
    # sitting retained as a live local for the rest of this function's scope.

    # A @ S via scatter-add over edges (A is symmetric here since graph is undirected).
    # Chunked over edges: s[col] gathers one row of S per edge, so an unchunked
    # call materializes an (num_edges, C) tensor -- with more edges than nodes
    # here, that's bigger than S itself. Chunking caps it at (EDGE_CHUNK, C).
    # Mathematically identical to a single index_add_ call (verified).
    as_ = torch.zeros_like(s)
    num_edges = row.size(0)
    for start in range(0, num_edges, EDGE_CHUNK):
        end = start + EDGE_CHUNK
        as_.index_add_(0, row[start:end], s[col[start:end]])

    tr_s_a_s = (s * as_).sum()
    d_dot_s = deg @ s                      # (C,)
    null_term = (d_dot_s ** 2).sum() / (2.0 * m)

    spectral_loss = -(tr_s_a_s - null_term) / (2.0 * m)

    cluster_sizes = s.sum(dim=0)           # (C,)
    collapse_loss = (math.sqrt(n_clusters) / num_nodes) * torch.norm(cluster_sizes, p=2) - 1.0

    total = spectral_loss + reg_weight * collapse_loss
    # All three returned so the caller gets real metrics from THIS forward pass —
    # no second/duplicate forward needed just for logging.
    return total, spectral_loss, collapse_loss


# ---------------------------------------------------------------------------
# Memory estimation (rule of thumb, not exact) and precision selection
# ---------------------------------------------------------------------------

def estimate_peak_gb(num_nodes: int, n_clusters: int, bytes_per_elem: int) -> float:
    """Rough upper-bound estimate, WITHOUT gradient checkpointing's savings
    factored in (checkpointing is on by default now — see cluster_and_loss —
    so real usage should land below this number, we just don't have an exact
    multiplier for the reduction yet). 10x is calibrated against a real OOM:
    at hidden=128, C=7,206, bf16, a 14.56GB T4 reported ~15.3GB actually
    needed pre-checkpointing (this formula's old 6x multiplier undershot that
    by ~40%). Treat this as a conservative planning number, not a guarantee —
    always sanity-check the first config against `nvidia-smi` / the OOM
    message itself before trusting the rest of the grid.
    """
    return 10 * num_nodes * n_clusters * bytes_per_elem / 1e9


def pick_precision(requested: str) -> tuple[torch.dtype | None, bool]:
    """Returns (autocast_dtype_or_None, use_grad_scaler)."""
    if requested == "fp32" or DEVICE.type != "cuda":
        return None, False
    if requested == "bf16":
        return torch.bfloat16, False
    if requested == "fp16":
        return torch.float16, True
    # auto
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


# ---------------------------------------------------------------------------
# Single training run for one hyperparameter config
# ---------------------------------------------------------------------------

@dataclass
class HParams:
    lr: float
    hidden_dim: int
    dropout: float
    reg_weight: float


def train_one_config(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    hp: HParams,
    n_clusters: int,
    autocast_dtype,
    use_scaler: bool,
    config_id: int = 0,
    assignments_dir: str | None = None,
    verbose: bool = False,
) -> dict:
    set_seed()
    num_nodes = x.size(0)

    est_gb = estimate_peak_gb(num_nodes, n_clusters, 2 if autocast_dtype else 4)
    print(f"  estimated peak memory for this config: ~{est_gb:.1f} GB "
          f"({'mixed precision' if autocast_dtype else 'fp32'})")

    model = DMoNEncoder(
        in_dim=x.size(1),
        hidden_dim=hp.hidden_dim,
        n_clusters=n_clusters,
        dropout=hp.dropout,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_scaler)

    # Precompute graph-derived tensors once per config, not once per epoch.
    row, col = edge_index
    m = edge_index.size(1) / 2.0
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)

    best_loss = float("inf")
    best_epoch = -1
    patience_ctr = 0
    t0 = time.time()
    status = "ok"

    try:
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                h = model.encode(x, edge_index)
                # Checkpointed: the (N, C) tensors inside cluster_and_loss are
                # NOT kept for backward — they're recomputed on demand instead,
                # which is what keeps peak memory down at C=7,206.
                loss, spectral_loss, collapse_loss = checkpoint(
                    cluster_and_loss, h, model.cluster_head, row, col, deg,
                    num_nodes, n_clusters, m, hp.reg_weight,
                    use_reentrant=False,
                )

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            metrics = {
                "spectral_loss": spectral_loss.item(),
                "collapse_loss": collapse_loss.item(),
                "total_loss": loss.item(),
            }

            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch:3d} | total {metrics['total_loss']:.4f} "
                      f"| spectral {metrics['spectral_loss']:.4f} "
                      f"| collapse {metrics['collapse_loss']:.4f}")

            if metrics["total_loss"] < best_loss - 1e-4:
                best_loss = metrics["total_loss"]
                best_epoch = epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype,
                                 enabled=autocast_dtype is not None):
                s = model(x, edge_index)
            hard_assign = s.argmax(dim=-1)
            used_clusters = hard_assign.unique().numel()

        assignment_path = None
        if assignments_dir is not None:
            adir = Path(assignments_dir)
            adir.mkdir(parents=True, exist_ok=True)
            assignment_path = str(adir / f"config_{config_id:03d}.pt")
            # Plain (N,) long tensor on CPU: entry i is node i's assigned cluster id.
            # This is the actual community-detection OUTPUT, not just a loss number.
            torch.save(hard_assign.detach().cpu(), assignment_path)

    except torch.cuda.OutOfMemoryError as e:  # py3.11+/torch 2.x
        status = "oom"
        best_loss, best_epoch, used_clusters, assignment_path = None, None, None, None
        print(f"  !! OOM at this config, skipping. ({e})")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            status = "oom"
            best_loss, best_epoch, used_clusters, assignment_path = None, None, None, None
            print(f"  !! OOM at this config, skipping. ({e})")
        else:
            raise
    finally:
        elapsed = time.time() - t0
        del model, optimizer, scaler
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "hparams": asdict(hp),
        "status": status,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "used_clusters": used_clusters,
        "target_clusters": n_clusters,
        "elapsed_sec": round(elapsed, 2),
        "assignment_path": assignment_path,
    }


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------
# Starting grid — narrow or widen based on how H0's homophily result plays out.
# d = 1.06 is a solid gap, so this grid is not padded defensively; widen dropout
# and reg_weight ranges first if training collapses to a handful of clusters.

LR_GRID = [1e-2, 1e-3, 1e-4]
HIDDEN_GRID = [128, 256, 512]
DROPOUT_GRID = [0.0, 0.3, 0.5]
REG_WEIGHT_GRID = [0.1, 0.5, 1.0]


def run_sweep(
    data: Data,
    n_clusters: int,
    precision: str,
    max_configs: int | None,
    results_path: str,
    assignments_dir: str | None = None,
) -> list[dict]:
    x = data.x.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)

    autocast_dtype, use_scaler = pick_precision(precision)
    print(f"Device: {DEVICE} | precision: "
          f"{autocast_dtype if autocast_dtype else 'fp32'} | grad scaler: {use_scaler}")

    configs = list(itertools.product(LR_GRID, HIDDEN_GRID, DROPOUT_GRID, REG_WEIGHT_GRID))
    if max_configs is not None:
        configs = configs[:max_configs]
    print(f"Running {len(configs)} configs...")

    results = []
    results_file = Path(results_path)
    with results_file.open("a") as f:
        for i, (lr, hidden, dropout, reg_weight) in enumerate(configs):
            hp = HParams(lr=lr, hidden_dim=hidden, dropout=dropout, reg_weight=reg_weight)
            print(f"[{i+1}/{len(configs)}] lr={lr} hidden={hidden} dropout={dropout} reg={reg_weight}")
            result = train_one_config(x, edge_index, hp, n_clusters, autocast_dtype, use_scaler,
                                       config_id=i, assignments_dir=assignments_dir)
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()
            if result["status"] == "ok":
                print(f"  -> best_loss={result['best_loss']:.4f} "
                      f"used_clusters={result['used_clusters']}/{n_clusters} "
                      f"({result['elapsed_sec']}s) -> saved {result['assignment_path']}")
    print(f"\nAll results appended to {results_file.resolve()} (one JSON line per config).")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "smoke"], default="smoke",
                         help="'smoke' = tiny synthetic graph, no data file needed, "
                              "validates the pipeline in seconds. 'full' = the real "
                              "sweep on entity_graph_full.pt (needs a GPU).")
    parser.add_argument("--graph-path", default=GRAPH_PATH)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max-configs", type=int, default=None,
                         help="Limit the sweep to the first N configs (useful while validating).")
    parser.add_argument("--allow-cpu", action="store_true",
                         help="Allow --mode full to run on CPU. Not recommended: a same-shape "
                              "synthetic graph OOM'd on a 7.8GB CPU machine at the smallest grid "
                              "config. Fine for --mode smoke regardless.")
    parser.add_argument("--results-path", default=RESULTS_PATH)
    parser.add_argument("--n-clusters", type=int, default=None,
                         help="Override N_CLUSTERS (7,206) for --mode full. Use this to get a "
                              "real, completed sweep with saved community assignments on hardware "
                              "that can't fit 7,206 clusters yet -- NOT the fair Leiden-comparison "
                              "number for the project's headline result, just a real run today.")
    parser.add_argument("--assignments-dir", default=ASSIGNMENTS_DIR)
    args = parser.parse_args()

    set_seed()

    if args.mode == "smoke":
        graph = make_smoke_graph()
        n_clusters = SMOKE_N_CLUSTERS
        print(f"[smoke mode] synthetic graph: {graph.num_nodes} nodes, "
              f"{graph.edge_index.size(1)} directed edges, C={n_clusters}")
    else:
        if DEVICE.type != "cuda" and not args.allow_cpu:
            raise SystemExit(
                "Refusing to run --mode full on CPU: a same-shape synthetic graph "
                "OOM-killed on a 7.8GB CPU machine at the smallest config in this grid "
                "(hidden=128). Run this on a GPU, or pass --allow-cpu to try anyway "
                "(expect it to be extremely slow or to fail the same way)."
            )
        graph = load_graph(args.graph_path)
        n_clusters = args.n_clusters if args.n_clusters is not None else N_CLUSTERS
        if args.n_clusters is not None and args.n_clusters != N_CLUSTERS:
            print(f"NOTE: running with C={n_clusters}, overriding the fair-comparison "
                  f"C={N_CLUSTERS} (Leiden's leaf-level count). Real communities, real graph -- "
                  f"just not the number to use for the H3 headline table against Leiden.")
        print(f"Loaded graph: {graph.x.size(0)} nodes, {graph.edge_index.size(1)} directed edges "
              f"(undirected), {graph.x.size(1)}-dim features. Target C = {n_clusters}.")

    results = run_sweep(graph, n_clusters, args.precision, args.max_configs, args.results_path,
                         assignments_dir=args.assignments_dir)
    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: r["best_loss"])

    print(f"\n{len(ok_results)}/{len(results)} configs completed successfully.")
    print("\nTop 5 configs by best_loss:")
    for r in ok_results[:5]:
        print(r)

    if ok_results:
        best = ok_results[0]
        print(f"\nBest config's community assignment saved at: {best['assignment_path']}")
        print("Load it with: torch.load(that_path) -> a tensor of shape (num_nodes,) where "
              "entry i is node i's assigned cluster id. This is the actual DMoN output.")

    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print(f"\n{len(failed)} config(s) failed (OOM) — see {args.results_path} for details, "
              f"or re-run just those with a smaller hidden_dim / mixed precision.")


if __name__ == "__main__":
    main()
