"""
Shared helpers for the H-benchmark scripts (benchmark_leiden.py,
benchmark_kcore.py, benchmark_dmon.py / aggregate_benchmarks.py).

Keeps the "quality" and "cost" definitions IDENTICAL across all three
methods, so the final comparison is apples-to-apples:

QUALITY  -> modularity of each run's FINAL-level partition (the
            is_final:true clusters). Modularity is unsupervised (no
            ground truth needed) and is the same objective DMoN itself
            optimizes, which makes it a fair, comparable score across
            all three very different algorithms.
COST     -> wall-clock time (seconds) for that single run, process RSS
            memory (MB) at the end of the run, and — for DMoN only —
            peak GPU memory (MB), since that's the resource DMoN spends
            that Leiden/k-core don't.
"""

import json
import time

import networkx as nx
import psutil
import torch
from torch_geometric.utils import to_networkx


def load_graph_undirected(path: str = "entity_graph_full.pt") -> nx.Graph:
    """Loads the entity graph and returns an undirected networkx Graph with
    self-loops removed -- the same preprocessing every one of the three
    grouping scripts already does before running its algorithm."""
    data = torch.load(path, weights_only=False)
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def process_rss_mb() -> float:
    """Current resident-memory footprint of this process, in MB. This is a
    snapshot (not a delta), so it's most meaningful read right after the
    method being measured has finished its heaviest allocation."""
    return psutil.Process().memory_info().rss / (1024 ** 2)


def modularity_from_assignments(G: nx.Graph, assignments: dict) -> tuple[float, int]:
    """Computes networkx modularity of the FINAL-level partition encoded in
    an `assignments` dict of the shared schema:
        {node_id: [{"level": int, "cluster": int, "is_final": bool}, ...]}

    Returns (modularity, number_of_final_communities).

    Only nodes present in G AND marked is_final in some path entry are
    used -- this mirrors how a downstream consumer (the chassis) would
    actually read "this node's final community" off the file.
    """
    communities: dict[tuple, set] = {}
    node_lookup = {str(n): n for n in G.nodes()}  # schema keys are strings

    for nid_str, path in assignments.items():
        node = node_lookup.get(nid_str)
        if node is None:
            continue  # node not in this graph (shouldn't normally happen)
        for p in path:
            if p.get("is_final"):
                key = (p["level"], p["cluster"])
                communities.setdefault(key, set()).add(node)
                break  # a node should have exactly one final entry

    comm_sets = list(communities.values())
    if len(comm_sets) < 2:
        raise ValueError(
            f"Only {len(comm_sets)} final community(ies) found -- modularity "
            "needs at least 2. Check that `is_final` is actually being set "
            "per-node (not just at one global level)."
        )
    # nx.modularity requires `communities` to be an EXACT partition of every
    # node in G. A few methods (Leiden drops isolate nodes; see graspologic's
    # own warning about this) don't cover 100% of nodes, so score modularity
    # on the induced subgraph of only the nodes that actually got a final
    # community -- consistent across methods, and the gap is always reported
    # so it stays visible rather than silently changing the denominator.
    covered = set().union(*comm_sets)
    G_scored = G.subgraph(covered) if len(covered) != G.number_of_nodes() else G
    q = nx.algorithms.community.modularity(G_scored, comm_sets)
    return q, len(comm_sets)


def modularity_from_leaf_tensor(G: nx.Graph, leaf_assignment) -> tuple[float, int]:
    """Same as modularity_from_assignments, but for DMoN's native output
    format: a (num_nodes,) LongTensor where entry i is node i's leaf id
    (recursive_dmon.py's leaf_assignment.pt). Node i in the tensor
    corresponds to networkx node i (to_networkx preserves index order).
    """
    communities: dict[int, set] = {}
    for node_idx in G.nodes():
        leaf_id = int(leaf_assignment[node_idx])
        if leaf_id < 0:
            continue  # unassigned (shouldn't happen -- summarize_leaves checks this)
        communities.setdefault(leaf_id, set()).add(node_idx)

    comm_sets = list(communities.values())
    if len(comm_sets) < 2:
        raise ValueError(f"Only {len(comm_sets)} leaf community(ies) found.")
    covered = set().union(*comm_sets)
    G_scored = G.subgraph(covered) if len(covered) != G.number_of_nodes() else G
    q = nx.algorithms.community.modularity(G_scored, comm_sets)
    return q, len(comm_sets)


class Timer:
    """Tiny context manager: `with Timer() as t: ...` then `t.elapsed`."""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0


def save_benchmark_json(path: str, method: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump({"method": method, "runs": rows}, f, indent=2)
