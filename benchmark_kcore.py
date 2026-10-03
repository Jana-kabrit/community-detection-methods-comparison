"""
Benchmark H (k-core arm) -- 10 runs. k-core has NO randomness (core_number,
connected_components and the BFS chunking below are all deterministic given
a fixed graph), so this is the "control": modularity should be identical
every run (std = 0) and time should only vary due to ordinary system noise,
not the algorithm itself. Still run 10 times as requested, both to fill out
the same table shape as Leiden/DMoN and because it's a second confirmation
(after H4) that nothing here is secretly nondeterministic.

NOTE ON THE is_final FIX: SA's original group_kcore.py sets
`is_final_level = (level == len(k_values) - 1)` -- true only at the single
deepest level GLOBALLY, so almost every node (whose own path stops earlier)
never gets an is_final entry at all (H4 found only 39/103,056 nodes marked
final in the reference file). This script fixes that: each node's own LAST
path entry is marked final, matching how Leiden's is_final_cluster and
DMoN's leaf assignment both work (every node has exactly one final
community). See the message already sent to SA about this bug's likely
impact on H3 too.

Usage:
    python benchmark_kcore.py --graph-path entity_graph_full.pt
"""

import argparse
import statistics as stats

import networkx as nx

from bench_common import (
    load_graph_undirected,
    process_rss_mb,
    modularity_from_assignments,
    Timer,
    save_benchmark_json,
)

MAX_COMMUNITY_SIZE = 50  # same placeholder bound as group_kcore.py


def split_oversized(nodes_in_component, subG, max_size):
    nodes_in_component = sorted(nodes_in_component)
    start = nodes_in_component[0]
    bfs_order = [start] + [v for _, v in nx.bfs_edges(subG, start)]
    remaining = set(nodes_in_component) - set(bfs_order)
    bfs_order += sorted(remaining)
    return [bfs_order[i:i + max_size] for i in range(0, len(bfs_order), max_size)]


def run_one(G, core_numbers, k_values) -> tuple[dict, dict]:
    with Timer() as t:
        assignments = {}
        for level, k in enumerate(k_values):
            nodes_at_k = [n for n, c in core_numbers.items() if c >= k]
            subG = G.subgraph(nodes_at_k)
            components = list(nx.connected_components(subG))

            cluster_id = 0
            for comp in components:
                chunks = (split_oversized(comp, G.subgraph(comp), MAX_COMMUNITY_SIZE)
                          if len(comp) > MAX_COMMUNITY_SIZE else [sorted(comp)])
                for chunk in chunks:
                    for node in chunk:
                        assignments.setdefault(str(node), []).append(
                            {"level": level, "cluster": cluster_id, "is_final": False}
                        )
                    cluster_id += 1

        # --- the fix: mark each node's own LAST path entry as final ---
        for path in assignments.values():
            path[-1]["is_final"] = True

    modularity, n_comms = modularity_from_assignments(G, assignments)
    rss_mb = process_rss_mb()

    return {
        "elapsed_sec": round(t.elapsed, 4),
        "modularity": modularity,
        "num_final_communities": n_comms,
        "rss_mb": round(rss_mb, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-path", default="entity_graph_full.pt")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--out", default="kcore_benchmark.json")
    args = ap.parse_args()

    print(f"Loading graph from {args.graph_path} ...")
    G = load_graph_undirected(args.graph_path)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Computing core numbers...")
    core_numbers = nx.core_number(G)
    k_values = sorted(set(core_numbers.values()))
    print(f"{len(k_values)} distinct k-shells: {k_values}\n")

    rows = []
    print(f"{'run':>5} | {'time_sec':>9} | {'modularity':>10} | {'n_comms':>8} | {'rss_mb':>8}")
    for i in range(1, args.n_runs + 1):
        row = run_one(G, core_numbers, k_values)
        row["seed"] = None  # no randomness -- included for schema parity with the other two
        row["run_index"] = i
        rows.append(row)
        print(f"{i:>5} | {row['elapsed_sec']:>9.3f} | {row['modularity']:>10.4f} | "
              f"{row['num_final_communities']:>8} | {row['rss_mb']:>8.1f}")

    times = [r["elapsed_sec"] for r in rows]
    quals = [r["modularity"] for r in rows]
    print(f"\nTime (s):       mean={stats.mean(times):.3f}  std={stats.pstdev(times):.3f}  "
          f"min={min(times):.3f}  max={max(times):.3f}")
    print(f"Modularity:     mean={stats.mean(quals):.4f}  std={stats.pstdev(quals):.4f}  "
          f"(should be ~0 -- k-core is deterministic)")

    save_benchmark_json(args.out, "kcore", rows)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
