"""
Benchmark H (Leiden arm) -- 10 runs, different seeds. For each run, measure:
  - quality:  modularity of the final-level partition
  - time:     wall-clock seconds for hierarchical_leiden() itself
  - cost:     process RSS memory (MB) after the run
Then report mean/std across the 10 runs for quality and time (the std on
quality is the same "wobble" measured in H4, now paired with cost).

Usage (Colab or locally, CPU only -- Leiden doesn't need a GPU):
    python benchmark_leiden.py --graph-path entity_graph_full.pt
    python benchmark_leiden.py --seeds 1 2 3 4 5 6 7 8 9 10   # explicit seeds
"""

import argparse
import statistics as stats

from graspologic.partition import hierarchical_leiden

from bench_common import (
    load_graph_undirected,
    process_rss_mb,
    modularity_from_assignments,
    Timer,
    save_benchmark_json,
)


def run_one(G, seed: int) -> dict:
    with Timer() as t:
        communities = hierarchical_leiden(G, random_seed=seed)

    assignments = {}
    for c in communities:
        assignments.setdefault(str(c.node), []).append(
            {"level": c.level, "cluster": c.cluster, "is_final": c.is_final_cluster}
        )

    modularity, n_comms = modularity_from_assignments(G, assignments)
    rss_mb = process_rss_mb()

    return {
        "seed": seed,
        "elapsed_sec": round(t.elapsed, 4),
        "modularity": modularity,
        "num_final_communities": n_comms,
        "rss_mb": round(rss_mb, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-path", default="entity_graph_full.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", default="leiden_benchmark.json")
    args = ap.parse_args()

    print(f"Loading graph from {args.graph_path} ...")
    G = load_graph_undirected(args.graph_path)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    rows = []
    print(f"{'seed':>5} | {'time_sec':>9} | {'modularity':>10} | {'n_comms':>8} | {'rss_mb':>8}")
    for seed in args.seeds:
        row = run_one(G, seed)
        rows.append(row)
        print(f"{row['seed']:>5} | {row['elapsed_sec']:>9.3f} | {row['modularity']:>10.4f} | "
              f"{row['num_final_communities']:>8} | {row['rss_mb']:>8.1f}")

    times = [r["elapsed_sec"] for r in rows]
    quals = [r["modularity"] for r in rows]
    print(f"\nTime (s):       mean={stats.mean(times):.3f}  std={stats.pstdev(times):.3f}  "
          f"min={min(times):.3f}  max={max(times):.3f}")
    print(f"Modularity:     mean={stats.mean(quals):.4f}  std={stats.pstdev(quals):.4f}  "
          f"min={min(quals):.4f}  max={max(quals):.4f}")

    save_benchmark_json(args.out, "leiden", rows)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
