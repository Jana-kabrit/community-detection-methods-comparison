"""
Benchmark H (DMoN arm) -- 10 runs, different seeds, run in Colab (needs GPU).

This is a DRIVER: it calls recursive_dmon.py once per seed as a subprocess
(same command you were already running by hand for H4, just looped and
timed automatically), then reads back each run's stats.json -- which now
also contains modularity and gpu_peak_mb, from the small instrumentation
added to recursive_dmon.py -- and aggregates all 10 into one summary, same
shape as leiden_benchmark.json / kcore_benchmark.json.

Requires recursive_dmon.py (the updated version, with modularity_of_leaves
and gpu_peak_mb tracking) to be in the same directory.

Usage (Colab, after entity_graph_full.pt is in the working directory):
    python benchmark_dmon.py --graph-path entity_graph_full.pt

    # override any recursive_dmon.py setting, e.g.:
    python benchmark_dmon.py --graph-path entity_graph_full.pt \\
        --target-c 128 --min-size-to-split 256 --epochs 200

This can take a while (10x a real DMoN run) -- on a free-tier Colab GPU,
budget roughly 10x whatever a single H4 run took you.
"""

import argparse
import json
import statistics as stats
import subprocess
import sys
import time


def run_one_seed(seed: int, args, run_dir: str) -> dict:
    assignment_path = f"{run_dir}/dmon_bench_seed{seed}.pt"
    tree_path = f"{run_dir}/dmon_bench_tree{seed}.jsonl"
    stats_path = f"{run_dir}/dmon_bench_stats{seed}.json"

    cmd = [
        sys.executable, "recursive_dmon.py",
        "--mode", "full",
        "--graph-path", args.graph_path,
        "--seed", str(seed),
        "--target-c", str(args.target_c),
        "--min-size-to-split", str(args.min_size_to_split),
        "--max-depth", str(args.max_depth),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--assignment-path", assignment_path,
        "--tree-path", tree_path,
        "--stats-path", stats_path,
    ]
    if args.allow_cpu:
        cmd.append("--allow-cpu")

    print(f"\n{'='*70}\nSeed {seed}: {' '.join(cmd)}\n{'='*70}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    wall_elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        raise RuntimeError(f"recursive_dmon.py failed for seed {seed} "
                            f"(exit code {result.returncode}) -- see output above.")

    with open(stats_path) as f:
        run_stats = json.load(f)

    return {
        "seed": seed,
        "elapsed_sec": run_stats["elapsed_sec"],       # timed inside recursive_dmon.py
        "wall_elapsed_sec": round(wall_elapsed, 2),    # includes subprocess/import overhead
        "modularity": run_stats["modularity"],
        "num_final_communities": run_stats["leaf_count"],
        "gpu_peak_mb": run_stats.get("gpu_peak_mb"),
        "size_mean": run_stats.get("size_mean"),
        "size_stdev": run_stats.get("size_stdev"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-path", default="entity_graph_full.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--target-c", type=int, default=128)
    ap.add_argument("--min-size-to-split", type=int, default=256)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--allow-cpu", action="store_true",
                     help="Only for testing without a GPU -- will be slow at full scale.")
    ap.add_argument("--run-dir", default=".", help="Where per-seed run files are written.")
    ap.add_argument("--out", default="dmon_benchmark.json")
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        row = run_one_seed(seed, args, args.run_dir)
        rows.append(row)

    print(f"\n\n{'='*70}\nSUMMARY -- {len(rows)} DMoN runs\n{'='*70}")
    print(f"{'seed':>5} | {'time_sec':>9} | {'modularity':>10} | {'n_comms':>8} | {'gpu_peak_mb':>11}")
    for r in rows:
        gpu_str = f"{r['gpu_peak_mb']:.1f}" if r["gpu_peak_mb"] is not None else "n/a"
        print(f"{r['seed']:>5} | {r['elapsed_sec']:>9.2f} | {r['modularity']:>10.4f} | "
              f"{r['num_final_communities']:>8} | {gpu_str:>11}")

    times = [r["elapsed_sec"] for r in rows]
    quals = [r["modularity"] for r in rows]
    print(f"\nTime (s):       mean={stats.mean(times):.2f}  std={stats.pstdev(times):.2f}  "
          f"min={min(times):.2f}  max={max(times):.2f}")
    print(f"Modularity:     mean={stats.mean(quals):.4f}  std={stats.pstdev(quals):.4f}  "
          f"min={min(quals):.4f}  max={max(quals):.4f}")
    gpu_vals = [r["gpu_peak_mb"] for r in rows if r["gpu_peak_mb"] is not None]
    if gpu_vals:
        print(f"GPU peak (MB):  mean={stats.mean(gpu_vals):.1f}  "
              f"min={min(gpu_vals):.1f}  max={max(gpu_vals):.1f}")

    with open(args.out, "w") as f:
        json.dump({"method": "dmon", "runs": rows}, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
