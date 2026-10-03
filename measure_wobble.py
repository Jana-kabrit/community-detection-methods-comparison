"""
Experiment H4 — measure the wobble.

Given a folder of N assignment JSON files (all from the same grouping method,
different random seeds, same schema as leiden_communities.json /
kcore_communities.json / dmon_communities.json: {"assignments": {node_id:
[{"level": int, "cluster": int, "is_final": bool}, ...]}}), compute pairwise
Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) between
every pair of runs, per hierarchy level, and report summary stats.

Level convention: a node's label at comparison-level L is taken from the
deepest entry it has at level <= L (since a node that already stopped
splitting at an earlier level keeps that same community for all deeper
levels too — it was never subdivided further). Labels are (level, cluster)
pairs so that a level-0 cluster id is never confused with a level-1 one.

Usage:
    python measure_wobble.py <folder_of_run_jsons> [--method-name NAME] [--out-prefix PREFIX]
"""

import argparse
import glob
import itertools
import json
import os
import statistics as stats

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load_run(path):
    with open(path) as f:
        return json.load(f)["assignments"]


def all_levels(assignments):
    return sorted(set(p["level"] for v in assignments.values() for p in v))


def labels_at_level(assignments, level, all_node_ids):
    """For each node, its (level, cluster) label using the deepest entry <= `level`."""
    labels = []
    for nid in all_node_ids:
        path = assignments.get(nid, [])
        best = None
        for p in path:
            if p["level"] <= level and (best is None or p["level"] > best["level"]):
                best = p
        if best is None:
            raise ValueError(f"node {nid} has no entry at or below level {level}")
        labels.append((best["level"], best["cluster"]))
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_folder")
    ap.add_argument("--method-name", default=None)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    method_name = args.method_name or os.path.basename(args.run_folder.rstrip("/"))
    out_prefix = args.out_prefix or method_name

    run_files = sorted(glob.glob(os.path.join(args.run_folder, "*.json")))
    print(f"Method: {method_name}")
    print(f"Found {len(run_files)} runs: {[os.path.basename(f) for f in run_files]}")
    if len(run_files) < 2:
        raise SystemExit("Need at least 2 runs to measure wobble.")

    runs = [load_run(f) for f in run_files]

    # union of node ids across all runs (should be identical, but be safe)
    node_id_sets = [set(r.keys()) for r in runs]
    common_nodes = set.intersection(*node_id_sets)
    all_nodes_union = set.union(*node_id_sets)
    if common_nodes != all_nodes_union:
        print(f"WARNING: node sets differ across runs! common={len(common_nodes)}, "
              f"union={len(all_nodes_union)}. Using common nodes only.")
    node_ids = sorted(common_nodes)
    print(f"Comparing on {len(node_ids)} common nodes.")

    # levels: use the levels present in every run
    level_sets = [set(all_levels(r)) for r in runs]
    common_levels = sorted(set.intersection(*level_sets))
    print(f"Levels compared: {common_levels}")

    results = {}  # level -> {"ari": [...], "nmi": [...]}
    for level in common_levels:
        labelings = [labels_at_level(r, level, node_ids) for r in runs]
        # encode (level,cluster) tuples as ints for sklearn
        label_ints = []
        for lab in labelings:
            vocab = {}
            ints = []
            for t in lab:
                if t not in vocab:
                    vocab[t] = len(vocab)
                ints.append(vocab[t])
            label_ints.append(ints)

        aris, nmis = [], []
        for i, j in itertools.combinations(range(len(runs)), 2):
            aris.append(adjusted_rand_score(label_ints[i], label_ints[j]))
            nmis.append(normalized_mutual_info_score(label_ints[i], label_ints[j]))
        results[level] = {"ari": aris, "nmi": nmis}

    # summary table
    print("\nLevel | metric | mean | std | min | max | n_pairs")
    summary_rows = []
    for level in common_levels:
        for metric in ("ari", "nmi"):
            vals = results[level][metric]
            row = {
                "method": method_name,
                "level": level,
                "metric": metric,
                "mean": stats.mean(vals),
                "std": stats.pstdev(vals),
                "min": min(vals),
                "max": max(vals),
                "n_pairs": len(vals),
            }
            summary_rows.append(row)
            print(f"{level:5d} | {metric:6s} | {row['mean']:.4f} | {row['std']:.4f} | "
                  f"{row['min']:.4f} | {row['max']:.4f} | {row['n_pairs']}")

    out = {
        "method": method_name,
        "run_files": [os.path.basename(f) for f in run_files],
        "levels": common_levels,
        "raw": {str(level): results[level] for level in common_levels},
        "summary": summary_rows,
    }
    out_path = f"{out_prefix}_wobble.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
