"""
Combines leiden_benchmark.json + dmon_benchmark.json + kcore_benchmark.json
(from benchmark_leiden.py / benchmark_dmon.py / benchmark_kcore.py) into:
  - benchmark_summary.csv   -- mean/std of quality (modularity) and time per method
  - benchmark_boxplot.png   -- 2-panel box plot: modularity spread, time spread

Run this LAST, once all three *_benchmark.json files exist in the same folder
(Leiden and k-core can be produced locally/CPU; DMoN's has to come from Colab).

Usage:
    python aggregate_benchmarks.py
    python aggregate_benchmarks.py --dir /path/to/folder/with/the/three/jsons
"""

import argparse
import json
import os
import statistics as stats

import matplotlib.pyplot as plt

# Fixed categorical order (validated colorblind-safe triple), same convention
# as the H4 box plots: Leiden=blue, DMoN=orange, k-core=aqua.
COLORS = {"leiden": "#2a78d6", "dmon": "#eb6834", "kcore": "#1baf7a"}
LABELS = {"leiden": "Leiden", "dmon": "DMoN", "kcore": "k-core"}
ORDER = ["leiden", "dmon", "kcore"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize(method_key, data):
    if data is None:
        return None
    runs = data["runs"]
    quals = [r["modularity"] for r in runs]
    times = [r["elapsed_sec"] for r in runs]
    row = {
        "method": LABELS[method_key],
        "n_runs": len(runs),
        "modularity_mean": stats.mean(quals),
        "modularity_std": stats.pstdev(quals) if len(quals) > 1 else 0.0,
        "modularity_min": min(quals),
        "modularity_max": max(quals),
        "time_mean_sec": stats.mean(times),
        "time_std_sec": stats.pstdev(times) if len(times) > 1 else 0.0,
    }
    # cost fields differ by method (CPU RSS for leiden/kcore, GPU MB for dmon)
    if "rss_mb" in runs[0]:
        row["rss_mb_mean"] = stats.mean(r["rss_mb"] for r in runs)
    if runs[0].get("gpu_peak_mb") is not None:
        gpu_vals = [r["gpu_peak_mb"] for r in runs if r.get("gpu_peak_mb") is not None]
        if gpu_vals:
            row["gpu_peak_mb_mean"] = stats.mean(gpu_vals)
    return row, quals, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out-csv", default="benchmark_summary.csv")
    ap.add_argument("--out-png", default="benchmark_boxplot.png")
    args = ap.parse_args()

    datasets = {k: load(os.path.join(args.dir, f"{k}_benchmark.json")) for k in ORDER}
    missing = [k for k, v in datasets.items() if v is None]
    if missing:
        print(f"NOTE: missing {[f'{m}_benchmark.json' for m in missing]} in {args.dir} -- "
              f"proceeding with whatever is present. Run the matching benchmark_*.py "
              f"first to fill these in.")

    summaries = {}
    qual_by_method = {}
    time_by_method = {}
    for k in ORDER:
        result = summarize(k, datasets[k])
        if result is None:
            continue
        row, quals, times = result
        summaries[k] = row
        qual_by_method[k] = quals
        time_by_method[k] = times

    if not summaries:
        raise SystemExit("No benchmark JSONs found -- nothing to aggregate.")

    # --- CSV ---
    import csv
    fieldnames = sorted({f for row in summaries.values() for f in row.keys()})
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in ORDER:
            if k in summaries:
                writer.writerow(summaries[k])
    print(f"Saved: {args.out_csv}")

    print("\nSummary:")
    for k in ORDER:
        if k not in summaries:
            continue
        r = summaries[k]
        print(f"  {r['method']:8s}  modularity={r['modularity_mean']:.4f}±{r['modularity_std']:.4f}  "
              f"time={r['time_mean_sec']:.2f}±{r['time_std_sec']:.2f}s  (n={r['n_runs']})")

    # --- box plot: modularity (left) and time (right), one box per available method ---
    present = [k for k in ORDER if k in qual_by_method]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("#fcfcfb")

    for ax, data_by_method, title, ylabel in [
        (axes[0], qual_by_method, "Community quality (modularity)", "Modularity"),
        (axes[1], time_by_method, "Run time", "Seconds"),
    ]:
        ax.set_facecolor("#fcfcfb")
        box_data = [data_by_method[k] for k in present]
        bp = ax.boxplot(box_data, tick_labels=[LABELS[k] for k in present], patch_artist=True,
                         widths=0.5, medianprops={"color": "black", "linewidth": 1.5})
        for patch, k in zip(bp["boxes"], present):
            patch.set_facecolor(COLORS[k])
            patch.set_alpha(0.75)
            patch.set_edgecolor(COLORS[k])
        for element in ("whiskers", "caps"):
            for line in bp[element]:
                line.set_color("#57534e")
        ax.set_title(title, fontsize=12, color="#1c1917")
        ax.set_ylabel(ylabel, fontsize=10, color="#44403c")
        ax.tick_params(colors="#44403c")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d6d3d1")
        ax.spines["bottom"].set_color("#d6d3d1")
        ax.grid(axis="y", color="#e7e5e4", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle("10 runs per method — quality vs. cost", fontsize=13, color="#1c1917", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()
