"""
Step 10 / Step 14 — "diff the configs of all runs to prove the grouping
method is the only thing that changed."

Each group_*.py script should save a small run_config.json alongside its
communities file (group_kcore.py and group_leiden_flat.py already do this —
add the same two lines to group_leiden.py and the DMoN script). This script
diffs any set of them and prints exactly what varies.

Usage:
    python diff_configs.py leiden_run_config.json kcore_run_config.json flat_leiden_run_config.json
"""
import json, argparse


def diff_configs(config_paths):
    configs = {}
    for path in config_paths:
        with open(path) as f:
            configs[path] = json.load(f)

    all_keys = set()
    for c in configs.values():
        all_keys.update(c.keys())

    print(f"{'key':<25}" + "".join(f"{p:<30}" for p in config_paths))
    differing_keys = []
    for key in sorted(all_keys):
        values = [str(configs[p].get(key, '—')) for p in config_paths]
        if len(set(values)) > 1:
            differing_keys.append(key)
        print(f"{key:<25}" + "".join(f"{v:<30}" for v in values))

    print("\nKeys that differ across runs:", differing_keys or "(none)")
    # "method", its own seed, and its own file names are *expected* to differ
    # — that's the whole point. Anything else differing means something besides
    # the grouping method changed between runs.
    expected_to_differ = {"method", "seed", "communities_file", "max_community_size"}
    unexpected = set(differing_keys) - expected_to_differ
    if unexpected:
        print(f"⚠️  Unexpected differences beyond the grouping method: {unexpected} — check before trusting the comparison.")
    else:
        print("✅ Only the grouping method (and its own seed/file names) differs — comparison is clean.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_paths', nargs='+', help='e.g. leiden_run_config.json kcore_run_config.json flat_leiden_run_config.json')
    args = parser.parse_args()
    diff_configs(args.config_paths)
