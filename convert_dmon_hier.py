"""Convert Aseel's DMoN export into the same JSON format as leiden_communities.json.

Inputs (put in the project folder):
  leaf_assignment.pt      (node index -> leaf_id; the renamed leaf_assignment_pt zip)
  recursion_tree.jsonl    (leaf_id -> path in the hierarchy)
  leaf_stats.json
Output:
  dmon_communities.json   {"seed": ..., "assignments": {node: [{"level","cluster","is_final"}, ...]}}
"""
import json, sys, collections

SEED = 0          # ask Aseel for the DMoN seed and put it here
PT   = 'leaf_assignment.pt (1).zip'
TREE = 'recursion_tree (1).jsonl'
REF  = 'flat_leiden_communities.json'   # only used to check node IDs match
OUT  = 'dmon_communities.json'

def load_leaf_ids(path):
    try:
        import torch
        return torch.load(path, weights_only=False).tolist()
    except ImportError:                 # fallback: read the raw int64 storage from the zip
        import zipfile, numpy as np
        z = zipfile.ZipFile(path)
        name = [n for n in z.namelist() if n.endswith('/data/0')][0]
        return np.frombuffer(z.read(name), dtype='<i8').tolist()

leaf_of_node = load_leaf_ids(PT)

paths = {}
for line in open(TREE):
    r = json.loads(line)
    if r['type'] == 'leaf':
        paths[r['leaf_id']] = tuple(r['path'])

# each unique path prefix at a given level gets its own dense cluster id
cluster_id = collections.defaultdict(dict)   # level -> {prefix: id}
def cid(level, prefix):
    d = cluster_id[level]
    if prefix not in d:
        d[prefix] = len(d)
    return d[prefix]

assignments = {}
for node, leaf in enumerate(leaf_of_node):
    p = paths[leaf]
    entries = [{"level": k, "cluster": cid(k, p[:k + 1]), "is_final": (k == len(p) - 1)}
               for k in range(len(p))]
    assignments[str(node)] = entries

# ---- checks ----
n_leaves = len({paths[l] for l in leaf_of_node})
print("nodes:", len(assignments), "| leaves:", n_leaves,
      "| levels:", len(cluster_id), "| clusters per level:", {k: len(v) for k, v in sorted(cluster_id.items())})
try:
    ref = json.load(open(REF))['assignments']
    assert set(assignments) == set(ref), "node IDs do not match the Leiden files!"
    print("node IDs match", REF)
except FileNotFoundError:
    print("(skipped node ID check:", REF, "not found)")

with open(OUT, 'w') as f:
    json.dump({"seed": SEED, "assignments": assignments}, f)
print("wrote", OUT)
