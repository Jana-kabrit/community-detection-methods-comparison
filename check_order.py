import json, random, collections

d = json.load(open('dmon_communities.json'))['assignments']
f = json.load(open('flat_leiden_communities.json'))['assignments']

def final(v):
    e = [x for x in v if x['is_final']][0]
    return (e['level'], e['cluster'])

nodes = list(d)
dm = {n: final(d[n]) for n in nodes}
fl = {n: f[n][0]['cluster'] for n in nodes}

def purity(flat_labels):
    g = collections.defaultdict(collections.Counter)
    for n in nodes:
        g[dm[n]][flat_labels[n]] += 1
    big = [c for c in g.values() if sum(c.values()) >= 10]
    return sum(max(c.values()) for c in big) / sum(sum(c.values()) for c in big), len(big)

real, nbig = purity(fl)
vals = [fl[n] for n in nodes]
random.seed(0); random.shuffle(vals)
shuf, _ = purity(dict(zip(nodes, vals)))
print(f"DMoN communities of size>=10: {nbig}")
print(f"agreement with flat Leiden: {real:.3f}   | shuffled baseline: {shuf:.3f}")
print("ORDER LOOKS RIGHT" if real > 3 * shuf and real > 0.2 else "ORDER LOOKS WRONG - tell me")
