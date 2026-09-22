r"""Is DMoN (H3) really different from each baseline? Paired bootstrap on the same 1,000 questions.

Usage (one line):
  python paired_test.py H1=C:\path\h1_answers.json H2=C:\path\h2_answers.json H2b=C:\path\h2b_answers.json H3=h3_answers.json

For each baseline it prints the mean difference (DMoN minus baseline) for F1 and EM,
with a 95% interval. If the interval contains 0, the difference could be noise.
"""
import sys, json, re, string, random, collections

def norm(s):
    s = ''.join(ch for ch in str(s).lower() if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)

def per_question(path):
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    for k, r in d.items():
        pred = r.get('generated_answer', '') or ''
        gold = r.get('gold_answer', '')
        if isinstance(gold, list):
            gold = gold[0] if gold else ''
        if not pred.strip() or pred.startswith('ERROR'):
            out[k] = (0.0, 0.0)
        else:
            out[k] = (float(norm(pred) == norm(gold)), f1(pred, gold))
    return out

runs = dict(a.split('=', 1) for a in sys.argv[1:])
if 'H3' not in runs or len(runs) < 2:
    sys.exit(__doc__)
scores = {n: per_question(p) for n, p in runs.items()}
keys = sorted(scores['H3'])
random.seed(0)
B = 5000
idx = [[random.randrange(len(keys)) for _ in keys] for _ in range(B)]

print(f"{'DMoN minus':<12}{'metric':<8}{'mean diff':>10}{'95% interval':>22}   verdict")
for name in runs:
    if name == 'H3':
        continue
    for m, label in ((1, 'F1'), (0, 'EM')):
        diff = [scores['H3'][k][m] - scores[name][k][m] for k in keys]
        obs = sum(diff) / len(diff)
        boots = sorted(sum(diff[i] for i in ix) / len(ix) for ix in idx)
        lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B)]
        verdict = 'DMoN better' if lo > 0 else ('DMoN worse' if hi < 0 else 'tie (could be noise)')
        print(f"{name:<12}{label:<8}{obs:>+10.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}   {verdict}")
