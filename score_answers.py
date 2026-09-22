r"""Score answer files the same way, side by side.

Usage (PowerShell, one line):
  python score_answers.py H1=C:\path\h1_answers.json H2=C:\path\h2_answers.json H2b=C:\path\h2b_answers.json H3=h3_answers.json

Metrics (standard open-domain QA scoring against gold_answer):
  EM       exact match after normalising (lowercase, no punctuation, no articles)
  F1       token-overlap F1 after the same normalising
  Contains gold answer appears inside the prediction (or the reverse) after normalising
Blank / ERROR answers score 0 on everything and are counted in the 'blank' column.
Writes scores_h3.csv next to where you run it.
"""
import sys, json, re, string, math, csv, collections

def norm(s):
    s = str(s).lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)

def score(path):
    d = json.load(open(path))
    rows = list(d.values()) if isinstance(d, dict) else d
    em = []; ff = []; ct = []; blank = 0
    for r in rows:
        pred = r.get('generated_answer', '') or ''
        gold = r.get('gold_answer', '')
        if isinstance(gold, list):
            gold = gold[0] if gold else ''
        if not pred.strip() or pred.startswith('ERROR'):
            blank += 1
            em.append(0.0); ff.append(0.0); ct.append(0.0)
            continue
        np_, ng = norm(pred), norm(gold)
        em.append(float(np_ == ng))
        ff.append(f1(pred, gold))
        ct.append(float(bool(ng) and (ng in np_ or (np_ and np_ in ng))))
    return em, ff, ct, blank

def mean(x): return sum(x) / len(x)
def ci(x):
    m = mean(x); sd = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
    return 1.96 * sd / math.sqrt(len(x))

runs = []
for a in sys.argv[1:]:
    name, path = a.split('=', 1)
    runs.append((name, path))
if not runs:
    sys.exit(__doc__)

out = []
print(f"{'run':<6}{'n':>6}{'EM':>9}{'F1':>9}{'  +-95%':>9}{'Contains':>10}{'blank':>7}")
for name, path in runs:
    em, ff, ct, blank = score(path)
    print(f"{name:<6}{len(em):>6}{mean(em):>9.3f}{mean(ff):>9.3f}{ci(ff):>9.3f}{mean(ct):>10.3f}{blank:>7}")
    out.append([name, len(em), round(mean(em), 4), round(mean(ff), 4), round(ci(ff), 4), round(mean(ct), 4), blank])
with open('scores_h3.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['run', 'n', 'EM', 'F1', 'F1_ci95', 'Contains', 'blank']); w.writerows(out)
print("wrote scores_h3.csv")
