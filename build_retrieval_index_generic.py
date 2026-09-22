"""
Generalized retrieval-index build step. Same chassis for every method's
community summaries (Leiden, k-core, flat Leiden, DMoN) — only the file
names change between runs.

Usage:
    python build_retrieval_index_generic.py --summaries leiden_summaries.json --output retrieval_index_leiden.pt
    python build_retrieval_index_generic.py --summaries kcore_summaries.json --output retrieval_index_kcore.pt
"""
import json, torch, argparse
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--summaries', default='community_summaries.json')
parser.add_argument('--corpus', default='reproduce/dataset/musique_corpus.json')
parser.add_argument('--output', default='retrieval_index.pt')
args = parser.parse_args()

print(f"Loading community summaries from {args.summaries}...")
with open(args.summaries) as f:
    summaries = json.load(f)

print(f"Loading MuSiQue passages from {args.corpus}...")
with open(args.corpus) as f:
    corpus = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

docs = []       # text content
doc_ids = []    # identifier
doc_types = []  # "summary" or "passage"

for key, v in summaries.items():
    docs.append(v['summary'])
    doc_ids.append(key)
    doc_types.append('summary')

for i, p in enumerate(corpus):
    text = p.get('title', '') + ": " + p.get('text', '')
    docs.append(text)
    doc_ids.append(f"passage_{i}")
    doc_types.append('passage')

print(f"Total documents to embed: {len(docs)}")
embeddings = model.encode(docs, show_progress_bar=True, batch_size=64)

torch.save({
    'doc_ids': doc_ids,
    'doc_types': doc_types,
    'docs': docs,
    'embeddings': torch.tensor(embeddings),
}, args.output)

print(f"Saved {args.output}")
