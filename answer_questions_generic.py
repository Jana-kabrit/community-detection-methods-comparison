"""
Generalized answer-generation step. Every experiment (H1 Leiden, H2 k-core,
H2b flat Leiden, H3 DMoN) runs through this identical chassis — same
retrieval, same prompt, same model — so the only thing that differs between
runs is which retrieval index gets passed in. That's what makes the config
diff in diff_configs.py meaningful.

Usage:
    python answer_questions_generic.py --index retrieval_index_leiden.pt --output h1_answers.json
    python answer_questions_generic.py --index retrieval_index_kcore.pt --output h2_answers.json
    python answer_questions_generic.py --index retrieval_index_flatleiden.pt --output h2b_answers.json
"""
import json, torch, os, argparse
from sentence_transformers import SentenceTransformer
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument('--index', default='retrieval_index.pt')
parser.add_argument('--questions', default='reproduce/dataset/musique.json')
parser.add_argument('--output', required=True, help='e.g. h2_answers.json')
parser.add_argument('--top-k', type=int, default=5)
args = parser.parse_args()

print(f"Loading retrieval index from {args.index}...")
idx = torch.load(args.index, weights_only=False)
doc_ids, doc_types, docs, doc_embeddings = idx['doc_ids'], idx['doc_types'], idx['docs'], idx['embeddings']

print(f"Loading questions from {args.questions}...")
with open(args.questions) as f:
    questions_data = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "openai/gpt-oss-20b"

try:
    with open(args.output) as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}

print(f"Already answered: {len(results)}")

remaining_indices = [
    i for i in range(len(questions_data))
    if str(i) not in results or not results[str(i)]['generated_answer'].strip() or results[str(i)]['generated_answer'].startswith('ERROR')]
print(f"Remaining to answer: {len(remaining_indices)}")


def cosine_sim(a, b):
    a_norm = a / a.norm(dim=-1, keepdim=True)
    b_norm = b / b.norm(dim=-1, keepdim=True)
    return a_norm @ b_norm.T


for count, i in enumerate(remaining_indices):
    q = questions_data[i]
    qid = str(i)
    question_text = q.get('question', '')
    gold_answer = q.get('answer', [''])
    if isinstance(gold_answer, list):
        gold_answer = gold_answer[0] if gold_answer else ''

    q_emb = torch.tensor(model.encode([question_text]))
    sims = cosine_sim(q_emb, doc_embeddings)[0]
    top_idx = torch.topk(sims, args.top_k).indices.tolist()
    retrieved_docs = [docs[j] for j in top_idx]
    retrieved_ids = [doc_ids[j] for j in top_idx]
    retrieved_types = [doc_types[j] for j in top_idx]

    context = "\n\n".join(retrieved_docs)
    prompt = f"""Answer the question using ONLY the context below. Be concise — give just the answer, no explanation.

Context:
{context}

Question: {question_text}
Answer:"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            reasoning_effort="low",
        )
        generated_answer = resp.choices[0].message.content
        if not generated_answer or not generated_answer.strip():
            generated_answer = f"ERROR: empty response (finish_reason={resp.choices[0].finish_reason})"
    except Exception as e:
        generated_answer = f"ERROR: {e}"
        if "limit" in str(e).lower():
            print("Limit hit — stopping early, re-run later.")
            break

    results[qid] = {
        "question": question_text,
        "gold_answer": gold_answer,
        "generated_answer": generated_answer,
        "retrieved_ids": retrieved_ids,
        "retrieved_types": retrieved_types,
    }

    if count % 20 == 0:
        print(f"Done {count}/{len(remaining_indices)}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Session done. Total answered: {len(results)} / {len(questions_data)}")
