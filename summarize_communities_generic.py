"""
Generalized community-summarization step. The chassis design means
summarization should be *identical* across Leiden (step 7), k-core / flat
Leiden (step 10), and DMoN (week 4) — only the input/output file names
change. This replaces having a separate hardcoded copy per method.

Usage:
    python summarize_communities_generic.py --method leiden
    python summarize_communities_generic.py --method kcore
    python summarize_communities_generic.py --method flatleiden
    python summarize_communities_generic.py --method dmon
"""
import json, os, csv, argparse
from collections import defaultdict
import torch
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True, help='e.g. leiden, kcore, flatleiden, dmon')
parser.add_argument('--communities-file', default=None, help='defaults to {method}_communities.json')
parser.add_argument('--summaries-file', default=None, help='defaults to {method}_summaries.json')
parser.add_argument('--ledger-file', default=None, help='defaults to {method}_ledger.csv')
parser.add_argument('--max-report-length', type=int, default=100)
args = parser.parse_args()

communities_file = args.communities_file or f"{args.method}_communities.json"
summaries_file = args.summaries_file or f"{args.method}_summaries.json"
ledger_file = args.ledger_file or f"{args.method}_ledger.csv"

emb = torch.load('entity_embeddings.pt', weights_only=False)
node_names = emb['nodes']

with open(communities_file) as f:
    comm_data = json.load(f)
assignments = comm_data['assignments']

PROMPT_TEMPLATE = """You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community. IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 2-3 key insights about the community. Each insight should have a short summary followed by one paragraph of explanatory text grounded according to the grounding rules below.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }}
        ]
    }}

# Grounding Rules
Points supported by data should list their data references as follows:
"This is an example sentence supported by data references [Data: <dataset name> (record ids)]."
Do not include information where the supporting evidence for it is not provided.
Limit the total report length to {max_report_length} words.

# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}

Output:"""

groups = defaultdict(list)
for node_idx_str, levels in assignments.items():
    idx = int(node_idx_str)
    name = node_names[idx] if idx < len(node_names) else node_idx_str
    for lv in levels:
        key = (lv['level'], lv['cluster'])
        groups[key].append(name)

try:
    with open(summaries_file) as f:
        summaries = json.load(f)
except FileNotFoundError:
    summaries = {}

summaries = {
    k: v for k, v in summaries.items()
    if not v['summary'].startswith('ERROR') and v['summary'].strip() != ''
}
print(f"[{args.method}] Already done (kept): {len(summaries)}")

remaining = {k: v for k, v in groups.items() if f"L{k[0]}_C{k[1]}" not in summaries}
print(f"[{args.method}] Remaining to process: {len(remaining)}")

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

MAX_TOKENS = 500
MODEL = "openai/gpt-oss-20b"

ledger_rows = []

for i, ((level, cluster), members) in enumerate(remaining.items()):
    key = f"L{level}_C{cluster}"
    input_text = "Entities: " + ", ".join(members[:30])
    prompt = PROMPT_TEMPLATE.format(input_text=input_text, max_report_length=args.max_report_length)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            reasoning_effort="low",
        )
        summary_text = resp.choices[0].message.content
        usage = resp.usage
        tokens_in = usage.prompt_tokens
        tokens_out = usage.completion_tokens
        if not summary_text or summary_text.strip() == '':
            summary_text = f"ERROR: empty response (finish_reason={resp.choices[0].finish_reason})"
    except Exception as e:
        print("FULL ERROR:", repr(e))
        summary_text = f"ERROR: {e}"
        tokens_in = tokens_out = 0
        if "limit" in str(e).lower():
            print("Limit hit — stopping early, re-run later.")
            break

    summaries[key] = {"level": level, "cluster": cluster, "members": members, "summary": summary_text}
    ledger_rows.append([key, level, cluster, tokens_in, tokens_out])

    if i % 20 == 0:
        print(f"[{args.method}] Done {i}/{len(remaining)}")
        with open(summaries_file, 'w') as f:
            json.dump(summaries, f, indent=2)

with open(summaries_file, 'w') as f:
    json.dump(summaries, f, indent=2)

with open(ledger_file, 'a', newline='') as f:
    w = csv.writer(f)
    w.writerows(ledger_rows)

print(f"[{args.method}] Session done. Total summaries now: {len(summaries)} / {len(groups)}")
