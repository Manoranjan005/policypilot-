"""Evaluate retrieval (Recall@k, MRR) per chunk setting, then end-to-end citation + abstention.
Usage: python -m eval.evaluate [--e2e]   (e2e needs LLM_API_KEY)"""
import json, sys, pandas as pd
from src.retriever import Retriever
from src.rag import answer

df = pd.read_csv("data/evaluation_questions.csv").fillna("")
ans = df[df.expected_type == "answerable"]
report = {"retrieval": {}}
for w in (1, 2, 3):
    r, rec, mrr = Retriever(window=w), {1: 0, 3: 0, 5: 0}, 0
    for q, pid in zip(ans.question, ans.expected_policy_id):
        ranked = []
        for h in r.search(q, 10):
            if h["policy_id"] not in ranked: ranked.append(h["policy_id"])
        for k in rec: rec[k] += pid in ranked[:k]
        mrr += 1 / (ranked.index(pid) + 1) if pid in ranked else 0
    n = len(ans)
    report["retrieval"][f"window={w}"] = {**{f"recall@{k}": v / n for k, v in rec.items()}, "mrr": mrr / n}
if "--e2e" in sys.argv:
    r, ok_cite, ok_abs, na = Retriever(2), 0, 0, 0
    for q, t, pid in zip(df.question, df.expected_type, df.expected_policy_id):
        out = answer(q, r)
        if t == "answerable": ok_cite += pid in out["citations"]
        else: na += 1; ok_abs += (not out["answerable"])
    report["e2e"] = {"citation_correctness": ok_cite / len(ans), "abstention_accuracy": ok_abs / max(na, 1)}
json.dump(report, open("reports/eval.json", "w"), indent=2); print(json.dumps(report, indent=2))
