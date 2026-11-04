"""Grounded answer generation with two-layer abstention (score gate + LLM verdict)."""
import os, time, logging
from . import llm
log = logging.getLogger("policypilot")  # logs metadata only, never prompt text

SYSTEM = """You are PolicyPilot, an enterprise HR/IT policy assistant.
Answer ONLY from the EVIDENCE. Never use outside knowledge. Cite policy IDs in square brackets, e.g. [IT-001].
If the evidence does not directly answer the question, set answerable=false.
Return JSON: {"answerable": bool, "answer": "short answer with [ID] citations", "citations": ["ID", ...]}"""

ABSTAIN = ("The policy corpus does not answer this question, so I can't determine the answer. "
           "I'm routing it to a human (HR/IT) for review.")

def answer(question, retriever, intent_model=None, k=4):
    t0 = time.time()
    thr = float(os.getenv("ABSTAIN_THRESHOLD", "0.30"))
    hits = retriever.search(question, k)
    try:
    intent = intent_model.predict(question) if intent_model else None
except Exception:
    intent = None  # e.g. scikit-learn version mismatch; answers still work
    res = dict(question=question, intent=intent, evidence=hits, answerable=False,
               answer=ABSTAIN, citations=[], reason="low_retrieval_score")
    if hits and hits[0]["score"] >= thr:
        ev = "\n".join(f"[{h['policy_id']}] {h['text']}" for h in hits)
        out = llm.chat_json([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": f"EVIDENCE:\n{ev}\n\nQUESTION: {question}"}])
        valid = {h["policy_id"] for h in hits}
        cites = [c for c in out.get("citations", []) if c in valid]
        if out.get("answerable") and cites:
            res.update(answerable=True, answer=out["answer"], citations=cites, reason="grounded")
        else:
            res["reason"] = "llm_abstained"
    log.info("intent=%s answerable=%s reason=%s ms=%d", intent, res["answerable"], res["reason"],
             (time.time() - t0) * 1000)
    return res
