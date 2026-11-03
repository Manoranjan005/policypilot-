from src.ingest import load_policies, chunk
from src.retriever import Retriever
from src import rag

def test_ingest_metadata():
    docs = load_policies("data/policies")
    assert {d["policy_id"] for d in docs} >= {"HR-001", "IT-001", "IT-003"}
    assert all(c["chunk_id"].startswith(d["policy_id"]) for d in docs for c in chunk(d))

def test_retrieval_finds_policy():
    top = [h["policy_id"] for h in Retriever().search("API key exposed what do I do", 3)]
    assert "IT-001" in top

def test_abstains_when_llm_says_no(monkeypatch):
    monkeypatch.setattr(rag.llm, "chat_json", lambda *a, **k: {"answerable": False, "answer": "", "citations": []})
    out = rag.answer("What is the maternity leave entitlement?", Retriever())
    assert not out["answerable"] and out["citations"] == []

def test_rejects_uncited_answer(monkeypatch):
    monkeypatch.setattr(rag.llm, "chat_json", lambda *a, **k: {"answerable": True, "answer": "x", "citations": ["FAKE-9"]})
    assert not rag.answer("How early should planned leave be submitted?", Retriever())["answerable"]
