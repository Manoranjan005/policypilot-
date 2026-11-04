import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .ingest import load_policies
from .rag import answer
from .retriever import Retriever

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.environ.setdefault("POLICY_DIR", os.path.join(ROOT, "data", "policies"))

app = FastAPI(title="PolicyPilot")
retriever = Retriever()

try:
    from .intent import IntentModel
    intent = IntentModel(os.path.join(ROOT, "models", "intent.joblib"))
except Exception:
    intent = None  # app still works, just without intent labels

class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)

@app.post("/api/ask")
def ask(body: Ask):
    try:
        return answer(body.question, retriever, intent)
    except Exception as e:
        return dict(question=body.question, answerable=False,
                    answer="The service could not produce a grounded answer. Please try again or contact IT.",
                    citations=[], evidence=[], intent=None, reason=f"error:{type(e).__name__}")

@app.get("/api/policies")
def policies():
    return [{k: d[k] for k in ("policy_id", "title", "file")} for d in load_policies()]

@app.get("/")
def home():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))
