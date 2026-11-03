import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .ingest import load_policies
from .intent import IntentModel, train, PATH
from .rag import answer
from .retriever import Retriever

app = FastAPI(title="PolicyPilot")
if not os.path.exists(PATH): train()
retriever, intent = Retriever(), IntentModel()

class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)

@app.post("/api/ask")
def ask(body: Ask):
    try:
        return answer(body.question, retriever, intent)
    except Exception as e:  # safe default: never fabricate an answer on failure
        return dict(question=body.question, answerable=False, answer="The service could not produce a grounded answer. Please try again or contact IT.",
                    citations=[], evidence=[], intent=None, reason=f"error:{type(e).__name__}")

@app.get("/api/policies")
def policies():
    return [{k: d[k] for k in ("policy_id", "title", "file")} for d in load_policies()]

@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "static", "index.html"))
