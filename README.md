# PolicyPilot — grounded HR/IT policy assistant
Free stack: local `all-MiniLM-L6-v2` embeddings, ChromaDB, scikit-learn MLP, and any OpenAI-compatible free LLM API (default **Groq**, key at console.groq.com; OpenRouter / Gemini-compat also work via `.env`).

## Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your free API key, then: export $(grep -v '^#' .env | xargs)
python -m src.intent            # trains baseline vs MLP, writes reports/intent_report.json
uvicorn src.api:app --reload    # open http://127.0.0.1:8000
pytest                          # tests
python -m eval.evaluate --e2e   # Recall@k, MRR, citation correctness, abstention
```
## Architecture
policy .txt -> ingest (metadata, normalize, sentence-window chunks) -> embeddings -> Chroma (access filter) -> top-k -> score gate -> LLM (JSON, evidence-only) -> citation validation -> answer or abstain + human routing. Intent MLP runs in parallel for routing/analytics.

## Using your other ~30 datasets
Point `POLICY_DIR` to another folder of `.txt` files with `Policy ID:` / `Title:` headers, grow `data/intent_examples.csv` and `data/evaluation_questions.csv`, rerun train + eval. Add an `access` field per doc to use the metadata filter.

## To finish for the deliverables
Expand intent examples (200+ with paraphrases/ambiguous), expand eval set to 50+ questions, add an LLM-judge for groundedness/answer correctness, record confusion matrix, add the demo recording. Tune `ABSTAIN_THRESHOLD` using eval.
