"""Any OpenAI-compatible provider. Uses only the standard library."""
import json, os, urllib.request, urllib.error

def _endpoint():
    base = (os.getenv("LLM_BASE_URL") or "https://api.groq.com/openai/v1").strip().strip('"').strip("'")
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):          # user pasted the full endpoint
        base = base[: -len("/chat/completions")]
    if base == "https://api.groq.com":              # missing /openai/v1
        base += "/openai/v1"
    return base + "/chat/completions"

def chat_json(messages, temperature=0.0):
    key = (os.getenv("LLM_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set")
    model = (os.getenv("LLM_MODEL") or "llama-3.1-8b-instant").strip()
    url = _endpoint()
    body = json.dumps({"model": model, "messages": messages, "temperature": temperature,
                       "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "policypilot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:200]
        raise RuntimeError(f"HTTP {e.code} url={url} model={model} reply={detail}")
    return json.loads(data["choices"][0]["message"]["content"])
