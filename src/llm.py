"""Any OpenAI-compatible free provider (Groq, OpenRouter, Gemini-compat...). No extra packages needed."""
import json, os, urllib.request

def chat_json(messages, temperature=0.0):
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set")
    base = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    body = json.dumps({
        "model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "policypilot/1.0"})  # Groq can block the default urllib agent
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.load(r)
    return json.loads(data["choices"][0]["message"]["content"])
