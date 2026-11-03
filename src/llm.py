"""Any OpenAI-compatible free provider (Groq, OpenRouter, Gemini-compat, Together...)."""
import json, os, requests

def chat_json(messages, temperature=0.0):
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
    base = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    r = requests.post(f"{base}/chat/completions", timeout=40,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"), "messages": messages,
              "temperature": temperature, "response_format": {"type": "json_object"}})
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])
