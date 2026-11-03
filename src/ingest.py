"""Ingestion: load policy files, keep metadata, normalize, chunk with source locations."""
import glob, os, re

def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def load_policies(d: str | None = None):
    d = d or os.getenv("POLICY_DIR", "data/policies")
    docs = []
    for p in sorted(glob.glob(os.path.join(d, "*.txt"))):
        raw = open(p, encoding="utf-8").read()
        pid = re.search(r"Policy ID:\s*(\S+)", raw)
        title = re.search(r"Title:\s*(.+)", raw)
        if not (pid and title):
            continue  # data-quality gate: skip files without required metadata
        body = raw.split("\n", 2)[2] if raw.count("\n") >= 2 else ""
        docs.append(dict(policy_id=pid.group(1), title=title.group(1).strip(),
                         file=os.path.basename(p), text=normalize(body), access="internal"))
    return docs

def chunk(doc, window=2, stride=1):
    """Sliding window over sentences; records sentence span as source location."""
    sents = re.split(r"(?<=[.!?])\s+", doc["text"])
    out = []
    for i in range(0, max(1, len(sents) - window + 1), stride):
        s = sents[i:i + window]
        out.append({**{k: doc[k] for k in ("policy_id", "title", "file", "access")},
                    "chunk_id": f"{doc['policy_id']}#{i}", "text": " ".join(s),
                    "sent_start": i, "sent_end": i + len(s) - 1})
    return out
