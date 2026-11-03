"""TF-IDF cosine retriever (lightweight, Vercel-friendly)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .ingest import load_policies, chunk

class Retriever:
    def __init__(self, window=2, stride=1):
        self.chunks = [c for d in load_policies() for c in chunk(d, window, stride)]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.m = self.vec.fit_transform([f"{c['title']}. {c['text']}" for c in self.chunks])

    def search(self, q, k=4, allowed_access=("internal", "public")):
        sims = (self.m @ self.vec.transform([q]).T).toarray().ravel()
        order = [i for i in np.argsort(-sims) if self.chunks[i]["access"] in allowed_access][:k]
        return [dict(score=round(float(sims[i]), 4), **self.chunks[i]) for i in order]
