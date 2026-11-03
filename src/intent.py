"""Intent classifier: TF-IDF+LogReg baseline vs TF-IDF+MLP. Reports per-class F1."""
import json, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

PATH = "models/intent.joblib"
def _pipe(clf): return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), clf)

def train(csv="data/intent_examples.csv", seed=42):
    df = pd.read_csv(csv)
    Xtr, Xte, ytr, yte = train_test_split(df.text, df.label, test_size=0.3, stratify=df.label, random_state=seed)
    models = {"baseline_logreg": _pipe(LogisticRegression(max_iter=1000)),
              "mlp": _pipe(MLPClassifier((64,), max_iter=800, random_state=seed))}
    report = {}
    for n, m in models.items():
        m.fit(Xtr, ytr)
        report[n] = classification_report(yte, m.predict(Xte), output_dict=True, zero_division=0)
    final = _pipe(MLPClassifier((64,), max_iter=800, random_state=seed)).fit(df.text, df.label)
    joblib.dump(final, PATH)
    json.dump(report, open("reports/intent_report.json", "w"), indent=2)
    for n, r in report.items():
        print(n, "macro-F1 = %.3f" % r["macro avg"]["f1-score"],
              {k: round(v["f1-score"], 2) for k, v in r.items() if isinstance(v, dict) and k not in ("macro avg", "weighted avg")})

class IntentModel:
    def __init__(self): self.m = joblib.load(PATH)
    def predict(self, q):
        p = self.m.predict_proba([q])[0]; i = p.argmax()
        return {"label": str(self.m.classes_[i]), "confidence": round(float(p[i]), 3)}

if __name__ == "__main__": train()
