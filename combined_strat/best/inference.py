# inference.py
import os, importlib.util, joblib

def _inject_mixed_tokenize(model_dir: str):
    import __main__ as _m
    if hasattr(_m, "mixed_tokenize"):
        return
    shim_path = os.path.join(model_dir, "tokenizer_shim.py")
    spec = importlib.util.spec_from_file_location("tokenizer_shim", shim_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    _m.mixed_tokenize = getattr(mod, "mixed_tokenize")

def predict_stress(text: str, model_dir: str, threshold: float | None = None):
    _inject_mixed_tokenize(model_dir)
    clf = joblib.load(os.path.join(model_dir, "classifier.joblib"))
    vec = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))

    thr_path = os.path.join(model_dir, "inference_threshold.txt")
    if threshold is None and os.path.exists(thr_path):
        try:
            with open(thr_path, "r") as f:
                threshold = float(f.read().strip())
        except Exception:
            threshold = 0.5
    if threshold is None:
        threshold = 0.5

    X = vec.transform([text])
    if hasattr(clf, "predict_proba"):
        prob = float(clf.predict_proba(X)[0, 1])
    else:
        import numpy as np
        score = float(clf.decision_function(X)[0])
        prob = 1.0 / (1.0 + np.exp(-score))

    return {
        "stress_probability": round(prob, 4),
        "prediction": "STRESS" if prob >= threshold else "CALM",
        "threshold": float(threshold),
    }
