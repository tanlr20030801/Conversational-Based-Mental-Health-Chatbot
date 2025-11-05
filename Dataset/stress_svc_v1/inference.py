import os, joblib
def predict_stress(text, model_dir, threshold=None):
    clf = joblib.load(os.path.join(model_dir, "classifier.joblib"))
    vec = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    thr_path = os.path.join(model_dir, "inference_threshold.txt")
    if threshold is None and os.path.exists(thr_path):
        with open(thr_path, "r") as f:
            threshold = float(f.read().strip())
    if threshold is None:
        threshold = 0.5
    X = vec.transform([text])
    prob = clf.predict_proba(X)[0, 1]
    return {"stress_probability": round(float(prob), 4),
            "prediction": "STRESS" if prob >= threshold else "CALM",
            "threshold": threshold}
