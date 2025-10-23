# ======================== model_utils.py ========================
# ======================== model_utils.py ========================
import subprocess, sys

# Auto-install required packages if missing
def ensure_package(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for package in ["joblib", "numpy", "sentence-transformers", "scikit-learn"]:
    ensure_package(package)

# ==== Normal imports ====
import os, json, joblib, numpy as np
from sentence_transformers import SentenceTransformer


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity using NumPy only (1xD or DxD)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float((a @ b.T).ravel()[0])

def load_model_by_id(model_id: str, base_dir: str):
    """Load a model (zero-shot or supervised) by ID."""
    path = os.path.join(base_dir, "models", model_id)
    with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    embed = SentenceTransformer(meta["embedding_model"])

    if meta["type"] == "zero_shot_similarity":
        # Load optional prototypes if available
        stress_proto = None
        nonstress_proto = None
        sp_path = os.path.join(path, "stress_proto.json")
        nsp_path = os.path.join(path, "nonstress_proto.json")
        if os.path.exists(sp_path) and os.path.exists(nsp_path):
            stress_proto = np.array(json.load(open(sp_path))["vector"], dtype=np.float32)
            nonstress_proto = np.array(json.load(open(nsp_path))["vector"], dtype=np.float32)
        return {
            "type": "zero_shot_similarity",
            "embed": embed,
            "meta": meta,
            "stress_proto": stress_proto,
            "nonstress_proto": nonstress_proto
        }

    elif meta["type"] == "supervised_lr":
        clf = joblib.load(os.path.join(path, "classifier.joblib"))
        return {"type": "supervised_lr", "clf": clf, "embed": embed, "meta": meta}

    else:
        raise ValueError("Unknown model type in meta.json")

def predict_with_model(model_id: str, text: str, base_dir: str):
    """Predict stress (1) or non-stress (0) for any text."""
    bundle = load_model_by_id(model_id, base_dir)
    v = bundle["embed"].encode([str(text)], convert_to_numpy=True)

    if bundle["type"] == "zero_shot_similarity":
        # Use prototypes if available; otherwise use quick anchors
        if bundle.get("stress_proto") is not None and bundle.get("nonstress_proto") is not None:
            s_stress = _cosine_np(v, bundle["stress_proto"])
            s_non    = _cosine_np(v, bundle["nonstress_proto"])
        else:
            proto = bundle["embed"].encode(["stress anxious panic died worried 索取 焦虑 紧张"], convert_to_numpy=True)
            non_proto = bundle["embed"].encode(["calm relaxed fine peaceful donation giving 给予 放松 平静"], convert_to_numpy=True)
            s_stress = _cosine_np(v, proto)
            s_non    = _cosine_np(v, non_proto)
        score = s_stress - s_non
        margin = float(bundle["meta"].get("threshold_margin", 0.0))
        pred = int(score > margin)
        return {
            "model_id": model_id,
            "prediction": pred,
            "score": float(score),
            "sim_stress": float(s_stress),
            "sim_nonstress": float(s_non),
            "margin": margin
        }

    else:  # supervised_lr
        p = float(bundle["clf"].predict_proba(v)[0, 1])
        thr = float(bundle["meta"].get("threshold", 0.5))
        return {
            "model_id": model_id,
            "prediction": int(p >= thr),
            "prob_stress": p,
            "threshold": thr
        }
