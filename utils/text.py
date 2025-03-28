def build_user_text(req):
    parts = [
        f"goal:{req.goal}" if req.goal else "",
        f"experience:{req.experience}" if req.experience else "",
        f"gender:{req.gender}" if req.gender else ""
    ]
    parts += [
        f"{p.group}:{p.tag}" if p.group else p.tag
        for p in req.preferences or []
        if p.tag
    ]
    return " ".join([p for p in parts if p])

import re
import numpy as np

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\\s]", "", text).lower().strip()

def embed_text(text):
    # Dummy vector for example; replace with model-based embedding
    np.random.seed(abs(hash(text)) % 1234567)
    return np.random.rand(128).tolist()

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
