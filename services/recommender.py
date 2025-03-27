import json
from sklearn.metrics.pairwise import cosine_similarity
from utils.text import build_user_text
from utils.vector_cache import MODEL, VECTORS, TEXTS, IDS, EQUIPMENT_DATA
from models.schema import RecommendRequest

def has(text, tag):
    return tag.lower() in text

def rule_based_score(option, req: RecommendRequest, text: str):
    score = 0
    tags = set(option.get("tags", []))
    attrs = set(option.get("attribute_values", []))

    def has(tag): return tag.lower() in text

    # 🔹 Base Tag Weight Boosting by Preference
    for pref in req.preferences or []:
        if pref.tag:
            if has(pref.tag):
                score += 5
            if pref.group == "muscle" and has(pref.tag):
                score += 3
            if pref.group == "goal" and has(pref.tag):
                score += 3
        if pref.max_price and option["price"] <= pref.max_price:
            score += 5
        if pref.min_weight and option["weight"] >= pref.min_weight:
            score += 5

    # 🔹 Gender-aware boost
    if req.gender == "female":
        if any(t in text for t in ["glutes", "core", "abs"]):
            score += 6
        if "compact" in text or "adjustable" in text:
            score += 3
    elif req.gender == "male":
        if any(t in text for t in ["arms", "chest", "pull-up"]):
            score += 6
        if "heavy" in text or option["weight"] >= 60:
            score += 4

    # 🔹 Age-aware logic
    if req.age:
        if req.age >= 50:
            if any(t in text for t in ["low-impact", "joint-friendly", "post-injury"]):
                score += 10
            if option["weight"] < 40:
                score += 4
        elif req.age < 18:
            score += 3  # youth-friendly content

    # 🔹 Goal → Tag Mapping
    goal_tags = {
        "tone": ["bodyweight", "multi-function", "compact"],
        "build-muscle": ["resistance", "weighted", "barbell-compatible"],
        "weight-loss": ["cardio", "endurance", "bodyweight"],
        "rehab": ["low-impact", "joint-friendly", "stretching"],
        "mobility": ["stretching", "flexibility", "balance"],
        "strength": ["weighted", "barbell-compatible", "resistance"],
        "endurance": ["cardio", "row", "treadmill"],
        "flexibility": ["stretching", "mobility"],
        "posture-correction": ["core", "back", "adjustable"],
        "pre/post-natal": ["low-impact", "core", "mobility"],
        "athletic-training": ["cable", "multi-function", "tower"],
        "injury-prevention": ["joint-friendly", "adjustable"],
        "functionality": ["full-body", "multi-function"]
    }

    if req.goal and req.goal in goal_tags:
        for tag in goal_tags[req.goal]:
            if has(tag):
                score += 4

    # 🔹 Experience-based bias
    if req.experience:
        exp = req.experience.lower()
        if exp == "beginner" and has("beginner-friendly"):
            score += 6
        if exp == "intermediate" and has("intermediate"):
            score += 4
        if exp == "advanced" and has("advanced"):
            score += 4
        if exp == "athlete":
            if has("athlete") or option["weight"] > 80:
                score += 6
        if exp == "elderly":
            if any(t in text for t in ["low-impact", "joint-friendly", "elderly"]):
                score += 8

    return score

def get_recommendations(req: RecommendRequest):
    user_text = build_user_text(req)
    user_vector = MODEL.encode(user_text, convert_to_tensor=False).reshape(1, -1)
    similarities = cosine_similarity(user_vector, VECTORS)[0]

    results = []
    for i, sim in enumerate(similarities):
        match = next((o for o in EQUIPMENT_DATA if str(o.get("id")) == IDS[i]), None)
        if match:
            text = json.dumps(match).lower()
            rules = rule_based_score(match, req, text)
            match["score"] = float(sim * 10 + rules)
            match["__debug"] = {
                "embedding_similarity": round(sim * 10, 2),
                "rule_score": rules,
                "user_text": user_text,
                "equipment_text": TEXTS[i]
            }
            results.append(match)

    return sorted(results, key=lambda x: x["score"], reverse=True)[:10]
