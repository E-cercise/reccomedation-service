import json
from sklearn.metrics.pairwise import cosine_similarity
from utils.text import build_user_text
from utils.vector_cache import MODEL, VECTORS, TEXTS, IDS, EQUIPMENT_DATA
from models.schema import RecommendRequest

def has(text, tag):
    return tag.lower() in text

def rule_based_score(option, req: RecommendRequest, text: str):
    score = 0
    tags = set([t.lower() if isinstance(t, str) else t.get("name", "").lower() for t in option.get("tags", [])])
    attrs = set([a.lower() for a in option.get("attribute_values", [])])

    def has_tag(name): return name.lower() in tags
    def has_attr(name): return name.lower() in attrs
    def has_text(name): return name.lower() in text

    # 🔸 Preference Matching
    for pref in req.preferences or []:
        if pref.tag:
            if has_tag(pref.tag): score += 6
            if has_text(pref.tag): score += 3
            if pref.group == "muscle" and has_tag(pref.tag): score += 4
            if pref.group == "goal" and has_tag(pref.tag): score += 4
        if pref.max_price and option["price"] <= pref.max_price:
            score += 5
        if pref.min_weight and option["weight"] >= pref.min_weight:
            score += 5

    # 🔸 Attribute Bonus
    if has_attr("adjustable"): score += 2
    if has_attr("compact"): score += 2
    if has_attr("portable"): score += 1
    if has_attr("foldable"): score += 1
    if has_attr("budget"): score += 1
    if has_attr("multi-function"): score += 2

    # 🔸 Gender-based muscle focus
    if req.gender == "female":
        if has_tag("glutes") or has_tag("core") or has_tag("abs"): score += 6
        if has_attr("compact") or has_attr("adjustable"): score += 3
    elif req.gender == "male":
        if has_tag("arms") or has_tag("chest") or has_tag("pull-up"): score += 6
        if "heavy" in text or option.get("weight", 0) >= 60: score += 4

    # 🔸 Age-based logic
    if req.age:
        if req.age >= 50:
            if has_tag("low-impact") or has_tag("joint-friendly") or has_tag("post-injury"):
                score += 10
            if option.get("weight", 0) < 40: score += 4
        elif req.age < 18:
            score += 3  # general bonus for youth-safe

    # 🔸 Goal → Tag mapping
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

    if req.goal in goal_tags:
        for tag in goal_tags[req.goal]:
            if has_tag(tag): score += 4
            if has_attr(tag): score += 2

    # 🔸 Experience-aware tag matching
    if req.experience:
        exp = req.experience.lower()
        if exp == "beginner" and has_tag("beginner-friendly"):
            score += 6
        if exp == "intermediate" and has_tag("intermediate"):
            score += 4
        if exp == "advanced" and has_tag("advanced"):
            score += 4
        if exp == "athlete":
            if has_tag("athlete") or option.get("weight", 0) > 80:
                score += 6
        if exp == "elderly":
            if has_tag("elderly") or has_tag("joint-friendly") or has_attr("low-impact"):
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
