import json
from sklearn.metrics.pairwise import cosine_similarity
from utils.text import build_user_text, clean_text
from models.schema import RecommendRequest


with open("data/vector_cache.json", "r") as f:
    vector_data = json.load(f)

VECTORS = [v for v in vector_data.values()]
IDS = [k for k in vector_data.keys()]
TEXTS = [clean_text(" ".join(map(str, v))) if isinstance(v, list) else "" for v in vector_data.values()]

with open("data/equipment_options_with_tags.json", "r") as f:
    EQUIPMENT_DATA = json.load(f)

def has(text, tag):
    return tag.lower() in text

def rule_based_score(option, req: RecommendRequest, text: str):
    score = 0
    debug = {}

    tags = set([t.lower() if isinstance(t, str) else t.get("name", "").lower() for t in option.get("tags", [])])
    attrs = set([a.lower() for a in option.get("attribute_values", [])])

    def has_tag(name): return name.lower() in tags
    def has_attr(name): return name.lower() in attrs
    def has_text(name): return name.lower() in text

    # 🔸 Preference Matching
    for pref in req.preferences or []:
        if pref.tag:
            if has_tag(pref.tag): score += 6; debug[f"pref:tag:{pref.tag}"] = 6
            if has_text(pref.tag): score += 3; debug[f"pref:text:{pref.tag}"] = 3
            if pref.group == "muscle" and has_tag(pref.tag): score += 4; debug[f"pref:muscle:{pref.tag}"] = 4
            if pref.group == "goal" and has_tag(pref.tag): score += 4; debug[f"pref:goal:{pref.tag}"] = 4
        if pref.max_price and option["price"] <= pref.max_price:
            score += 5; debug[f"pref:max_price:{pref.max_price}"] = 5
        if pref.min_weight and option["weight"] >= pref.min_weight:
            score += 5; debug[f"pref:min_weight:{pref.min_weight}"] = 5

    # 🔸 Attribute Bonus
    for attr, val in {
        "adjustable": 2, "compact": 2, "portable": 1,
        "foldable": 1, "budget": 1, "multi-function": 2
    }.items():
        if has_attr(attr): score += val; debug[f"attr:{attr}"] = val

    # 🔸 Gender-based muscle focus
    if req.gender == "female":
        if any(has_tag(tag) for tag in ["glutes", "core", "abs"]): score += 6; debug["gender:female:glutes/core/abs"] = 6
        if any(has_attr(attr) for attr in ["compact", "adjustable"]): score += 3; debug["gender:female:compact/adjustable"] = 3
    elif req.gender == "male":
        if any(has_tag(tag) for tag in ["arms", "chest", "pull-up"]): score += 6; debug["gender:male:arms/chest/pull-up"] = 6
        if "heavy" in text or option.get("weight", 0) >= 60: score += 4; debug["gender:male:heavy_or_weight>=60"] = 4

    # 🔸 Age-based logic
    if req.age:
        if req.age >= 50:
            if any(has_tag(tag) for tag in ["low-impact", "joint-friendly", "post-injury"]):
                score += 10; debug["age:50+:safety_tags"] = 10
            if option.get("weight", 0) < 40: score += 4; debug["age:50+:weight<40"] = 4
        elif req.age < 18:
            score += 3; debug["age<18"] = 3

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
            if has_tag(tag): score += 4; debug[f"goal:tag:{tag}"] = 4
            if has_attr(tag): score += 2; debug[f"goal:attr:{tag}"] = 2

    # 🔸 Experience-aware tag matching
    exp = (req.experience or "").lower()
    if exp == "beginner" and has_tag("beginner-friendly"): score += 6; debug["exp:beginner"] = 6
    if exp == "intermediate" and has_tag("intermediate"): score += 4; debug["exp:intermediate"] = 4
    if exp == "advanced" and has_tag("advanced"): score += 4; debug["exp:advanced"] = 4
    if exp == "athlete":
        if has_tag("athlete") or option.get("weight", 0) > 80: score += 6; debug["exp:athlete"] = 6
    if exp == "elderly":
        if has_tag("elderly") or has_tag("joint-friendly") or has_attr("low-impact"):
            score += 8; debug["exp:elderly"] = 8

    # 🔸 New: weight and height consideration
    if req.weight and req.weight >= 90:
        score += 3; debug["user:weight>=90"] = 3
    if req.height and req.height >= 190:
        score += 2; debug["user:height>=190"] = 2

    # 🔸 New: user_type
    if req.user_type:
        if req.user_type.lower() == "athlete": score += 3; debug["user_type:athlete"] = 3
        elif req.user_type.lower() == "elderly": score += 5; debug["user_type:elderly"] = 5

    return score, debug

def get_recommendations(req: RecommendRequest, app):
    user_text = build_user_text(req)
    user_vector = app.state.model.encode(user_text, convert_to_tensor=False).reshape(1, -1)
    similarities = cosine_similarity(user_vector, VECTORS)[0]

    results = []
    for i, sim in enumerate(similarities):
        match = next((o for o in EQUIPMENT_DATA if str(o.get("option_id")) == IDS[i]), None)
        if match:
            text = json.dumps(match).lower()
            rules, rule_debug = rule_based_score(match, req, text)
            match["score"] = float(sim * 10 + rules)
            match["__debug"] = {
                "embedding_similarity": round(sim * 10, 2),
                "rule_score": rules,
                "user_text": user_text,
                "equipment_text": TEXTS[i],
                "rule_breakdown": rule_debug
            }
            results.append(match)

    seen = {}
    for option in sorted(results, key=lambda x: x["score"], reverse=True):
        eq_id = option.get("equipment_id")
        if eq_id not in seen:
            seen[eq_id] = option

    sorted_results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:100]
