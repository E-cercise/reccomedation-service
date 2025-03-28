# Re-run tag enrichment script after kernel reset
import json
import re

input_path = "data/equipment_options.json"
output_path = "data/equipment_options_with_tags.json"

TAG_KEYWORDS = {
    "adjustable": ["adjustable", "customizable"],
    "compact": ["compact", "space-saving", "small"],
    "portable": ["portable", "lightweight", "foldable"],
    "budget": ["cheap", "budget", "affordable"],
    "glutes": ["glutes", "butt", "hip thrust"],
    "core": ["core", "abs", "abdominal"],
    "abs": ["abs", "abdominal"],
    "arms": ["arms", "biceps", "triceps"],
    "shoulders": ["shoulder", "delts"],
    "chest": ["chest", "pecs"],
    "back": ["back", "lats"],
    "legs": ["legs", "quads", "hamstrings", "calves"],
    "full-body": ["full-body", "total-body"],
    "bodyweight": ["bodyweight", "calisthenics"],
    "resistance": ["resistance", "band", "tension"],
    "weighted": ["weighted", "plate", "dumbbell", "barbell"],
    "cardio": ["cardio", "aerobic"],
    "stretching": ["stretch", "mobility"],
    "beginner-friendly": ["beginner", "easy", "entry"],
    "intermediate": ["intermediate", "moderate"],
    "advanced": ["advanced", "intense"],
    "athlete": ["athlete", "pro", "sport"],
    "low-impact": ["low-impact", "joint-friendly", "gentle"],
    "joint-friendly": ["joint", "rehab", "recovery"],
    "post-injury": ["post-injury", "rehab"],
    "elderly": ["elderly", "senior"],
    "tone": ["tone", "shaping"],
    "build-muscle": ["build-muscle", "mass", "gain"],
    "weight-loss": ["weight-loss", "fat burn", "slim"],
    "endurance": ["endurance", "stamina"],
    "rehab": ["rehab", "recovery", "therapy"],
    "mobility": ["mobility", "flexibility", "range of motion"],
    "flexibility": ["flexibility", "stretch"],
    "posture-correction": ["posture", "alignment"],
    "pre/post-natal": ["prenatal", "postnatal", "pregnancy"],
    "athletic-training": ["athletic-training", "performance"],
    "injury-prevention": ["injury-prevention", "protective"],
    "functionality": ["functional", "daily use"],
    "pull-up": ["pull-up", "pullup"],
    "dip": ["dip"],
    "ab-machine": ["ab machine", "abs machine"],
    "rowing": ["rowing", "rower"],
    "cable": ["cable"],
    "tower": ["tower"],
    "barbell-compatible": ["barbell"],
    "gym-grade": ["commercial", "gym-grade"],
    "joint-friendly": ["joint", "rehab", "recovery"],
"post-injury": ["post-injury", "rehab"],
"low-impact": ["low-impact", "joint-friendly", "gentle"],
"elderly": ["elderly", "senior"],
}

def extract_tags(option):
    fields = [
        option.get("equipment_name", ""),
        option.get("brand", ""),
        option.get("model", ""),
        option.get("color", ""),
        option.get("material", "")
    ]
    attributes = option.get("attributes", {})
    fields.extend(attributes.values())

    full_text = " ".join(fields).lower()
    tags = set()

    for tag, keywords in TAG_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", full_text):
                tags.add(tag)

    return sorted(tags)


with open(input_path, "r") as f:
    equipment_options = json.load(f)

for opt in equipment_options:
    opt["tags"] = extract_tags(opt)

with open(output_path, "w") as f:
    json.dump(equipment_options, f, indent=2)

output_path
