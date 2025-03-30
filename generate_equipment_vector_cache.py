import json
from sentence_transformers import SentenceTransformer
from utils.text import clean_text
from utils.vector_cache import save_vector_cache

model = SentenceTransformer("all-MiniLM-L6-v2")

INPUT_JSON_PATH = "data/equipment_options_with_tags.json"
OUTPUT_JSON_PATH = "data/equipment_options_with_tags.json"  # Overwrite with preprocessed data

def generate_vector_cache():
    with open(INPUT_JSON_PATH, "r") as file:
        equipment_options = json.load(file)

    vector_cache = {}

    for option in equipment_options:
        option_id = option.get("option_id")
        if not option_id:
            continue

        # Safely extract attributes
        attribute_values = option.get("attribute_values", [])
        if isinstance(attribute_values, dict):  # fallback safety
            attribute_values = list(attribute_values.values())

        # Extract tag names
        tags = [t.get("name", "") for t in option.get("tags", []) if isinstance(t, dict)]

        fields = [
            option.get("equipment_name", ""),
            option.get("brand", ""),
            option.get("model", ""),
            option.get("color", ""),
            option.get("material", ""),
        ] + attribute_values + tags

        full_text = clean_text(" ".join(fields))
        option["_preprocessed_text"] = full_text  # 🔥 Used for fast scoring

        vector_cache[option_id] = model.encode(full_text, convert_to_tensor=False).tolist()

    # Save updated equipment options JSON with `_preprocessed_text`
    with open(OUTPUT_JSON_PATH, "w") as file:
        json.dump(equipment_options, file, indent=2)

    # Save vector cache
    save_vector_cache(vector_cache)

if __name__ == "__main__":
    generate_vector_cache()
