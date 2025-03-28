import json
from sentence_transformers import SentenceTransformer
from utils.text import clean_text
from utils.vector_cache import save_vector_cache

model = SentenceTransformer("all-MiniLM-L6-v2")
INPUT_JSON_PATH = "data/equipment_options_with_tags.json"

def generate_vector_cache():
    with open(INPUT_JSON_PATH, "r") as file:
        equipment_options = json.load(file)

    vector_cache = {}

    for option in equipment_options:
        option_id = option.get("option_id")
        if not option_id:
            continue

        fields = [
            option.get("equipment_name", ""),
            option.get("brand", ""),
            option.get("model", ""),
            option.get("color", ""),
            option.get("material", "")
        ]
        fields += list(option.get("attributes", {}).values())
        fields += option.get("tags", [])

        full_text = clean_text(" ".join(fields))
        vector_cache[option_id] = model.encode(full_text, convert_to_tensor=False).tolist()

    save_vector_cache(vector_cache)

if __name__ == "__main__":
    generate_vector_cache()
