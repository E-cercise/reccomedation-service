from sentence_transformers import SentenceTransformer
import json
import uuid
import os

def build_equipment_text(option):
    tags = [
        f"{tag['group']}:{tag['name']}" if isinstance(tag, dict) and 'group' in tag else tag
        for tag in option.get("tags", [])
    ]
    attrs = option.get("attribute_values", [])
    return " ".join(tags + attrs)

with open("data/equipment_options.json") as f:
    data = json.load(f)

texts = [build_equipment_text(opt) for opt in data]
ids = [opt.get("id") or opt.get("equipment_option_id") or str(uuid.uuid4()) for opt in data]

model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = model.encode(texts, convert_to_tensor=False)

with open("data/equipment_vector_cache.json", "w") as f:
    json.dump({
        "ids": ids,
        "texts": texts,
        "vectors": [v.tolist() for v in vectors]
    }, f)

print("✅ equipment_vector_cache.json generated")