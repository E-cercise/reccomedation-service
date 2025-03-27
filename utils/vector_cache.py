import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("data/equipment_options.json") as f:
    EQUIPMENT_DATA = json.load(f)

with open("data/equipment_vector_cache.json") as f:
    cache = json.load(f)

VECTORS = np.array(cache["vectors"])
IDS = cache["ids"]
TEXTS = cache["texts"]

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
