import orjson as json
import os

VECTOR_CACHE_PATH = "data/vector_cache.json"

def save_vector_cache(cache):
    if not cache:
        print("Skipping save_vector_cache because cache is empty.")
        return
    with open("data/vector_cache.json", "wb") as f:
        f.write(json.dumps(cache))
        


def load_vector_cache():
    if not os.path.exists(VECTOR_CACHE_PATH):
        return {}
    with open(VECTOR_CACHE_PATH, "rb") as f:
        return json.loads(f.read())

def get_vector_for_option(option_id, vector_cache):
    return vector_cache.get(option_id)

def get_all_vectors():
    return load_vector_cache()
