import pandas as pd

def prepare_options_dataframe(options):
    rows = []
    for opt in options:
        rows.append({
            "option_id": str(opt.get("option_id")),
            "equipment_id": opt.get("equipment_id"),
            "tags": {t.get("name", "").lower() for t in opt.get("tags", []) if isinstance(t, dict)},
            "attrs": {a.lower() for a in opt.get("attribute_values", []) if isinstance(a, str)},
            "text": opt.get("_preprocessed_text", "").lower(),
            "price": opt.get("price", 0),
            "weight": opt.get("weight", 0),
            "data": opt  # hold full object for later use
        })
    return pd.DataFrame(rows)
