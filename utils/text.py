import re

def build_user_text(req):
    parts = [
        f"goal:{req.goal}" if req.goal else "",
        f"experience:{req.experience}" if req.experience else "",
        f"gender:{req.gender}" if req.gender else ""
    ]
    parts += [
        f"{p.group}:{p.tag}" if p.group else p.tag
        for p in req.preferences or []
        if p.tag
    ]
    return " ".join([p for p in parts if p])

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9\\s]", "", text).lower().strip()