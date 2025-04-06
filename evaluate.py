from models.schema import RecommendRequest
from services.recommender import evaluate_recommender

req = RecommendRequest(
    goal="weight-loss",
    preferences=[{"tag": "cardio", "group": "goal"}, {"tag": "compact"}],
    gender="female",
    experience="beginner",
    age=28,
    height=165,
    weight=60,
    user_type="user"
)

modes = ["embedding_only", "rule_only", "hybrid"]
results = {}

for m in modes:
    print(f"\n🔎 Evaluating mode: {m}")
    top_items = evaluate_recommender(req, mode=m, top_k=10, plot=True)
    results[m] = top_items
    for i, opt in enumerate(top_items[:5]):
        print(f"{i+1}. {opt.get('name')} — score: {opt.get('score'):.2f}")
