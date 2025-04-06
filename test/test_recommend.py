import requests
import json
import pandas as pd

API_URL = "http://localhost:8000/recommend"
HEADERS = {"Content-Type": "application/json"}

test_cases = None
with open("./recommendation_test_cases.json") as f:
    test_cases = json.load(f)

results_data = []


def determine_user_feedback(score, emb_sim, equip_name, option_name, rule_applied):
    if score > 7 and emb_sim >= 1.0 and "Matched tags" in rule_applied:
        return "Highly Relevant"
    elif score > 5:
        if "Matched tags" in rule_applied or equip_name.lower() in option_name.lower():
            return "Relevant"
        return "Somewhat Relevant"
    elif score > 3:
        return "Marginal"
    else:
        return "Irrelevant"


for case in test_cases:
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(case["input"]))

    if response.status_code == 200:
        res = response.json()
        if not res:
            results_data.append({
                "Test Case ID": case["id"],
                "Test Case Name": case.get("name", f"Test Case {case['id']}"),
                "Test Case Input": json.dumps(case["input"]),
                "Output (Equipment Name)": "N/A",
                "Rule Used": "No recommendations returned",
                "Embedding Similarity": "N/A",
                "User Feedback": "Irrelevant"
            })
            continue

        for rank_index, r in enumerate(res[:3]):  # Top 3 only
            debug = r.get("__debug", {})
            emb_sim = debug.get("embedding_similarity", 0)
            score = r.get("score", 0)
            rule_applied = r.get("rule_applied", "N/A")

            equip_name = r.get("equipment_name", "")
            option_name = equip_name
            feedback = determine_user_feedback(score, emb_sim, equip_name, option_name, rule_applied)

            results_data.append({
                "Test Case ID": case["id"],
                "Test Case Name": case.get("name", f"Test Case {case['id']}"),
                "Test Case Input": json.dumps(case["input"]),
                "Output (Equipment Name)": equip_name,
                "Rule Used": rule_applied,
                "Embedding Similarity": emb_sim,
                "User Feedback": feedback
            })
    else:
        results_data.append({
            "Test Case ID": case["id"],
            "Test Case Name": case.get("name", f"Test Case {case['id']}"),
            "Test Case Input": json.dumps(case["input"]),
            "Output (Equipment Name)": "N/A",
            "Rule Used": f"HTTP Error {response.status_code}",
            "Embedding Similarity": "N/A",
            "User Feedback": "Error"
        })

# Save to CSV
df = pd.DataFrame(results_data)
csv_path = "./recommendation_testcase_output.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Results saved to {csv_path}")
