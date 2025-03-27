# 🧠 E-cercise Recommender Service

This is the personalized recommendation microservice for **E-cercise**, powered by FastAPI and Sentence Transformers.

---

## 🚀 Features

- Equipment option recommendations based on:
  - User goal, weight, height, gender, age
  - Experience level (e.g. Beginner, Athlete)
  - Tag-based preferences
- Embedding-based vector similarity with fallback scoring
- CORS-ready for frontend integration
- Fully containerized with Docker

---

## 🛠️ Project Structure

```
.
├── api/                    # FastAPI router
├── services/               # Recommendation logic
├── utils/                  # Vector handling, text utils
├── models/                 # Pydantic schema models
├── data/                   # Precomputed vector cache + equipment options
├── generate_equipment_vector_cache.py  # Precomputes vector cache
├── main.py                 # FastAPI entrypoint
├── config.py               # Loads .env config
├── .env.example            # Example config file
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container setup
```

---

## ⚙️ Configuration

Copy `.env.example` → `.env` and fill in your settings:

```bash
cp .env.example .env
```

---

## 🐳 Docker Usage

### Build

```bash
docker build -t recommender .
```

### Run

```bash
docker run --env-file .env -p 8000:8000 recommender
```

---

## 📦 Regenerate Vector Cache (if equipment changed)

```bash
python generate_equipment_vector_cache.py
```

---

## 🔌 Example API Request

```http
POST /recommend
Content-Type: application/json

{
  "goal": "weight-loss",
  "experience": "beginner",
  "gender": "female",
  "age": 25,
  "preferences": [
    { "tag": "glutes", "group": "muscle" }
  ]
}
```

Returns top 10 scored equipment options 🎯

---

## 🧪 Dev Mode

```bash
uvicorn main:app --reload
```

---

## 🤝 License

MIT
