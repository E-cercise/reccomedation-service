from fastapi import APIRouter
from models.schema import RecommendRequest
from services.recommender import get_recommendations

router = APIRouter()

@router.post("/recommend")
async def recommend(req: RecommendRequest):
    "recommend the first 100 equipment option based on scoring"
    return get_recommendations(req)
