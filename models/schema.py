from pydantic import BaseModel
from typing import List, Optional

class Preference(BaseModel):
    tag: Optional[str] = None
    group: Optional[str] = None
    max_price: Optional[float] = None
    min_weight: Optional[float] = None

class RecommendRequest(BaseModel):
    user_type: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    goal: Optional[str] = None
    experience: Optional[str] = None
    preferences: Optional[List[Preference]] = []
