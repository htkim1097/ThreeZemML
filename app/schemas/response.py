from pydantic import BaseModel
from typing import List, Dict, Any

class PredictResponse(BaseModel):
    elecPredictions: List[Dict[str, Any]]
    gasPredictions: List[Dict[str, Any]]
    waterPredictions: List[Dict[str, Any]]