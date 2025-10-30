from pydantic import BaseModel
from typing import List
from datetime import datetime

class Reading(BaseModel):
    timestamp: datetime
    usage: float

class EnergyReadings(BaseModel):
    datas: List[Reading]
