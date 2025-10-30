from pydantic import BaseModel
from app.schemas.readings import EnergyReadings

class PredictRequest(BaseModel):
    electricityReadings: EnergyReadings
    gasReadings: EnergyReadings
    waterReadings: EnergyReadings

