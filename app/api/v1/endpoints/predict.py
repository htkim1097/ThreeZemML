from fastapi import APIRouter
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.predict_service import predict_service

router = APIRouter()

@router.post("/", response_model=PredictResponse)
def predict(request: PredictRequest):
    elec_prediction = predict_service('elec', request.electricityReadings)
    gas_prediction = predict_service('gas', request.gasReadings)
    water_prediction = predict_service('water', request.waterReadings)
    
    return PredictResponse(
        elecPredictions=elec_prediction,
        gasPredictions=gas_prediction,
        waterPredictions=water_prediction
    )