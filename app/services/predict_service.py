from app.models.model_loader import get_model
from app.schemas.readings import EnergyReadings
from app.utils.preprocessor import preprocess
from app.utils.postprocessor import postprocess

def predict_service(energy_type: str, readings: EnergyReadings):
    # lgbm 모델 파일명을 전달
    model = get_model(f"{energy_type}_model.pkl")
    
    # request에서 usage_data를 전처리기에게 전달
    x = preprocess([d.usage for d in readings.datas])
    
    # 전처리된 데이터로 예측 수행
    pred = model.predict(x)
    
    # 예측 결과를 후처리
    result = postprocess(pred)

    print(f"[INFO] {energy_type} 예측 완료")
    
    return result