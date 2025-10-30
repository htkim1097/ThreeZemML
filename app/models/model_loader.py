import joblib
from pathlib import Path

_model = None

def get_model(modelFileName:str):
    global _model
    if _model is None:
        model_path = Path(__file__).resolve().parent / modelFileName
        _model = joblib.load(model_path)
    return _model