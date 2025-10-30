import numpy as np

def create_sequences(data, window_size, horizon):             
    X = []                                           
    # 전체 데이터에서 윈도우를 만들 수 있는 만큼 반복
    for i in range(len(data) - window_size - horizon + 1):
        window = data[i:(i + window_size)]           
        X.append(window)  
    return np.array(X)

def preprocess(usage_data: list[float]) -> np.ndarray:
    """
    클라이언트에서 받은 입력값(usage_data)을
    모델이 예측할 수 있는 형태로 전처리합니다.
    """
    # train_model.py와 동일한 조건으로 설정
    window_size = 4320
    horizon = 8760

    # numpy array로 변환
    usage_arr = np.array(usage_data, dtype=np.float32)

    # 예측에 사용할 입력 시퀀스 생성
    X = create_sequences(usage_arr, window_size, horizon)
    
    return X