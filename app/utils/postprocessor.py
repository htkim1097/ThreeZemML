import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

def postprocess(prediction: np.ndarray) -> list[dict[str, any]]:
    """
    모델 예측 결과(numpy array)를 일별로 합산하고, 서버 DTO 형식에 맞게 JSON으로 변환 가능한 리스트로 변환.
    """
    
    # 소수점 4자리까지 반올림
    prediction = np.round(prediction, 4)
    
    # 현재 시간을 기준으로 1시간씩 증가하는 타임스탬프 생성
    now = datetime.now()
    time_stamps = [now + timedelta(hours=i) for i in range(len(prediction))]
    
    # 일별 사용량 합산
    daily_usage = defaultdict(float)
    for ts, p in zip(time_stamps, prediction):
        daily_usage[ts.date()] += p.item()
        
    # 결과를 딕셔너리 리스트로 변환
    result = []
    for d, u in sorted(daily_usage.items()):
        dt = datetime.combine(d, datetime.min.time())
        result.append({
            "timestamp": dt.strftime("%Y-%m-%d 00:00:00"),
            "usage": round(u, 4)
        })
        
    return result