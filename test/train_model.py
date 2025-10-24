import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings

import joblib
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

def create_sequences(data, window_size, horizon):             
    X, y = [], []                                           
    # 전체 데이터에서 윈도우와 타겟을 만들 수 있는 만큼 반복
    for i in range(len(data) - window_size - horizon + 1):
        window = data[i:(i + window_size)]           
        target = data[i + window_size + horizon - 1] 
        X.append(window)  
        y.append(target)  
    return np.array(X), np.array(y)

# 데이터 로드
host = "http://localhost"
port = "7770"
baseUrl = f"{host}:{port}"
url = baseUrl + "/api/energy/elec"
params = { "start": "2023-10-23 12:00:00", "end": "2024-10-22 12:00:00", "datetimeType": 0}
response = requests.get(url, params=params)

if (str(response.status_code) == "200"):
    json_data = response.json()

# 데이터프레임으로 변환
timestamp = []
usage = []

for row in json_data['datas']:
    timestamp.append(row['timestamp'])
    usage.append(row['usage'])

data = {
    "timestamp" : pd.to_datetime(timestamp),
    "usage" : usage
}

df = pd.DataFrame(data)
df.set_index('timestamp')

results = []
cnt = 0
for i in range(1, 1095):
    temp = {}
    for j in range(1, 365):
        if i < j:
            continue
        # 데이터 분할(슬라이딩 윈도우)
        window_size = i  # 과거 데이터를 보고
        horizon = j  # 다음의 데이터를 예측
        temp["window"] = i
        temp["horizon"] = j

        usage_data = df['usage'].values
        X, y = create_sequences(usage_data, window_size, horizon)

        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        if split_index == 0 and len(X) > 0:
            split_index = 1

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # 학습
        # 베이스 라인 모델(Naive Forecast: 직전 값이 다음 값이 될 것이다)
        naive_preds = X_test[:, -1]  # 예측값 = 입력 데이터의 가장 마지막 값
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
        temp["naive"] = naive_rmse

        # lightGBM
        lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
        lgbm.fit(X_train, y_train)
        lgbm_preds = lgbm.predict(X_test)
        lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_preds))
        temp["lgbm"] = lgbm_rmse

        # XGBoost
        xgbr = xgb.XGBRegressor(random_state=42)
        xgbr.fit(X_train, y_train)
        xgb_preds = xgbr.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
        temp["xgb"] = xgb_rmse

        cnt += 1
        print(f"{cnt}: {temp}")
        results.append(temp)

res_df = pd.DataFrame(results)
print(res_df.describe())


# # 모델 저장
# model_path = Path("../app/models/model.pkl")
# model_path.parent.mkdir(parents=True, exist_ok=True)
# joblib.dump(pipeline, model_path)

# print(f"Model saved to {model_path}")