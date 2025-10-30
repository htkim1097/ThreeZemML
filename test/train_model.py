import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings
import joblib
from pathlib import Path

# 전력 예측
# window : 4320(6개월)
# horizon : 8760(12개월)
# model : lgbm 
# rmse : 1.449615

# 가스 예측
# window : 270(11.25일)
# horizon : 8760
# model : lgbm
# rmse : 0.1778

# 수도 예측
# window : 168
# horizon : 8760
# model : lgbm
# rmse : 0.02945

warnings.filterwarnings("ignore", message="X does not have valid feature names")

def create_sequences(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        window = data[i:(i + window_size)]
        target = data[i + window_size + horizon - 1]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

# 데이터 로드
host = "localhost"
port = "7770"
baseUrl = f"http://{host}:{port}"
url = baseUrl + "/api/energy/water"
params = {"start": "2022-10-01 00:00:00", "end": "2025-10-28 00:00:00", "datetimeType": 0}
response = requests.get(url, params=params)

json_data = response.json()

# 데이터프레임으로 변환
timestamp = [row['timestamp'] for row in json_data['datas']]
usage = [row['usage'] for row in json_data['datas']]

data = {
    "timestamp": pd.to_datetime(timestamp),
    "usage": usage
}
df = pd.DataFrame(data)
df.set_index('timestamp', inplace=True)

# print(df.count())

# window_size_list = [168, 192, 270, 540, 1080, 2160, 4320, 8760, 17520]
# horizon_list = [1080, 2160, 4320, 8760]

# cnt = 0
# pred_list = []
# for i in window_size_list:
#     temp = {}
#     for j in horizon_list:

#         # 학습 데이터 준비
#         window_size = i
#         horizon = j

#         usage_data = df['usage'].values
#         X, y = create_sequences(usage_data, window_size, horizon)

#         # 데이터셋이 충분히 큰 경우에만 분할하여 테스트합니다.
#         if len(X) > 1:
#             split_ratio = 0.8
#             split_index = int(len(X) * split_ratio)
#             if split_index == 0:
#                 split_index = 1

#             X_train, X_test = X[:split_index], X[split_index:]
#             y_train, y_test = y[:split_index], y[split_index:]
#         else:
#             X_train, y_train = X, y
#             X_test, y_test = X, y # 테스트 데이터가 없는 경우

        
#         naive_preds = X_test[:, -1]  # 예측값 = 입력 데이터의 가장 마지막 값
#         naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
#         temp["naive"] = naive_rmse

#         lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
#         lgbm.fit(X_train, y_train)
#         lgbm_preds_train = lgbm.predict(X_train)
#         lgbm_rmse_train = np.sqrt(mean_squared_error(y_train, lgbm_preds_train))
#         lgbm_preds_test = lgbm.predict(X_test)
#         lgbm_rmse_test = np.sqrt(mean_squared_error(y_test, lgbm_preds_test))
#         temp["lgbm_train"] = lgbm_rmse_train
#         temp["lgbm_test"] = lgbm_rmse_test
        
#         xgbr = xgb.XGBRegressor(random_state=42)
#         xgbr.fit(X_train, y_train)
#         xgb_preds_train = xgbr.predict(X_train)
#         xgb_rmse_train = np.sqrt(mean_squared_error(y_train, xgb_preds_train))
#         xgb_preds_test = xgbr.predict(X_test)
#         xgb_rmse_test = np.sqrt(mean_squared_error(y_test, xgb_preds_test))
#         temp["xgb_train"] = xgb_rmse_train
#         temp["xgb_test"] = xgb_rmse_test

#         cnt += 1
#         print(f"{cnt} => {window_size=}, {horizon=}, naive : {str(temp['naive'])}, lgbm_train : {str(temp['lgbm_train'])}, lgbm_test : {str(temp['lgbm_test'])}, xgb_train : {str(temp['xgb_train'])}, xgb_test : {str(temp['xgb_test'])}")

# print(df.describe())
# print(pred_list.describe())


window_size = 168
horizon = 8760

usage_data = df['usage'].values
X, y = create_sequences(usage_data, window_size, horizon)

# 데이터셋이 충분히 큰 경우에만 분할하여 테스트합니다.
if len(X) > 1:
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    if split_index == 0:
        split_index = 1

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
else:
    X_train, y_train = X, y
    X_test, y_test = X, y # 테스트 데이터가 없는 경우


lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
lgbm.fit(X_train, y_train)

# 모델 저장
model_path = Path(__file__).resolve().parent.parent / "app" / "models" / "lgbm_model.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(lgbm, model_path)

print(f"모델이 다음 경로에 저장되었습니다: {model_path}")
