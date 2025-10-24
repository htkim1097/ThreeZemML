from typing import Union
import requests
from fastapi import FastAPI
from app.api.router import router as api_router
from contextlib import asynccontextmanager
import pandas as pd

async def init():
    host = "http://localhost"
    port = "7770"
    baseUrl = f"{host}:{port}"
    url = baseUrl + "/api/energy/elec"
    params = { "start": "2024-10-23 12:00:00", "end": "2025-10-23 12:00:00", "datetimeType": 0}
    response = requests.get(url, params=params)

    if (str(response.status_code) == "200"):
        json_data = response.json()
        df = pd.DataFrame(json_data["datas"])
        df


    


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 실행 시 실행될 코드
    await init()
    yield 
    
    # 서버 종료 시 실행될 코드

app = FastAPI(lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"It's": "Running"}
