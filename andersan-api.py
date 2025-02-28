# 大幅にandersan/をリファクタリングしたので、調整が必要。

import datetime
import math
import os
from functools import lru_cache
from logging import getLogger
from typing import List, Union, Literal
from logging import getLogger, basicConfig, INFO, DEBUG

import numpy as np

# import tenbou
import uvicorn

# from algorithms import algorithms
# from cache import cache
from fastapi import Depends, FastAPI, Request, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
import andersan
import andersan.airmonitor
import predict
import json

# class Predict(BaseModel):
#     loc: int = Field(example=14131030)
#     date: Union[datetime.datetime, List] # 時刻を複数与えてもいい。
#     algorithm: str = Field(example="mm_E")
#     fmax: int = Field(example=24)
#     col: str = Field(example="OX")


app = FastAPI()

origins = [
    "*",
    "http://localhost",
    "http://172.23.78.218:8088",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ITEMS = ["NMHC", "OX", "NOX", "TEMP", "WX", "WY"]

ITEMSPECS = {
    "NMHC": {"desc": "Non-methane hydrocarbons", "unit": "10ppbC", "range": [0, 100]},
    "OX": {"desc": "Oxidants", "unit": "ppb", "range": [0, 100]},
    "NOX": {"desc": "Nitrogen oxides", "unit": "ppb", "range": [0, 100]},
    "TEMP": {"desc": "Temperature", "unit": "0.1C", "range": [0, 250]},
    "WX": {
        "desc": "X component of the window speed",
        "unit": "0.1m/s",
        "range": [0, 100],
    },
    "WY": {
        "desc": "Y component of the window speed",
        "unit": "0.1m/s",
        "range": [0, 100],
    },
}


@app.get("/raw/{prefecture}/{datehour}")
async def raw_data(
    prefecture: Literal[tuple(andersan.airmonitor.prefecture_retrievers)],
    datehour: datetime.datetime,
):
    """県内の全測定局の実測値を返す。

    Args:
    -   prefecture (str): 県名 ["kanagawa"]
    -   datehour (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"] 正時にそろえられ、分以下は無視されます。

    Returns:
    -   _str_: 県提供の大気監視データ
    """
    if prefecture not in andersan.airmonitor.prefecture_retrievers:
        raise HTTPException(status_code=404, detail="Out of the cover area")
    # もうちょっと補助情報も出さないと使えないよ。
    # APIを叩く側はJSなので、JSONにしておくほうが便利。
    isodate = datetime.datetime.isoformat(datehour)
    try:
        raw_data = andersan.airmonitor.prefecture_retrievers[prefecture].retrieve(
            isodate, station_set="air"
        )
    except:
        raise HTTPException(status_code=404, detail="Data not available.")

    dict_data = dict(data=raw_data.to_dict(), spec={})
    return Response(content=json.dumps(dict_data, indent=2, ensure_ascii=False))


def dictize(df, items=[]):
    spec = ITEMSPECS.copy()
    loc = ("X", "Y", "lon", "lat", "Z")
    for col in loc:
        spec[col] = sorted(df[col].unique().tolist())
    spec["timestamp"] = sorted(df.index.unique().map(lambda x: int(x.timestamp())))
    spec["items"] = items

    data = dict()
    data["XY"] = df[["X", "Y"]].to_numpy().tolist()
    data["lon"] = df[["lon", "lat"]].to_numpy().tolist()
    cols = df.columns
    for col in cols:
        if col not in loc:
            data[col] = list(df[col])
    return dict(spec=spec, data=data)


@app.get("/tile/{zoom}/{prefecture}/{datehour}")
async def tile_data(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: datetime.datetime,
    zoom: int,
):
    """県内のタイル点での実測値を返す。

    Args:
    -   zoom (int): 地理院タイルのzoom値
    -   prefecture (str): 県名 ["kanagawa"]
    -   datehour (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"] 正時にそろえられ、分以下は無視されます。

    Returns:
    -   _str_: 実測値
    """
    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")
    # もうちょっと補助情報も出さないと使えないよ。
    # APIを叩く側はJSなので、JSONにしておくほうが便利。
    isodate = datetime.datetime.isoformat(datehour)
    raw_data = andersan.airmonitor.tiles(prefecture, isodate, zoom, items=ITEMS)
    if raw_data is None:
        raise HTTPException(status_code=404, detail="Data not available")
    # 付加情報を添える。単位なども必要。
    data = dictize(raw_data, items=ITEMS)
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False))



@app.get("/ox/{model}/{prefecture}/{datehour}")
async def predict_Ox(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: datetime.datetime,
    model: str,
):
    """県内のタイル点でのOX予測値を返す。

    Args:
    -   prefecture (str): 県名 ["kanagawa"]
    -   datehour (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"] 正時にそろえられ、分以下は無視されます。
    -   model (str): 予測モデル。 "v0":andersan0_1(12th tile, 8 hours ahead.)

    Returns:
    -   _str_: 県内の地理院タイル点でのOxの予測値。
    """

    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")

    # もうちょっと補助情報も出さないと使えないよ。
    # APIを叩く側はJSなので、JSONにしておくほうが便利。
    isodate = datetime.datetime.isoformat(datehour)

    if model == "v0":
        foreseer = predict.Foreseer_v0()
    elif model == "v1":
        foreseer = predict.Foreseer_v1()

    raw_data = foreseer.predict_ox(prefecture, isodate)

    if raw_data is None:
        raise HTTPException(status_code=404, detail="Data not available")
    data = dictize(raw_data)
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import os

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    basicConfig(level=DEBUG)
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(
        "andersan-api:app",
        host="0.0.0.0",
        port=8087,
        reload=True,
    )
