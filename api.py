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
from andersan_api.backend import prefecture_retrievers, observes, Neighbors
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


@app.get("/raw/{prefecture}/{datehour}")
async def raw_data(
    prefecture: Literal[tuple(prefecture_retrievers)], datehour: datetime.datetime
):
    """県内の全測定局の実測値を返す。

    Args:
    -   prefecture (str): 県名 ["kanagawa"]
    -   isodate (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"]

    Returns:
    -   _str_: 県提供の大気監視データ
    """
    if prefecture not in prefecture_retrievers:
        raise HTTPException(status_code=404, detail="Out of the cover area")
    # もうちょっと補助情報も出さないと使えないよ。
    # APIを叩く側はJSなので、JSONにしておくほうが便利。
    isodate = datetime.datetime.isoformat(datehour)
    raw_data = prefecture_retrievers[prefecture].retrieve(isodate, station_set="air")

    dict_data = dict(data=raw_data.to_dict(), spec={})
    return Response(content=json.dumps(dict_data, indent=2, ensure_ascii=False))


def dictize(df):
    spec = dict()
    loc = ("X", "Y", "lon", "lat", "Z")
    for col in loc:
        spec[col] = sorted(df[col].unique().tolist())
    spec["timestamp"] = sorted(df.index.unique().map(lambda x: int(x.timestamp())))

    data = dict()
    data["XY"] = df[["X", "Y"]].to_numpy().tolist()
    data["lon"] = df[["lon", "lat"]].to_numpy().tolist()
    cols = df.columns
    for col in cols:
        if col not in loc:
            data[col] = list(df[col])
    return dict(spec=spec, data=data)


@app.get("/tile/{prefecture}/{datehour}/{zoom}")
async def tile_data(
    prefecture: Literal[tuple(prefecture_retrievers)],
    datehour: datetime.datetime,
    zoom: int,
):
    """県内のタイル点での実測値を返す。

    Args:
    -   prefecture (str): 県名 ["kanagawa"]
    -   isodate (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"]
    -   zoom (int): 地理院メッシュのzoom値

    Returns:
    -   _str_: 実測値
    """
    if prefecture not in Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")
    # もうちょっと補助情報も出さないと使えないよ。
    # APIを叩く側はJSなので、JSONにしておくほうが便利。
    isodate = datetime.datetime.isoformat(datehour)
    raw_data = observes(prefecture, isodate, zoom)
    if raw_data is None:
        raise HTTPException(status_code=404, detail="Data not available")
    data = dictize(raw_data)
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False))


def predict(prefecture, isodate, zoom):
    X = X_instant(prefecture, isodate, zoom)
    basename = "../andersan-train/andersan0_1.py"
    model = keras.models.load_model(f"{basename}.best.keras")
    pred = model.predict(X)
    return pred**0.5


# @app.get("/ox/{prefecture}/{datehour}/{zoom}")
# async def predict_ox(
#     prefecture: Literal[tuple(prefecture_retrievers)],
#     datehour: datetime.datetime,
#     zoom: int,
# ):
#     """県内のタイル点でのOX予測値を返す。

#     Args:
#     -   prefecture (str): 県名 ["kanagawa"]
#     -   isodate (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"]
#     -   zoom (int): 地理院メッシュのzoom値

#     Returns:
#     -   _str_: 実測値
#     """
#     if prefecture not in Neighbors:
#         raise HTTPException(status_code=404, detail="Out of the cover area")
#     isodate = datetime.datetime.isoformat(datehour)

#     y = predict(prefecture, isodate, zoom)
#     if y is None:
#         raise HTTPException(status_code=404, detail="Data not available")


#     data = dictize(raw_data)
#     return Response(content=json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    basicConfig(level=DEBUG)
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8087,
        reload=True,
    )
