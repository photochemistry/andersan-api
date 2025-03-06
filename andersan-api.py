# 大幅にandersan/をリファクタリングしたので、調整が必要。

import datetime
import os
from typing import List, Union, Literal
from logging import getLogger, basicConfig, INFO, DEBUG

import numpy as np

import uvicorn

from fastapi import Depends, FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import andersan
import andersan.airmonitor
import predict
import json


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


class InvalidModelException(Exception):
    """モデル指定がおかしい場合の例外"""

    def __init__(self, model):
        self.message = f"Model '{model}' is not available."
        super().__init__(self.message)


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
    -   datehour (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"] 正時にそろえられ、分以下は無視されます。あるいは、"now"で現時点での予測を返します。
    -   model (str): 予測モデル。
            "v0":andersan0_1(12th tile, 8 hours ahead.)
            "v0":andersan0_1(12th tile, 8 hours ahead.)
            "v0":andersan0_1(12th tile, 8 hours ahead.)

    Returns:
    -   _str_: 県内の地理院タイル点でのOxの予測値。
    """

    # logger = getLogger()

    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")

    if datehour == "now":
        isodate = "now"
    else:
        isodate = datetime.datetime.isoformat(datehour)

    # prediction function switcher
    if model == "v0":
        predict_ox = predict.predict_ox_v0
    elif model == "v0a":
        predict_ox = predict.predict_ox_v0a
    elif model == "v1":
        predict_ox = predict.predict_ox_v1
    elif model == "v1a":
        predict_ox = predict.predict_ox_v1a
    else:
        raise InvalidModelException(model)

    raw_data = predict_ox(prefecture, isodate)
    # logger.debug(raw_data)

    if raw_data is None:
        raise HTTPException(status_code=404, detail="Data not available")
    data = dictize(raw_data)
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False))


# @app.get("/oxnow/{model}/{prefecture}")
# async def predict_Ox_now(
#     prefecture: Literal[tuple(andersan.Neighbors)],
#     model: str,
# ):
#     """県内のタイル点でのOX予測値を返す。

#     Args:
#     -   prefecture (str): 県名 ["kanagawa"]
#     -   model (str): 予測モデル。 "v0":andersan0_1(12th tile, 8 hours ahead.)

#     Returns:
#     -   _str_: 県内の地理院タイル点でのOxの予測値。
#     """

#     # logger = getLogger()

#     if prefecture not in andersan.Neighbors:
#         raise HTTPException(status_code=404, detail="Out of the cover area")

#     isodate = "now"

#     # prediction function switcher
#     if model == "v0":
#         predict_ox = predict.predict_ox_v0
#     elif model == "v0a":
#         predict_ox = predict.predict_ox_v0a
#     elif model == "v1":
#         predict_ox = predict.predict_ox_v1
#     elif model == "v1a":
#         predict_ox = predict.predict_ox_v1a
#     else:
#         raise InvalidModelException(model)

#     raw_data = predict_ox(prefecture, isodate)
#     # logger.debug(raw_data)

#     if raw_data is None:
#         raise HTTPException(status_code=404, detail="Data not available")
#     data = dictize(raw_data)
#     return Response(content=json.dumps(data, indent=2, ensure_ascii=False))


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
