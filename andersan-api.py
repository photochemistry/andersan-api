# 大幅にandersan/をリファクタリングしたので、調整が必要。

import datetime
import os
from typing import Literal, Union
from logging import basicConfig, DEBUG, getLogger
import pandas as pd
import uvicorn
import time
import pytz

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import andersan
import andersan.airmonitor
from andersan_core import predict
import json

# ログ設定
basicConfig(
    level=DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = getLogger(__name__)
logger.setLevel(DEBUG)

# sqlitedictのログも有効化
sqlitedict_logger = getLogger('sqlitedict')
sqlitedict_logger.setLevel(DEBUG)

app = FastAPI()

# 処理時間計測用のミドルウェア
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.debug(f"Processing time: {request.url.path} - {process_time:.3f} seconds")
    return response

# CORS設定
origins = [
    "*",
    "http://localhost:8087",
    "http://172.23.78.207:8087",
    "http://192.168.3.234:8087",
    "http://172.23.78.44:8080",
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


# @app.get("/raw/{prefecture}/{datehour}")
# async def raw_data(
#     prefecture: Literal[tuple(andersan.airmonitor.prefecture_retrievers)],
#     datehour: datetime.datetime,
# ):
#     """県内の全測定局の実測値を返す。

#     Args:
#     -   prefecture (str): 県名 ["kanagawa"]
#     -   datehour (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"] 正時にそろえられ、分以下は無視されます。

#     Returns:
#     -   _str_: 県提供の大気監視データ
#     """
#     start_time = time.time()
    
#     if prefecture not in andersan.airmonitor.prefecture_retrievers:
#         raise HTTPException(status_code=404, detail="Out of the cover area")
    
#     datehour = datehour.replace(minute=0, second=0, microsecond=0)
#     isodate = datetime.datetime.isoformat(datehour)
#     try:
#         raw_data = andersan.airmonitor.prefecture_retrievers[prefecture].retrieve(
#             isodate, station_set="air"
#         )
#     except:
#         raise HTTPException(status_code=404, detail="Data not available.")

#     dict_data = dict(data=raw_data.to_dict(), spec={})
    
#     process_time = time.time() - start_time
#     logger.debug(f"raw_data internal processing time: {process_time:.3f} seconds")
    
#     return Response(content=json.dumps(dict_data, indent=2, ensure_ascii=False))


def dictize(df, items=[]):
    spec = ITEMSPECS.copy()
    loc = ("X", "Y", "lon", "lat", "Z")
    for col in loc:
        spec[col] = sorted(df[col].unique().tolist())
    spec["timestamp"] = sorted(df.index.unique().map(lambda x: int(x.timestamp())))
    spec["items"] = items

    data = dict()
    data["XY"] = df[["X", "Y"]].to_numpy().tolist()
    data["lon"] = df["lon"].tolist()
    data["lat"] = df["lat"].tolist()
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
    start_time = time.time()
    
    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")
    
    datehour = datehour.replace(minute=0, second=0, microsecond=0)
    isodate = datetime.datetime.isoformat(datehour)
    raw_data = andersan.airmonitor.tiles(prefecture, isodate, zoom, items=ITEMS)
    if raw_data is None:
        raise HTTPException(status_code=404, detail="Data not available")
    
    data = dictize(raw_data, items=ITEMS)
    
    process_time = time.time() - start_time
    logger.debug(f"tile_data internal processing time: {process_time:.3f} seconds")
    
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False))


@app.get("/ox/{model}/{prefecture}/{datehour}")
async def predict_Ox(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
    model: str,
):
    """県内のタイル点でのOX予測値を返す。

    Args:
    -   prefecture (str): 県名 ["kanagawa"]
    -   datehour (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"] または "now"。正時にそろえられ、分以下は無視されます。
    -   model (str): 予測モデル。
            "v0":andersan0_1(12th tile, 8 hours ahead.)
            "v0":andersan0_1(12th tile, 8 hours ahead.)
            "v0":andersan0_1(12th tile, 8 hours ahead.)

    Returns:
    -   _str_: 県内の地理院タイル点でのOxの予測値。
    """
    start_time = time.time()

    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")

    if datehour == "now":
        datehour = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
        datehour = datehour.replace(minute=0, second=0, microsecond=0)
    else:
        datehour = datehour.replace(minute=0, second=0, microsecond=0)
    try:
        isodate = datetime.datetime.isoformat(datehour)
        logger.debug(f"Using datetime: {datehour} (tzinfo: {datehour.tzinfo})")
    except Exception as e:
        logger.error(f"Error processing datetime: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing datetime: {e}")

    # prediction function switcher
    predict_ox = load_model(model)

    try:
        raw_data = predict_ox(prefecture, isodate)
        if raw_data is None:
            raise HTTPException(status_code=404, detail="Data not available")
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {e}")
    
    data = dictize(raw_data)
    process_time = time.time() - start_time
    logger.debug(f"predict_Ox internal processing time: {process_time:.3f} seconds")
    
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False))



from geopy.geocoders import Nominatim
import time

# ジオコーダーをグローバル変数として保持
geolocator = None

def get_geolocator():
    global geolocator
    if geolocator is None:
        geolocator = Nominatim(user_agent="andersan")
    return geolocator

def reverse_geocode_geopy(lon, lat):
    """
    緯度経度から住所を逆ジオコーディングする関数

    Args:
        lon (float): 経度
        lat (float): 緯度

    Returns:
        str: 住所 (取得できなかった場合はNone)
    """
    geolocator = get_geolocator()
    try:
        location = geolocator.reverse((lat, lon))
        if location:
            return location.address
        else:
            return None
    except Exception as e:
        logger.error(f"ジオコーディングエラー: {e}")
        return None


@app.get("/loc/{lon}/{lat}")
async def location(lon: float, lat: float) -> str:
    """緯度経度を住所などの情報に変換する。

    Args:
        lon (float): 経度
        lat (float): 緯度

    Returns:
        JSON str: 住所情報
            X, Y (int): 地理院タイルのX,Y
            Z (int): 地理院タイルのZoom
            address (str): 指定された地点の住所
            pref (str): 指定された地点の県名(アルファベット表記)

    """
    start_time = time.time()
    
    logger.debug(f"Geocoding location: lon={lon}, lat={lat}")
    x, y = andersan.tile.code(zoom=12, lon=lon, lat=lat)
    address = reverse_geocode_geopy(lon, lat)

    # Return the prefecture containing the tile. (Some areas like the outskirts of Kanagawa may not be included in the tile)
    prefecture = None
    for pref, ra in andersan.prefecture_ranges.items():
        if ra[0][0] <= lon < ra[1][0] and ra[0][1] <= lat < ra[1][1]:
            prefecture = pref
            break

    data = dict(X=int(x), Y=int(y), Z=12, address=address, pref=prefecture)
    
    process_time = time.time() - start_time
    logger.debug(f"location internal processing time: {process_time:.3f} seconds")
    
    return Response(content=json.dumps(data, indent=2, ensure_ascii=False))



# 次のAPI: 確率換算表を提供する。/pmap/
@app.get("/ptable/{model}")
async def probability_table(
    model: str,
):
    """NNの予測値を積分確率分布に変換する表を提供する。

    Args:
        model (str): 予測モデル。
    """
    # model switcher
    if model == "v0":
        MODEL = "andersan0_1"
    elif model == "v0a":
        MODEL = "andersan0_1_1"
    elif model == "v1":
        MODEL = "andersan0_2"
    elif model == "v1a":
        MODEL = "andersan0_2_1"
    else:
        raise InvalidModelException(model)

    table = f"/AIR/andersan-train/{MODEL}.table.feather"
    df = pd.read_feather(table)
    return Response(content=df.to_json(indent=2))


def load_model(model_name):
    if model_name == "v0":
        predict_ox = predict.predict_ox_v0
    elif model_name == "v0a":
        predict_ox = predict.predict_ox_v0a
    elif model_name == "v1":
        predict_ox = predict.predict_ox_v1
    elif model_name == "v1a":
        predict_ox = predict.predict_ox_v1a
    else:
        raise InvalidModelException(model_name)
    return predict_ox

if __name__ == "__main__":
    import os
    import signal
    
    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    basicConfig(level=DEBUG)
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
# sqlitedictのログも有効化
    sqlitedict_logger = getLogger('sqlitedict')
    sqlitedict_logger.setLevel(DEBUG)

    
    uvicorn.run(
        "andersan-api:app",
        host="0.0.0.0",
        port=8087,
        reload=True,  # リロード機能を有効化
        reload_dirs=["."],  # 現在のディレクトリのみを監視
        reload_includes=["*.py"],  # Pythonファイルのみを監視
        reload_excludes=["*.pyc", "*.pyo", "*.pyd", "__pycache__", "*.so"],  # 監視対象から除外
        reload_delay=1.0,  # 監視間隔を1秒に設定
    )
