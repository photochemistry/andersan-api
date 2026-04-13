# 大幅にandersan/をリファクタリングしたので、調整が必要。

import datetime
import os
import argparse
from pathlib import Path
from typing import Literal, Union
from logging import basicConfig, getLogger, INFO, DEBUG
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
DEFAULT_LOG_LEVEL_NAME = os.getenv("ANDERSAN_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_LEVEL = DEBUG if DEFAULT_LOG_LEVEL_NAME == "DEBUG" else INFO

basicConfig(
    level=DEFAULT_LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = getLogger(__name__)
logger.setLevel(DEFAULT_LOG_LEVEL)

# sqlitedictのログも有効化
sqlitedict_logger = getLogger("sqlitedict")
sqlitedict_logger.setLevel(DEFAULT_LOG_LEVEL)

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

# API 経由の airmonitor.tiles（APW）はワーカー占有を避けるため HTTP 再試行を行わない（1 試行のみ）
API_TILES_MAX_RETRIES = 1

# 確率換算表（*.table.feather）。学習側から `tables/` にコピーして運用する。
PTABLE_DIR = Path(__file__).resolve().parent / "tables"

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
    # itemsはspecのために指定するだけで、dataの中身はitemsに依存しない。
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
    raw_data = andersan.airmonitor.tiles(
        prefecture, isodate, zoom, items=ITEMS, max_retries=API_TILES_MAX_RETRIES
    )
    if raw_data is None:
        raise HTTPException(status_code=404, detail="Data not available")

    data = dictize(raw_data, items=ITEMS)

    process_time = time.time() - start_time
    logger.debug(f"tile_data internal processing time: {process_time:.3f} seconds")

    return Response(content=json.dumps(data, indent=2, ensure_ascii=False))


@app.get("/obs/{item}/{prefecture}/{datehour}")
async def observed_item(
    item: Literal[tuple(ITEMS)],
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: datetime.datetime,
):
    """県内のタイル点での実測値を返す。Zoomは12固定。

    Args:
    -   item (str): 実測値の種類 ["NMHC", "OX", "NOX", "TEMP", "WX", "WY"]
    -   prefecture (str): 県名 ["kanagawa"]
    -   datehour (str): 時刻(isoformat) ["2024-09-03T06:00+09:00"] 正時にそろえられ、分以下は無視されます。

    Returns:
    -   _str_: 実測値
    """
    start_time = time.time()

    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")

    datehour = datehour.replace(minute=0, second=0, microsecond=0)
    data = dict()
    data["data"] = dict()
    for hour in range(-23, 1):
        isodate = datetime.datetime.isoformat(datehour + datetime.timedelta(hours=hour))
        # raw_data = andersan.airmonitor.tiles(prefecture, isodate, zoom=12, items=(item,))

        # 全itemを取得する必要はない。これはデバッグのため。
        raw_data = andersan.airmonitor.tiles(
            prefecture, isodate, zoom=12, max_retries=API_TILES_MAX_RETRIES
        )
        # print(f"prefecture: {prefecture}, isodate: {isodate}")
        if raw_data is None:
            raise HTTPException(status_code=404, detail="Data not available")
        data1 = dictize(raw_data, items=(item,))
        if hour == 0:
            data["spec"] = data1["spec"]
            data["data"]["XY"] = raw_data[["X", "Y"]].to_numpy().tolist()
            data["data"]["lon"] = raw_data["lon"].tolist()
            data["data"]["lat"] = raw_data["lat"].tolist()
        data["data"][hour] = data1["data"][item]

    process_time = time.time() - start_time
    logger.debug(f"tile_data internal processing time: {process_time:.3f} seconds")

    # replace NaNs for JSON compliance.
    for hour in range(-23, 1):
        data["data"][hour] = [
            float(x) if pd.notna(x) else None for x in data["data"][hour]
        ]
    return data
    # print(data)
    # return Response(content=json.dumps(data, indent=2, ensure_ascii=False))


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
    -   model (str): 予測モデル（例: v0, v0a, v1, v1a, a1）。`a1` は andersan1（直接回帰・24h先まで）。

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
        logger.exception(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {e}")

    data = dictize(raw_data)
    data["spec"]["items"] = ["OX"]
    process_time = time.time() - start_time
    logger.debug(f"predict_Ox internal processing time: {process_time:.3f} seconds")

    return Response(content=json.dumps(data, indent=2, ensure_ascii=False))


@app.get("/a1/{prefecture}/{datehour}")
async def predict_Ox_a1_route(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
):
    """andersan1（直接数値予測）。`/ox/a1/{prefecture}/{datehour}` と同じ応答。"""
    return await predict_Ox(prefecture=prefecture, datehour=datehour, model="a1")


from geopy.geocoders import Nominatim
from functools import lru_cache
import time

# ジオコーダーをグローバル変数として保持
geolocator = None


def get_geolocator():
    global geolocator
    if geolocator is None:
        geolocator = Nominatim(user_agent="andersan")
    return geolocator


@lru_cache(maxsize=1000)
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
    # わざと精度を落す。これにより、キャッシュが効く。
    lon = round(lon, 3)
    lat = round(lat, 3)
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
_PTABLE_STEM = {
    "v0": "andersan0_1",
    "v0a": "andersan0_1_1",
    "v1": "andersan0_2",
    "v1a": "andersan0_2_1",
    "a1": "andersan1",
}


@app.get("/ptable/{model}")
async def probability_table(
    model: str,
):
    """NNの予測値を積分確率分布に変換する表を提供する。

    Args:
        model (str): 予測モデル。
    """
    if model not in _PTABLE_STEM:
        raise InvalidModelException(model)
    stem = _PTABLE_STEM[model]
    table_path = PTABLE_DIR / f"{stem}.table.feather"
    if not table_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"確率換算表がありません: {table_path.name} を {PTABLE_DIR} に置いてください。"
            ),
        )
    df = pd.read_feather(table_path)
    return Response(content=df.to_json(indent=2))


def load_model(model_name):
    if model_name == "v0":
        runner = predict.predict_ox_v0
    elif model_name == "v0a":
        runner = predict.predict_ox_v0a
    elif model_name == "v1":
        runner = predict.predict_ox_v1
    elif model_name == "v1a":
        runner = predict.predict_ox_v1a
    elif model_name == "a1":
        runner = predict.predict_ox_a1
    else:
        raise InvalidModelException(model_name)

    def predict_ox(prefecture, isodate):
        return runner(
            prefecture, isodate, tiles_max_retries=API_TILES_MAX_RETRIES
        )

    return predict_ox


if __name__ == "__main__":
    import signal

    parser = argparse.ArgumentParser(description="Run Andersan API server.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (equivalent to --log-level DEBUG).",
    )
    parser.add_argument(
        "--log-level",
        choices=["INFO", "DEBUG"],
        default=DEFAULT_LOG_LEVEL_NAME if DEFAULT_LOG_LEVEL_NAME in ("INFO", "DEBUG") else "INFO",
        help="Application log level.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8087)
    args = parser.parse_args()

    log_level_name = "DEBUG" if args.debug else args.log_level
    log_level = DEBUG if log_level_name == "DEBUG" else INFO
    # reload=True の子プロセスでも import 時設定を揃える
    os.environ["ANDERSAN_LOG_LEVEL"] = log_level_name

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    basicConfig(level=log_level)
    logger.setLevel(log_level)
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    # sqlitedictのログも有効化
    sqlitedict_logger = getLogger("sqlitedict")
    sqlitedict_logger.setLevel(log_level)

    uvicorn.run(
        "andersan-api:app",
        host=args.host,
        port=args.port,
        log_level=log_level_name.lower(),
        reload=True,  # リロード機能を有効化
        reload_dirs=["."],  # 現在のディレクトリのみを監視
        reload_includes=["*.py"],  # Pythonファイルのみを監視
        reload_excludes=["*.pyc", "*.pyo", "*.pyd", "__pycache__", "*.so"],  # 監視対象から除外
        reload_delay=1.0,  # 監視間隔を1秒に設定
    )
