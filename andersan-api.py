# 大幅にandersan/をリファクタリングしたので、調整が必要。

import asyncio
import datetime
import os
import argparse
import math
from pathlib import Path
from typing import Literal, Union
import re
from logging import basicConfig, getLogger, INFO, DEBUG, WARNING, StreamHandler
from logging.handlers import RotatingFileHandler
import pandas as pd
import uvicorn
import time
import pytz

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import andersan
import andersan.airmonitor
from andersan.sqlitedictcache import sqlitedict_cache
from andersan_core import predict
import json

# ログ設定
DEFAULT_LOG_LEVEL_NAME = os.getenv("ANDERSAN_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_LEVEL = DEBUG if DEFAULT_LOG_LEVEL_NAME == "DEBUG" else INFO
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "andersan-api.log"

# requests_cache / urllib3 は DEBUG だと1回の HTTP で数十行出し、ログI/Oだけで体感が遅くなる。
_NOISY_LOGGER_NAMES = (
    "requests_cache",
    "requests_cache.policy",
    "requests_cache.policy.actions",
    "requests_cache.backends",
    "requests_cache.backends.base",
    "requests_cache.backends.sqlite",
    "urllib3",
    "urllib3.connectionpool",
    "http.client",
)


def _silence_noisy_third_party_loggers() -> None:
    for name in _NOISY_LOGGER_NAMES:
        getLogger(name).setLevel(WARNING)
    # grid 取得の DEBUG は /obs の24ループで行数が嵩む
    getLogger("andersan.airmonitor").setLevel(INFO)


def setup_logging(log_level=DEFAULT_LOG_LEVEL):
    """コンソールとローカルファイルへログを出力する。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    stream_handler = StreamHandler()
    basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[stream_handler, file_handler],
        force=True,
    )
    _silence_noisy_third_party_loggers()


setup_logging(DEFAULT_LOG_LEVEL)
logger = getLogger(__name__)
logger.setLevel(DEFAULT_LOG_LEVEL)

# sqlitedictのログも有効化
sqlitedict_logger = getLogger("sqlitedict")
sqlitedict_logger.setLevel(DEFAULT_LOG_LEVEL)


def _lonlat_to_tile_xy_rounded(zoom: int, lon: float, lat: float) -> tuple[int, int]:
    """
    経緯度を最寄りの Web Mercator タイル (x, y) に変換する。
    境界では floor ではなく四捨五入（中心基準）でタイルを選ぶ。
    """
    n = 2**zoom
    x_f = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y_f = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n

    # Python の round() は銀行丸めなので、通常の四捨五入を使う。
    x = int(math.floor(x_f + 0.5))
    y = int(math.floor(y_f + 0.5))
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y

app = FastAPI()

# 処理時間計測用のミドルウェア
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    client = request.headers.get("x-andersan-client") or "-"
    ua = (request.headers.get("user-agent") or "")[:100]
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"
    logger.info(
        "HTTP client=%s method=%s path=%s %.3fs ua=%r",
        client,
        request.method,
        path,
        process_time,
        ua,
    )
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
DOCS_DIR = Path(__file__).resolve().parent / "docs"
UI_API_CONTRACT_MD = DOCS_DIR / "api-contract-for-ui.md"
UI_API_CONTRACT_JSON = DOCS_DIR / "api-contract-for-ui.json"

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


class PredictNoDataError(Exception):
    """予測データが無い場合（SQLite キャッシュには保存しない）。"""


class OxTilesRequest(BaseModel):
    prefecture: Literal[tuple(andersan.Neighbors)] = "kanagawa"
    tiles: list[tuple[int, int]] = Field(
        ..., description="予測したい地理院タイルの (X, Y) 配列"
    )


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


def build_meta(source_time: datetime.datetime) -> dict:
    """レスポンス共通メタデータを作る。"""
    jst = pytz.timezone("Asia/Tokyo")
    now_jst = datetime.datetime.now(jst)
    return {
        "cached_at": datetime.datetime.isoformat(now_jst),
        "source_time": datetime.datetime.isoformat(source_time),
    }


def _cdf_from_three_quantiles(
    y: float,
    v1: float,
    v2: float,
    v3: float,
    p1: float,
    p2: float,
    p3: float,
) -> float:
    """3分位点から単調な区分線形 CDF を近似する。"""
    eps = 1e-6
    pairs = sorted(zip((float(v1), float(v2), float(v3)), (p1, p2, p3)))
    (q_a, q_b, q_c), (p_a, p_b, p_c) = zip(*pairs)
    s1 = (p_b - p_a) / max(q_b - q_a, eps)
    s2 = (p_c - p_b) / max(q_c - q_b, eps)

    if y <= q_a:
        cdf = p_a - s1 * (q_a - y)
    elif y <= q_b:
        cdf = p_a + s1 * (y - q_a)
    elif y <= q_c:
        cdf = p_b + s2 * (y - q_b)
    else:
        cdf = p_c + s2 * (y - q_c)
    return max(0.0, min(1.0, cdf))


def _cdf_from_q10_q50_q90(y: float, q10: float, q50: float, q90: float) -> float:
    return _cdf_from_three_quantiles(y, q10, q50, q90, 0.1, 0.5, 0.9)


def _cdf_from_q50_q90_q95(y: float, q50: float, q90: float, q95: float) -> float:
    return _cdf_from_three_quantiles(y, q50, q90, q95, 0.5, 0.9, 0.95)


def _exceedance_prob_from_quantiles(
    low_series: pd.Series,
    mid_series: pd.Series,
    high_series: pd.Series,
    threshold: float,
    cdf_fn,
) -> list[float]:
    probs = []
    for low, mid, high in zip(low_series, mid_series, high_series):
        if pd.isna(low) or pd.isna(mid) or pd.isna(high):
            probs.append(None)
            continue
        cdf = cdf_fn(float(threshold), low, mid, high)
        probs.append(1.0 - cdf)
    return probs


def _exceedance_prob_from_q10_q50_q90(
    q10_series: pd.Series, q50_series: pd.Series, q90_series: pd.Series, threshold: float
) -> list[float]:
    return _exceedance_prob_from_quantiles(
        q10_series, q50_series, q90_series, threshold, _cdf_from_q10_q50_q90
    )


def _exceedance_prob_from_q50_q90_q95(
    q50_series: pd.Series, q90_series: pd.Series, q95_series: pd.Series, threshold: float
) -> list[float]:
    return _exceedance_prob_from_quantiles(
        q50_series, q90_series, q95_series, threshold, _cdf_from_q50_q90_q95
    )


def _load_probability_table(model: str) -> pd.DataFrame:
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
    table = pd.read_feather(table_path)
    # 生成元によっては MultiIndex(bin, hours) が index 側に残るため、
    # 列参照で一貫して扱えるように正規化する。
    if "bin" not in table.columns or "hours" not in table.columns:
        table = table.reset_index()
    if "bin" not in table.columns or "hours" not in table.columns:
        raise HTTPException(
            status_code=500,
            detail=(
                f"確率換算表の形式が不正です: {table_path.name} に 'bin'/'hours' がありません。"
            ),
        )
    return table


def _lookup_exceedance_prob_from_table(
    table: pd.DataFrame, pred_value: float, hour: int, threshold: int = 120
) -> float | None:
    if pd.isna(pred_value):
        return None
    bin_value = int(float(pred_value) // 5 * 5)
    bin_value = max(0, min(145, bin_value))
    row = table[(table["hours"] == hour) & (table["bin"] == bin_value)]
    if row.empty:
        return None

    col = str(threshold) if str(threshold) in row.columns else threshold
    if col not in row.columns:
        return None
    value = row.iloc[0][col]
    return float(value) if pd.notna(value) else None


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
    data["meta"] = build_meta(datehour)

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
    data["meta"] = build_meta(datehour)
    data["data"] = dict()
    for hour in range(-23, 1):
        isodate = datetime.datetime.isoformat(datehour + datetime.timedelta(hours=hour))
        raw_data = andersan.airmonitor.tiles(
            prefecture,
            isodate,
            zoom=12,
            items=(item,),
            max_retries=API_TILES_MAX_RETRIES,
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
    x, y = _lonlat_to_tile_xy_rounded(zoom=12, lon=lon, lat=lat)
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
    "a1p": "andersan1_16",
    "a1q": "andersan1_17",
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


@app.get("/contract/ui")
async def ui_api_contract():
    """UI/AI向け固定API仕様を返す（JSON）。"""
    if not UI_API_CONTRACT_JSON.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"仕様ファイルがありません: {UI_API_CONTRACT_JSON.name} を {DOCS_DIR} に置いてください。"
            ),
        )
    return Response(
        content=UI_API_CONTRACT_JSON.read_text(encoding="utf-8"),
        media_type="application/json",
    )


@app.get("/contract/ui.md")
async def ui_api_contract_markdown():
    """UI/AI向け固定API仕様を返す（Markdown）。"""
    if not UI_API_CONTRACT_MD.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"仕様ファイルがありません: {UI_API_CONTRACT_MD.name} を {DOCS_DIR} に置いてください。"
            ),
        )
    return Response(
        content=UI_API_CONTRACT_MD.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
    )


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
    elif model_name == "a1p":
        runner = predict.predict_ox_a1p
    elif model_name == "a1q":
        runner = predict.predict_ox_a1q
    else:
        raise InvalidModelException(model_name)

    def predict_ox(prefecture, isodate):
        return runner(
            prefecture, isodate, tiles_max_retries=API_TILES_MAX_RETRIES
        )

    return predict_ox


def load_model_by_tiles(model_name):
    if model_name == "v0":
        runner = predict.predict_ox_v0_by_tiles
    elif model_name == "v0a":
        runner = predict.predict_ox_v0a_by_tiles
    elif model_name == "v1":
        runner = predict.predict_ox_v1_by_tiles
    elif model_name == "v1a":
        runner = predict.predict_ox_v1a_by_tiles
    elif model_name == "a1":
        runner = predict.predict_ox_a1_by_tiles
    elif model_name == "a1p":
        runner = predict.predict_ox_a1p_by_tiles
    elif model_name == "a1q":
        runner = predict.predict_ox_a1q_by_tiles
    else:
        raise InvalidModelException(model_name)

    def predict_ox(prefecture, isodate, tiles):
        return runner(
            prefecture, isodate, tiles, tiles_max_retries=API_TILES_MAX_RETRIES
        )

    return predict_ox


def _predicted_tiles_list(df: pd.DataFrame) -> list[list[int]]:
    if "X" not in df.columns or "Y" not in df.columns:
        return []
    return [[int(x), int(y)] for x, y in df[["X", "Y"]].to_numpy().tolist()]


@sqlitedict_cache("api_predict_ox")
def _predict_ox_payload_sync(
    model: str,
    prefecture: str,
    isodate: str,
    tiles: tuple[tuple[int, int], ...] | None = None,
) -> dict:
    """予測結果の dict（meta 含む）を組み立てる。SQLite キャッシュの単位。"""
    if tiles:
        predict_ox = load_model_by_tiles(model)
        raw_data = predict_ox(prefecture, isodate, tiles)
    else:
        predict_ox = load_model(model)
        raw_data = predict_ox(prefecture, isodate)
    if raw_data is None:
        raise PredictNoDataError()
    source_dt = datetime.datetime.fromisoformat(isodate)
    data = dictize(raw_data)
    data["spec"]["items"] = ["OX"]
    data["meta"] = build_meta(source_dt)
    data["meta"]["predicted_tiles"] = _predicted_tiles_list(raw_data)
    return data


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
    -   model (str): 予測モデル（例: v0, v0a, v1, v1a, a1, a1p, a1q）。`a1` は andersan1（直接回帰・24h先まで）、`a1p` は andersan1_16、`a1q` は andersan1_17（いずれも a1 と同じ入力構造の改良版）。

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

    try:
        payload = await asyncio.to_thread(
            _predict_ox_payload_sync, model, prefecture, isodate
        )
    except PredictNoDataError:
        raise HTTPException(status_code=404, detail="Data not available")
    except InvalidModelException as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.exception(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {e}")

    process_time = time.time() - start_time
    logger.debug(f"predict_Ox internal processing time: {process_time:.3f} seconds")

    return Response(content=json.dumps(payload, indent=2, ensure_ascii=False))


@app.get("/a1/{prefecture}/{datehour}")
async def predict_Ox_a1_route(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
):
    """andersan1（直接数値予測）。`/ox/a1/{prefecture}/{datehour}` と同じ応答。"""
    return await predict_Ox(prefecture=prefecture, datehour=datehour, model="a1")


@app.get("/a1p/{prefecture}/{datehour}")
async def predict_Ox_a1p_route(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
):
    """andersan1_16（a1 と同じ入力構造の改良版）。`/ox/a1p/{prefecture}/{datehour}` と同じ応答。"""
    return await predict_Ox(prefecture=prefecture, datehour=datehour, model="a1p")


@app.get("/a1q/{prefecture}/{datehour}")
async def predict_Ox_a1q_route(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
):
    """andersan1_17（a1 と同じ入力構造の改良版）。`/ox/a1q/{prefecture}/{datehour}` と同じ応答。"""
    return await predict_Ox(prefecture=prefecture, datehour=datehour, model="a1q")


@app.post("/ox/{model}/{datehour}")
async def predict_Ox_by_tiles(
    datehour: Union[datetime.datetime, Literal["now"]],
    model: str,
    req: OxTilesRequest,
):
    """タイル集合を指定して OX 予測を返す。"""
    if datehour == "now":
        datehour = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
        datehour = datehour.replace(minute=0, second=0, microsecond=0)
    else:
        datehour = datehour.replace(minute=0, second=0, microsecond=0)
    isodate = datetime.datetime.isoformat(datehour)
    try:
        normalized_tiles = tuple((int(x), int(y)) for x, y in req.tiles)
        payload = await asyncio.to_thread(
            _predict_ox_payload_sync,
            model,
            req.prefecture,
            isodate,
            normalized_tiles,
        )
    except PredictNoDataError:
        raise HTTPException(status_code=404, detail="Data not available")
    except InvalidModelException as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.exception(f"Error in tile-set prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {e}")
    return Response(content=json.dumps(payload, indent=2, ensure_ascii=False))


def _quantile_ox_items(
    *, suffixes: tuple[str, str, str] = ("q10", "q50", "q90")
) -> list[str]:
    q_lo, q_mid, q_hi = suffixes
    return (
        [f"+{h}_{q_lo}" for h in range(1, 25)]
        + [f"+{h}_{q_mid}" for h in range(1, 25)]
        + [f"+{h}_{q_hi}" for h in range(1, 25)]
    )


@sqlitedict_cache("api_predict_oxq_a4_1")
def _predict_oxq_a4_1_payload_sync(prefecture: str, isodate: str) -> dict:
    """andersan4_1（分位点回帰）の予測結果を返す。"""
    raw_data = predict.predict_oxq_a4_1(
        prefecture, isodate, tiles_max_retries=API_TILES_MAX_RETRIES
    )
    if raw_data is None:
        raise PredictNoDataError()
    source_dt = datetime.datetime.fromisoformat(isodate)
    data = dictize(raw_data)
    data["spec"]["items"] = _quantile_ox_items()
    data["meta"] = build_meta(source_dt)
    return data


@sqlitedict_cache("api_predict_oxq_a3_16_q509095")
def _predict_oxq_a3_16_payload_sync(prefecture: str, isodate: str) -> dict:
    """andersan3_16（分位点回帰）の予測結果を返す。"""
    raw_data = predict.predict_oxq_a3_16(
        prefecture, isodate, tiles_max_retries=API_TILES_MAX_RETRIES
    )
    if raw_data is None:
        raise PredictNoDataError()
    source_dt = datetime.datetime.fromisoformat(isodate)
    data = dictize(raw_data)
    data["spec"]["items"] = _quantile_ox_items(suffixes=("q50", "q90", "q95"))
    data["meta"] = build_meta(source_dt)
    return data


@app.get("/oxq/a4_1/{prefecture}/{datehour}")
async def predict_Ox_quantile_a4_1(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
):
    """andersan4_1（分位点回帰）を返す。+1..+24 時間先の q10/q50/q90 を含む。"""
    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")

    if datehour == "now":
        datehour = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
    datehour = datehour.replace(minute=0, second=0, microsecond=0)
    isodate = datetime.datetime.isoformat(datehour)
    try:
        payload = await asyncio.to_thread(
            _predict_oxq_a4_1_payload_sync, prefecture, isodate
        )
    except PredictNoDataError:
        raise HTTPException(status_code=404, detail="Data not available")
    except Exception as e:
        logger.exception(f"Error in quantile prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in quantile prediction: {e}")

    return Response(content=json.dumps(payload, indent=2, ensure_ascii=False))


@app.get("/oxq/a3_16/{prefecture}/{datehour}")
async def predict_Ox_quantile_a3_16(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
):
    """andersan3_16（分位点回帰）を返す。+1..+24 時間先の q50/q90/q95 を含む。"""
    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")

    if datehour == "now":
        datehour = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
    datehour = datehour.replace(minute=0, second=0, microsecond=0)
    isodate = datetime.datetime.isoformat(datehour)
    try:
        payload = await asyncio.to_thread(
            _predict_oxq_a3_16_payload_sync, prefecture, isodate
        )
    except PredictNoDataError:
        raise HTTPException(status_code=404, detail="Data not available")
    except Exception as e:
        logger.exception(f"Error in quantile prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in quantile prediction: {e}")

    return Response(content=json.dumps(payload, indent=2, ensure_ascii=False))


@sqlitedict_cache("api_predict_oxq_a4_1_pgt120")
def _predict_oxq_a4_1_pgt120_payload_sync(prefecture: str, isodate: str) -> dict:
    """andersan4_1 の q10/q50/q90 から 120ppb 超過確率を返す。"""
    raw_data = predict.predict_oxq_a4_1(
        prefecture, isodate, tiles_max_retries=API_TILES_MAX_RETRIES
    )
    if raw_data is None:
        raise PredictNoDataError()

    result = raw_data[["X", "Y", "lon", "lat", "Z"]].copy()
    threshold = 120.0
    for h in range(1, 25):
        p = _exceedance_prob_from_q10_q50_q90(
            raw_data[f"+{h}_q10"],
            raw_data[f"+{h}_q50"],
            raw_data[f"+{h}_q90"],
            threshold=threshold,
        )
        result[f"+{h}_p_gt_120"] = p

    source_dt = datetime.datetime.fromisoformat(isodate)
    data = dictize(result)
    data["spec"]["items"] = [f"+{h}_p_gt_120" for h in range(1, 25)]
    data["meta"] = build_meta(source_dt)
    return data


@sqlitedict_cache("api_predict_ox_pgt120")
def _predict_ox_pgt120_payload_sync(model: str, prefecture: str, isodate: str) -> dict:
    """通常回帰モデルの予測値を、確率換算表で P(OX>120) に変換する。"""
    predict_ox = load_model(model)
    raw_data = predict_ox(prefecture, isodate)
    if raw_data is None:
        raise PredictNoDataError()

    table = _load_probability_table(model)
    result = raw_data[["X", "Y", "lon", "lat", "Z"]].copy()
    hour_cols = []
    for col in raw_data.columns:
        m = re.fullmatch(r"\+(\d+)", str(col))
        if m:
            hour_cols.append((int(m.group(1)), col))
    hour_cols.sort(key=lambda x: x[0])

    for h, pred_col in hour_cols:
        probs = [
            _lookup_exceedance_prob_from_table(table, pred_value, hour=h, threshold=120)
            for pred_value in raw_data[pred_col]
        ]
        result[f"+{h}_p_gt_120"] = probs

    source_dt = datetime.datetime.fromisoformat(isodate)
    data = dictize(result)
    data["spec"]["items"] = [f"+{h}_p_gt_120" for h, _ in hour_cols]
    data["meta"] = build_meta(source_dt)
    return data


@app.get("/oxq/a4_1/pgt120/{prefecture}/{datehour}")
async def predict_Ox_quantile_a4_1_pgt120(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
):
    """andersan4_1 の予測から OX が 120ppb を超える確率を返す。"""
    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")

    if datehour == "now":
        datehour = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
    datehour = datehour.replace(minute=0, second=0, microsecond=0)
    isodate = datetime.datetime.isoformat(datehour)
    try:
        payload = await asyncio.to_thread(
            _predict_oxq_a4_1_pgt120_payload_sync, prefecture, isodate
        )
    except PredictNoDataError:
        raise HTTPException(status_code=404, detail="Data not available")
    except Exception as e:
        logger.exception(f"Error in exceedance prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in exceedance prediction: {e}")

    return Response(content=json.dumps(payload, indent=2, ensure_ascii=False))


@sqlitedict_cache("api_predict_oxq_a3_16_pgt120_q509095")
def _predict_oxq_a3_16_pgt120_payload_sync(prefecture: str, isodate: str) -> dict:
    """andersan3_16 の q50/q90/q95 から 120ppb 超過確率を返す。"""
    raw_data = predict.predict_oxq_a3_16(
        prefecture, isodate, tiles_max_retries=API_TILES_MAX_RETRIES
    )
    if raw_data is None:
        raise PredictNoDataError()

    result = raw_data[["X", "Y", "lon", "lat", "Z"]].copy()
    threshold = 120.0
    for h in range(1, 25):
        p = _exceedance_prob_from_q50_q90_q95(
            raw_data[f"+{h}_q50"],
            raw_data[f"+{h}_q90"],
            raw_data[f"+{h}_q95"],
            threshold=threshold,
        )
        result[f"+{h}_p_gt_120"] = p

    source_dt = datetime.datetime.fromisoformat(isodate)
    data = dictize(result)
    data["spec"]["items"] = [f"+{h}_p_gt_120" for h in range(1, 25)]
    data["meta"] = build_meta(source_dt)
    return data


@app.get("/oxq/a3_16/pgt120/{prefecture}/{datehour}")
async def predict_Ox_quantile_a3_16_pgt120(
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
):
    """andersan3_16 の予測から OX が 120ppb を超える確率を返す。"""
    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")

    if datehour == "now":
        datehour = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
    datehour = datehour.replace(minute=0, second=0, microsecond=0)
    isodate = datetime.datetime.isoformat(datehour)
    try:
        payload = await asyncio.to_thread(
            _predict_oxq_a3_16_pgt120_payload_sync, prefecture, isodate
        )
    except PredictNoDataError:
        raise HTTPException(status_code=404, detail="Data not available")
    except Exception as e:
        logger.exception(f"Error in exceedance prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in exceedance prediction: {e}")

    return Response(content=json.dumps(payload, indent=2, ensure_ascii=False))


@app.get("/ox/{model}/pgt120/{prefecture}/{datehour}")
async def predict_Ox_pgt120(
    model: str,
    prefecture: Literal[tuple(andersan.Neighbors)],
    datehour: Union[datetime.datetime, Literal["now"]],
):
    """通常回帰モデル（v0/v0a/v1/v1a/a1/a1p/a1q）の 120ppb 超過確率を返す。"""
    if prefecture not in andersan.Neighbors:
        raise HTTPException(status_code=404, detail="Out of the cover area")
    if model not in _PTABLE_STEM:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not available.")

    if datehour == "now":
        datehour = datetime.datetime.now(pytz.timezone("Asia/Tokyo"))
    datehour = datehour.replace(minute=0, second=0, microsecond=0)
    isodate = datetime.datetime.isoformat(datehour)
    try:
        payload = await asyncio.to_thread(
            _predict_ox_pgt120_payload_sync, model, prefecture, isodate
        )
    except PredictNoDataError:
        raise HTTPException(status_code=404, detail="Data not available")
    except InvalidModelException as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.exception(f"Error in exceedance prediction from table: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error in exceedance prediction from table: {e}"
        )

    return Response(content=json.dumps(payload, indent=2, ensure_ascii=False))


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

    setup_logging(log_level)
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
