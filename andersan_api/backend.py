# andersanのデータ取得関数群。APIに必要なもののみ集約する。
# 1ファイルだけならディレクトリ階層は要らないが、とりあえず残す。

from lru import cache, shelf_cache, sqlitedict_cache
from airpollutionwatch import kanagawa, shizuoka, tokyo, chiba, yamanashi, amedas
import pandas as pd
import numpy as np
from andersan import tile
from airpollutionwatch.convert import stations as fullstations
from delaunayextrapolation import DelaunayE
import os
from logging import getLogger, basicConfig, INFO, DEBUG
import requests_cache
from retry_requests import retry
import datetime
import json
from datetime import timedelta
import keras

# 県ごとの大気監視ウェブサイトからデータをもってくる関数の名前
prefecture_retrievers = dict(
    kanagawa=kanagawa, shizuoka=shizuoka, tokyo=tokyo, chiba=chiba, yamanashi=yamanashi
)

# ある県のグリッドを構成するために必要な近隣県の名前
Neighbors = dict(
    kanagawa={"kanagawa", "shizuoka", "tokyo", "chiba", "yamanashi"},
)

# ある県のグリッドの範囲。細かさはzoomであとで指定する。
prefecture_ranges = dict(kanagawa=np.array([[138.94, 35.13], [139.84, 35.66]]))


def station2lonlat(stations: list):
    lonlats = dict()
    for station in stations:
        if station in fullstations.index:
            lonlats[station] = (
                fullstations.loc[station, "経度"],
                fullstations.loc[station, "緯度"],
            )
    return lonlats


def wdws2wxwy(wdws):
    """風向きをベクトルになおす。通例にあわせ、ベクトルは風が来る方向を向いていることに注意。"""
    direc, speed = wdws[:, 0], wdws[:, 1]
    theta = direc * np.pi / 8  # verified with convert_wind.py
    notnan = np.logical_not(np.isnan(speed))
    x = np.full_like(direc, np.nan)  # fill with nan
    y = np.full_like(direc, np.nan)  # fill with nan
    x[notnan] = speed[notnan] * np.sin(theta[notnan])
    y[notnan] = speed[notnan] * np.cos(theta[notnan])
    return x, y


# @lru_cache(maxsize=9999)
# @shelf_cache("observes")
@sqlitedict_cache("observes")  # vscodeで中身をチェックできる分、こちらのほうが便利
def observes_(
    target_prefecture: str,
    isodate: str,
    zoom: int,
    use_amedas=False,
    items=["NMHC", "OX", "NOX", "TEMP", "WX", "WY"],  # order in datatype3
):
    """各県の特定時刻の大気監視データを入手し、地理院メッシュ点での測定値を内挿する。"""

    if target_prefecture not in Neighbors:
        return None # 神奈川以外はまだ動かない

    # 地理院メッシュの間隔
    pref_range = np.array(prefecture_ranges[target_prefecture])  # lon,lat
    tiles, shape = tile.tiles(zoom, pref_range)

    # 測定値をとってくる。
    # 2回目からのアクセスはairpollution.sqliteに保存された内容を利用する
    try:
        dfs = [
            prefecture_retrievers[pref].retrieve(isodate)
            for pref in Neighbors[target_prefecture]
        ]
    except:
        return None

    if use_amedas:
        # 気温はAMeDASから入手 (ゆくゆくは風速も)
        amedas_df = amedas.retrieve(isodate)
        amedas_df = amedas_df.replace({pd.NA: None})
        amedas_df["WX"], amedas_df["WY"] = wdws2wxwy(
            amedas_df[["WD", "WS"]].to_numpy().astype(float)
        )

    # 全県の測定値を連結。欠測はNaNとする。
    full = pd.concat(dfs, join="outer")
    full = full.replace({pd.NA: None})
    full["WX"], full["WY"] = wdws2wxwy(full[["WD", "WS"]].to_numpy().astype(float))

    # 神奈川県の範囲
    # 範囲の指定方法を変更
    lonlats = tile.lonlat(zoom=zoom, xy=tiles)

    # 測定値の表。columnsは測定値名
    table = pd.DataFrame()
    table["lon"] = lonlats[:, 0]
    table["lat"] = lonlats[:, 1]
    table["X"] = tiles[:, 0]
    table["Y"] = tiles[:, 1]
    table["Z"] = zoom
    dt = datetime.datetime.fromisoformat(isodate)
    table["timestamp"] = dt
    table = table.set_index("timestamp")
    for item in items:

        # 欠測の測定局は除外する
        series = full[item].dropna()

        # itemとlonlatだけのdfを作る。
        # 各測定局の経度緯度
        item_df = pd.DataFrame.from_dict(station2lonlat(series.index), orient="index")
        item_df[item] = series
        item_df.columns = ["lon", "lat", item]

        # 副作用をさける
        del series

        # Mix amedas
        if use_amedas:
            if item in ("TEMP", "WX", "WY"):
                series2 = amedas_df[["lon", "lat", item]].dropna()
                item_df = pd.concat([item_df, series2])
                # print(item_df.tail())

        # 測定局でDelaunay三角形を作り、gridsの格子点の内挿比を求める
        tri = DelaunayE(item_df[["lon", "lat"]])

        values = []

        for lonlat in lonlats:
            v, mix = tri.mixratio(lonlat)
            if np.all(mix > 0):
                values.append(mix @ item_df.iloc[v][item])
            else:
                # 外挿はしない
                values.append(np.nan)

        table[item] = np.array(values)

    # table.index = table.index.astype(int)

    return table


def observes(
    target_prefecture: str,
    isodate: str,
    zoom: int,
    use_amedas=True,
    items=["NMHC", "OX", "NOX", "TEMP", "WX", "WY"],  # order in datatype3
):    # ここで、isodateに時刻が含まれる場合に日付と時だけに修正する。
    dt = datetime.datetime.fromisoformat(isodate)
    datestr = dt.strftime("%Y-%m-%dT%H:00:00+09:00")
    return observes_(target_prefecture, datestr, zoom, use_amedas=use_amedas, items=items)



OPENMETEO_ITEMS = [
    "temperature_2m",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "pressure_msl",
    "shortwave_radiation",
]


# @lru_cache
# @shelf_cache("openmeteo")
@sqlitedict_cache("openmeteo")  # vscodeで中身をチェックできる分、こちらのほうが便利
def openmeteo_tiles_(target_prefecture: str, datestr: str, zoom):
    logger = getLogger()

    if target_prefecture not in Neighbors:  # 神奈川以外はまだ動かない
        return None

    # 地理院メッシュの間隔
    pref_range = np.array(prefecture_ranges[target_prefecture])  # lon,lat
    tiles, shape = tile.tiles(zoom, pref_range)

    lonlats = tile.lonlat(xy=tiles, zoom=zoom)
    # Setup the cache and retry mechanism
    cache_session = requests_cache.CachedSession("airpollution")
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

    # dt = datetime.datetime.fromisoformat(isodate)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": ",".join([f"{x:.4f}" for x in lonlats[:, 1]]),
        "longitude": ",".join([f"{x:.4f}" for x in lonlats[:, 0]]),
        "hourly": ",".join(OPENMETEO_ITEMS),
        "start_date": datestr,
        "end_date": datestr,
        "timezone": "Asia/Tokyo",
    }
    logger.debug(params)
    response = retry_session.get(url, params=params)
    logger.info(response)
    logger.debug(f"Cached: {response.from_cache}")
    data = response.json()

    # データを格納するリスト
    all_forecast_data = []

    # 測定量、時刻、緯度経度の3次元の配列でいいか。
    # dataの構造に沿って、まず緯度経度、物質量x時間で作表する。
    for elem, (x, y) in zip(data, tiles):
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(elem["hourly"]["time"][0]),
                periods=len(elem["hourly"]["time"]),
                freq="H",
                tz="Asia/Tokyo",
            ),
            "X": x,
            "Y": y,
            "Z": zoom,
        } | {item: elem["hourly"][item] for item in OPENMETEO_ITEMS}
        hourly_dataframe = pd.DataFrame(data=hourly_data)
        all_forecast_data.append(hourly_dataframe)

    all_forecast_dataframe = pd.concat(all_forecast_data, ignore_index=True)

    return all_forecast_dataframe


def openmeteo_tiles(target_prefecture: str, isodate: str, zoom):
    # ここで、isodateに時刻が含まれる場合に日付けだけに修正する。
    # そうしないと、キャッシュに同じデータが24個も保管されてしまう。
    dt = datetime.datetime.fromisoformat(isodate)
    datestr = dt.strftime("%Y-%m-%d")
    return openmeteo_tiles_(target_prefecture, datestr, zoom)


def X_instant(
    pref_name: str,
    isodate: str,
    zoom: int,
    lookback=24,
    forecast=8,
    items=("NMHC", "OX", "NOX", "TEMP", "WX", "WY"),
    noaa_cols=(
        "temperature_2m",
        "cloud_cover",
        "pressure_msl",
        "shortwave_radiation",
        "wind_speed_10m",
    ),
    stdfilename = "standards.json"

):
    logger = getLogger()

    observes_table = pd.DataFrame()
    timeorigin = datetime.datetime.fromisoformat(isodate)
    for delta in range(-23, 1):
        dt = timeorigin + timedelta(hours=delta)
        table = observes("kanagawa", dt.isoformat(), zoom)
        observes_table = pd.concat([observes_table, table], axis=0)

    assert timeorigin.hour < 16, "予測値が翌日にまたがるケースはまだ対応していません。"
    all_forecast_dataframe = openmeteo_tiles("kanagawa", isodate, zoom)
    timebegin = timeorigin + timedelta(hours=1)
    timeend = timeorigin + timedelta(hours=8)
    tiles = np.unique(all_forecast_dataframe[["X", "Y"]].to_numpy(), axis=0)
    # print(tiles)

    X0 = np.zeros([len(tiles), 24, 6])
    X2 = np.zeros([len(tiles), 8, 5])
    X3 = np.zeros([len(tiles), 8], dtype=int)
    for j, (tileX, tileY) in enumerate(tiles):
        for i, item in enumerate(items):
            X0[j, :, i] = observes_table[
                (observes_table.X == tileX) & (observes_table.Y == tileY)
            ][item].to_numpy()
        # print(X0)

        for i, item in enumerate(noaa_cols):
            X2[j, :, i] = all_forecast_dataframe[
                (all_forecast_dataframe.X == tileX)
                & (all_forecast_dataframe.Y == tileY)
                & (all_forecast_dataframe.date >= timebegin)
                & (all_forecast_dataframe.date <= timeend)
            ][item]

        X3[j, :] = all_forecast_dataframe[
            (all_forecast_dataframe.X == tileX)
            & (all_forecast_dataframe.Y == tileY)
            & (all_forecast_dataframe.date >= timebegin)
            & (all_forecast_dataframe.date <= timeend)
        ]["weather_code"]

    X = {
        "Input_lookbacks": X0,
        "Input_forecasts": X2,
        "Input_weathercodes": X3,
    }

    logger.info(f"Standardization with {stdfilename}")

    with open(stdfilename) as f:
        specs = json.load(f)

    for label in specs:
        for icol in range(X[label].shape[-1]):
            average = specs[label][icol]["average"]
            std = specs[label][icol]["std"]
            X[label][:, :, icol] = (X[label][:, :, icol] - average) / std

    return X

def predict_ox(prefecture, isodate, zoom, basename = "andersan0_1.py"):
    # タイルと時刻の情報を得る
    table = observes("kanagawa", isodate, zoom)

    # NNに食わせるデータの生成
    X = X_instant(prefecture, isodate, zoom)

    # モデルの準備
    model = keras.models.load_model(f"{basename}.best.keras")

    # 予測
    pred = model.predict(X)

    # andersan0_1はOX値の二乗を予測するので、ここで平方根をとって戻す。
    # 二乗を予測するのは、OXが大きい時の精度を高めるため。
    pred = pred**0.5
    table = table.drop(columns=["OX", "NOX", "TEMP", "WX", "WY", "NMHC"])
    for i in range(8):
        table[f"+{i+1}"] = pred[:,i]
    return table