# andersanのデータ取得関数群。APIに必要なもののみ集約する。
# 1ファイルだけならディレクトリ階層は要らないが、とりあえず残す。

import datetime
from datetime import timedelta
from logging import getLogger, basicConfig, INFO, DEBUG

import numpy as np
import pandas as pd
import json
import keras

from andersan import openmeteo, airmonitor


class InvalidForecastingRangeError(Exception):
    """予報可能な時間範囲を越える場合に発生する例外"""
    pass
    # def __init__(self, message):
    #     self.message = message
    #     super().__init__(message)

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
    stdfilename="standards.json",
):
    logger = getLogger()

    air_table = pd.DataFrame()
    timeorigin = datetime.datetime.fromisoformat(isodate)
    for delta in range(-lookback+1, 1):
        dt = timeorigin + timedelta(hours=delta)
        table = airmonitor.tiles("kanagawa", dt.isoformat(), zoom)
        air_table = pd.concat([air_table, table], axis=0)

    # if timeorigin.hour + forecast >= 24:
    #     raise InvalidForecastingRangeError
    timebegin = timeorigin + timedelta(hours=1)
    all_forecast_dataframe = openmeteo.tiles("kanagawa", datehour=timebegin.strftime("%Y-%m-%dT%H"), hours=forecast, zoom=zoom) 
    tiles = np.unique(all_forecast_dataframe[["X", "Y"]].to_numpy(), axis=0)
    print(all_forecast_dataframe)

    X0 = np.zeros([len(tiles), lookback, len(items)])
    X2 = np.zeros([len(tiles), forecast, len(noaa_cols)])
    X3 = np.zeros([len(tiles), forecast], dtype=int)
    for j, (tileX, tileY) in enumerate(tiles):
        for i, item in enumerate(items):
            X0[j, :, i] = air_table[
                (air_table.X == tileX) & (air_table.Y == tileY)
            ][item].to_numpy()
        # print(X0)

        for i, item in enumerate(noaa_cols):
            X2[j, :, i] = all_forecast_dataframe[
                (all_forecast_dataframe.X == tileX)
                & (all_forecast_dataframe.Y == tileY)
            ][item]

        X3[j, :] = all_forecast_dataframe[
            (all_forecast_dataframe.X == tileX)
            & (all_forecast_dataframe.Y == tileY)
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


class Foreseer():
    def predict_ox(self, prefecture, isodate):
        pass

class Foreseer_v0(Foreseer):
    def predict_ox(self, prefecture, isodate):
        # settings
        model = "andersan0_1"
        zoom = 12
        lookback_hours=24
        forecast_hours=8
        stdfilename="/AIR/andersan-train/datatype3/standards.json"
        
        # タイルと時刻の情報を得る
        table = airmonitor.tiles("kanagawa", isodate, zoom)

        # NNに食わせるデータの生成
        X = X_instant(prefecture, isodate, zoom, lookback=lookback_hours, forecast=forecast_hours, stdfilename=stdfilename)

        # モデルの準備
        model = keras.models.load_model(f"{model}.py.best.keras")

        # 予測
        pred = model.predict(X)

        # andersan0_1はOX値の二乗を予測するので、ここで平方根をとって戻す。
        # 二乗を予測するのは、OXが大きい時の精度を高めるため。
        pred = pred**0.5
        table = table.drop(columns=["OX", "NOX", "TEMP", "WX", "WY", "NMHC"])
        for i in range(8):
            table[f"+{i+1}"] = pred[:, i]
        return table


class Foreseer_v1(Foreseer):
    def predict_ox(self, prefecture, isodate):
        # settings
        model = "andersan0_2"
        zoom = 12
        lookback_hours=24
        forecast_hours=24
        noaa_cols=(
            "temperature_2m",
            "cloud_cover",
            "pressure_msl",
            # "shortwave_radiation",
            "wind_speed_10m",
        )
        stdfilename="/AIR/andersan-train/datatype4/standards.json"
        
        # タイルと時刻の情報を得る
        table = airmonitor.tiles("kanagawa", isodate, zoom)

        # NNに食わせるデータの生成
        X = X_instant(prefecture, isodate, zoom, lookback=lookback_hours, forecast=forecast_hours, noaa_cols=noaa_cols, stdfilename=stdfilename)

        # モデルの準備
        model = keras.models.load_model(f"{model}.py.best.keras")

        # 予測
        pred = model.predict(X)

        # andersan0_1はOX値の二乗を予測するので、ここで平方根をとって戻す。
        # 二乗を予測するのは、OXが大きい時の精度を高めるため。
        pred = pred**0.5
        table = table.drop(columns=["OX", "NOX", "TEMP", "WX", "WY", "NMHC"])
        for i in range(8):
            table[f"+{i+1}"] = pred[:, i]
        return table


def test():
    basicConfig(level=DEBUG)
    logger = getLogger()
    foreseer = Foreseer_v0()
    # logger.info(foreseer.predict_ox("kanagawa", "2025-02-20T09:00+09:00"))
    logger.info(foreseer.predict_ox("kanagawa", "2015-08-19T09:00+09:00"))


if __name__ == "__main__":
    import os
    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    test()
