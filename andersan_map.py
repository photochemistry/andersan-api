# 地図関連の関数

import matplotlib.pyplot as plt
import folium
import numpy as np
from andersan_backend import prefecture_ranges


def colorize(array, cmap="viridis"):
    # 値を色に変換
    normed_data = (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
    cm = plt.cm.get_cmap(cmap)
    return cm(normed_data)


def map_layer(pref_name, values, name):
    pref_range = prefecture_ranges[pref_name]

    # 色に変換する。NaNは透明になるらしい
    image = colorize(values)

    return folium.raster_layers.ImageOverlay(
        image=image,
        bounds=[
            [np.min(pref_range[:, 1]), np.min(pref_range[:, 0])],  # lat,lon
            [np.max(pref_range[:, 1]), np.max(pref_range[:, 0])],  # lat,lon
        ],
        opacity=0.5,
        name=name,
    )
