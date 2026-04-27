# andersan-api

大気環境予測システムのAPIサーバー

## UI/AI向け固定仕様

- Cursor AI が `andersan-ui` 側から参照しやすい固定仕様:
  - `docs/api-contract-for-ui.md`
  - `docs/api-contract-for-ui.json`
- API経由でも取得可能:
  - `/contract/ui`（JSON）
  - `/contract/ui.md`（Markdown）

## エンドポイント

### 1. 実測値データ

#### 1.1 測定局データ
- エンドポイント: `/raw/{prefecture}/{datehour}`
- 説明: 指定された県の全測定局の実測値を返す
- パラメータ:
  - `prefecture`: 県名（例: "kanagawa"）
  - `datehour`: 時刻（ISOフォーマット、例: "2024-09-03T06:00+09:00"）
- 返却値: 県提供の大気監視データ（JSON形式）

#### 1.2 タイルデータ
- エンドポイント: `/tile/{zoom}/{prefecture}/{datehour}`
- 説明: 指定された県の地理院タイル点での実測値を返す
- パラメータ:
  - `zoom`: 地理院タイルのズーム値
  - `prefecture`: 県名（例: "kanagawa"）
  - `datehour`: 時刻（ISOフォーマット、例: "2024-09-03T06:00+09:00"）
- 返却値: タイル点での実測値（JSON形式）

### 2. 予測データ

#### 2.1 OX予測値
- エンドポイント: `/ox/{model}/{prefecture}/{datehour}`
- 説明: 指定された県のタイル点でのOX予測値を返す
- パラメータ:
  - `model`: 予測モデル（"v0", "v0a", "v1", "v1a", "a1"）
  - `prefecture`: 県名（例: "kanagawa"）
  - `datehour`: 時刻（ISOフォーマット、例: "2024-09-03T06:00+09:00"）
- 返却値: タイル点でのOX予測値（JSON形式）

#### 2.2 確率分布表
- エンドポイント: `/ptable/{model}`
- 説明: ニューラルネットワークの予測値を積分確率分布に変換する表を提供する
- パラメータ:
  - `model`: 予測モデル（"v0", "v0a", "v1", "v1a", "a1"）
- 返却値: 確率分布表（JSON形式）

#### 2.3 OX分位点予測値（andersan4_1）
- エンドポイント: `/oxq/a4_1/{prefecture}/{datehour}`
- 説明: 指定された県のタイル点で、1〜24時間先の OX 分位点予測（q10/q50/q90）を返す
- パラメータ:
  - `prefecture`: 県名（例: "kanagawa"）
  - `datehour`: 時刻（ISOフォーマット、例: "2024-09-03T06:00+09:00" または "now"）
- 返却値: タイル点での分位点予測（JSON形式）

#### 2.4 OX120ppb超過確率（andersan4_1）
- エンドポイント: `/oxq/a4_1/pgt120/{prefecture}/{datehour}`
- 説明: `q10/q50/q90` から単調CDF補間（区分線形）で推定した、1〜24時間先の `P(OX > 120ppb)` を返す
- パラメータ:
  - `prefecture`: 県名（例: "kanagawa"）
  - `datehour`: 時刻（ISOフォーマット、例: "2024-09-03T06:00+09:00" または "now"）
- 返却値: タイル点ごとの超過確率（JSON形式）

#### 2.5 OX120ppb超過確率（通常回帰モデル）
- エンドポイント: `/ox/{model}/pgt120/{prefecture}/{datehour}`
- 説明: `v0/v0a/v1/v1a/a1` の予測値を、対応する確率換算表（`tables/*.table.feather`）で `P(OX > 120ppb)` に変換して返す
- パラメータ:
  - `model`: 予測モデル（"v0", "v0a", "v1", "v1a", "a1"）
  - `prefecture`: 県名（例: "kanagawa"）
  - `datehour`: 時刻（ISOフォーマット、例: "2024-09-03T06:00+09:00" または "now"）
- 返却値: タイル点ごとの超過確率（JSON形式）

### 3. 位置情報

#### 3.1 位置情報変換
- エンドポイント: `/loc/{lon}/{lat}`
- 説明: 緯度経度を住所などの情報に変換する
- パラメータ:
  - `lon`: 経度
  - `lat`: 緯度
- 返却値: 位置情報（JSON形式）
  - `X`, `Y`: 地理院タイルのX,Y座標
  - `Z`: 地理院タイルのズームレベル
  - `address`: 指定地点の住所
  - `pref`: 指定地点の県名（アルファベット表記）

## データ仕様

### 測定項目
- `NMHC`: 非メタン炭化水素（単位: 10ppbC）
- `OX`: 酸化剤（単位: ppb）
- `NOX`: 窒素酸化物（単位: ppb）
- `TEMP`: 温度（単位: 0.1℃）
- `WX`: 風速X成分（単位: 0.1m/s）
- `WY`: 風速Y成分（単位: 0.1m/s）

## 注意事項
- 時刻は正時にそろえられ、分以下は無視されます
- `/ox` の重複計算抑制（キャッシュ・計算中の合流）はプロセス単位にのみ効きます。`uvicorn --workers` や gunicorn でワーカーが複数あるとメモリは別々のため、同じ条件のリクエストが別プロセスに分散すると合流せず重複計算が起き得ます。横断で抑えたい場合はワーカー1、または分散ロック・外部キュー等の別仕組みが必要です（実装メモは `andersan_core/predict.py` の `_PREDICT_OX_INFLIGHT` 付近コメント参照）。
- 予測モデルは以下のバージョンが利用可能です：
  - `v0`: andersan0_1（12次タイル、8時間先予測）
  - `v0a`: andersan0_1_1
  - `v1`: andersan0_2
  - `v1a`: andersan0_2_1
  - `a1`: andersan1（直接回帰、24時間先まで）
- 分位点回帰モデル:
  - `a4_1`: andersan4_1（`/oxq/a4_1/...` で q10/q50/q90 を返す）

## 機能

### データ提供
- 都道府県ごとの大気環境データの提供
  - NMHC（非メタン炭化水素）
  - OX（オキシダント）
  - NOX（窒素酸化物）
  - 温度
  - 風速（X成分、Y成分）
- タイル形式でのデータ提供
- 生データの取得

### 予測機能
- OX（オキシダント）の予測モデル
- 確率テーブルの提供

### 位置情報サービス
- 緯度経度からの逆ジオコーディング機能

## 技術スタック

- **フレームワーク**: FastAPI
- **データベース**: SQLite
- **機械学習**: Keras
- **依存関係管理**: Poetry
- **開発環境**: Jupyter Notebook

## セットアップ

1. 依存関係のインストール:
```bash
poetry install
```

2. 環境変数の設定:
必要な環境変数を設定してください。

3. サーバーの起動:
```bash
poetry run uvicorn andersan-api:app --reload
```

