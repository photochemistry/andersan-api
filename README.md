# andersan-api

大気環境データを提供するためのFastAPIベースのバックエンドサービスです。

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

## APIエンドポイント

- `/prefectures`: 都道府県一覧の取得
- `/items`: 利用可能なデータ項目の取得
- `/tiles/{item}/{z}/{x}/{y}.png`: タイル形式でのデータ提供
- `/raw/{item}/{prefecture}`: 生データの取得
- `/predict/ox/{prefecture}`: OXの予測データ取得
- `/probability_table/{prefecture}`: 確率テーブルの取得
- `/reverse_geocode`: 逆ジオコーディング

## ライセンス

このプロジェクトは[ライセンスファイル](LICENSE)に従って公開されています。
