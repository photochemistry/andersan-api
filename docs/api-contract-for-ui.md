# Andersan API Contract for UI/AI

このファイルは `andersan-ui` から Cursor AI が参照しやすいように、主要APIの契約を固定化したものです。

## Base

- Base URL: 環境ごとに異なる（例: `http://localhost:8087`）
- 県パラメータ: `kanagawa` など `andersan.Neighbors` に含まれる値
- `datehour`: ISO8601（例: `2026-04-28T02:00+09:00`）または一部APIで `"now"`

## 共通レスポンス構造（地図タイル系）

ほとんどの予測APIは次の構造を返します。

- `spec`
  - `items`: 返却項目名配列
  - `X`, `Y`, `lon`, `lat`, `Z`, `timestamp` などの仕様情報
- `data`
  - `XY`, `lon`, `lat`
  - `items` に対応する配列（タイル点順）
- `meta`
  - `cached_at`: API生成時刻
  - `source_time`: 予測の基準時刻

## Endpoints

### 1) 通常回帰の濃度予測

- `GET /ox/{model}/{prefecture}/{datehour}`
- `model`: `v0`, `v0a`, `v1`, `v1a`, `a1`
- `data` の主なキー:
  - `+1`, `+2`, ...（モデルにより `+8` または `+24` まで）

### 2) 通常回帰の120ppb超過確率（確率換算表ベース）

- `GET /ox/{model}/pgt120/{prefecture}/{datehour}`
- `model`: `v0`, `v0a`, `v1`, `v1a`, `a1`
- 計算:
  - 予測値を5ppbビン化
  - `tables/{model_stem}.table.feather` から `P(obs>=120 | bin, hour)` を参照
- `data` の主なキー:
  - `+1_p_gt_120`, `+2_p_gt_120`, ...

### 3) 分位点回帰（andersan4_1）

- `GET /oxq/a4_1/{prefecture}/{datehour}`
- 内容:
  - 24時間先までの分位点 `q10/q50/q90`
- `data` の主なキー:
  - `+1_q10`, `+1_q50`, `+1_q90`, ..., `+24_q10`, `+24_q50`, `+24_q90`

### 4) 分位点回帰の120ppb超過確率（補間ベース）

- `GET /oxq/a4_1/pgt120/{prefecture}/{datehour}`
- 計算:
  - `q10/q50/q90` から単調区分線形CDFを構成
  - `P(OX>120) = 1 - F(120)`
- `data` の主なキー:
  - `+1_p_gt_120`, `+2_p_gt_120`, ..., `+24_p_gt_120`

### 5) 確率換算表の取得

- `GET /ptable/{model}`
- `model`: `v0`, `v0a`, `v1`, `v1a`, `a1`
- 用途:
  - UI側で独自に超過確率を計算する場合の参照テーブル

## UI実装上の注意

- `/ox/...` と `/oxq/...` は意味が違う
  - `/ox`: 濃度予測（単一値系列）
  - `/oxq`: 分位点予測
- 120ppb超過確率を使う場合は、まず専用APIを優先
  - 通常回帰: `/ox/{model}/pgt120/...`
  - 分位点回帰: `/oxq/a4_1/pgt120/...`
- `datehour=now` はサーバー側で正時に丸められる
