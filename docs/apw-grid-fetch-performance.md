# APW グリッド取得の遅延・タイムアウト（調査メモ）

**作成**: 2026-05-21  
**状態**: 未対応（APW 側の処理能力改善は別リポジトリで実施予定）  
**関連コード**: `andersan/airmonitor.py`, `andersan_core/predict.py`, `andersan-api.py`  
**APW 実装**: `/AIR/airpollutionwatch-api`（`routers/grid.py`, `grid/interpolators/_idw.py`, `grid/cache.py`）

---

## 1. 症状

- OX 予測 API（例: `GET /ox/v0a/kanagawa/2026-05-21T21:00:00+09:00`）が **500** になり、処理に **20秒前後** かかることがある。
- ログ・例外例:

  ```
  RuntimeError: Missing observed data at 2026-05-21T20:00:00+09:00 (delta=-1)
  ```

- APW（`http://andersan.net:8089/v1/grid/field`）を直接叩くと **同じ時刻のデータは返る**（観測欠損ではない）。

---

## 2. 原因の整理

### 2.1 直接原因（andersan-api）

| 要因 | 現状 |
|------|------|
| HTTP `timeout` | **10秒**（`APW_FIELD_HTTP_TIMEOUT`、既定値） |
| 再試行 | **`API_TILES_MAX_RETRIES = 1`**（再試行なし） |
| 失敗時の戻り値 | 旧: `None` → `Missing observed data` と誤解しやすかった |

APW の応答が 10秒を超えると **Read timed out** となり、予測用 lookback の構築が中断される。  
特に `delta=-1`（基準時刻の1時間前）はフォールバックがなく、ここで落ちやすい。

### 2.2 根本原因（APW サーバー）

`/v1/grid/field` のキャッシュミス時、**bbox の大小に関わらず日本全国のタイル格子で IDW 補間**している。

```python
# airpollutionwatch-api/routers/grid.py（抜粋）
lon2d, lat2d = make_lonlat_grid_tiles(JAPAN_BBOX, z)  # 全国
field = interpolate_idw(lon, lat, values, lon2d, lat2d)
```

| 範囲 | z=12 の格子数（目安） |
|------|----------------------|
| 日本全国 | **310×274 ≈ 84,940 セル** |
| 神奈川 bbox のみ | **8×12 = 96 セル** |

レスポンスの `bbox` は **全国グリッドからの切り出し** であり、計算コストは全国分。

IDW 実装（`_idw.py`）はグリッド点ごとの Python ループのため、セル数にほぼ比例して遅い。

### 2.3 実測（2026-05-21、神奈川 bbox、`method=idw`）

| 条件 | 1回目 | 2回目以降（APW サーバーキャッシュヒット） |
|------|-------|------------------------------------------|
| `ox` のみ | 約 4〜15秒（キャッシュ状態による） | 約 **4〜8秒** |
| `nmhc,nox,ox` | 約 **11〜12秒** | 約 **8秒** |
| `timeout=10` で打ち切り | **失敗** | — |
| `timeout=60` で待つ | 約 **13秒で成功** | 続けて約 8秒 |

**キャッシュミス時は 10秒タイムアウトを超えやすい。**  
キャッシュヒット後も **4〜8秒** はかかる（全国 BLOB 読み出し・切り出し・JSON 化）。

---

## 3. キャッシュは二層ある

「2回目から速くなる」は **条件付き** で正しい。層が違う。

### 3.1 APW サーバー（`grid_cache.sqlite3`）

- キー例: `{datetime_hour}|{z}|{method}|{pollutant}|{smoothing}`
- **全国 IDW が完了したとき** に `put_cache`
- 物質（`ox`, `nmhc` など）ごとに別エントリ

### 3.2 andersan-api（`apw_field.sqlite` / `requests_cache`）

- キー: リクエスト URL + クエリパラメータ
- **HTTP が成功した応答だけ** 保存（タイムアウト・エラーは保存されない）
- TTL: 既定 **48時間**（`APW_FIELD_CACHE_SECONDS=172800`）

### 3.3 タイムアウト時に起きること

1. クライアント（andersan-api）が 10秒で接続を諦める → **取得失敗・予測中断**
2. APW 側はその後も計算を続け、完了すれば **サーバーキャッシュには載る**（13秒前後の例）
3. しかし andersan-api は **`max_retries=1`** のため **同じ時刻を即リトライしない**
4. 失敗は **`apw_field.sqlite` に残らない** → 次の予測でもネットワーク取得が必要

結果: **「APW には載ったが、andersan-api は一度も成功していない」** 状態が起きうる。

---

## 4. 予測1回で「毎回遅い」ように見える理由

`_prepare_data_for_nn` は lookback **24時刻** を順に `tiles_by_tiles` / APW へ問い合わせる。

| 理由 | 説明 |
|------|------|
| **時刻が24種類** | キャッシュは `(時刻, 物質)` 単位。別の `delta` は別ミス |
| **初回ミスが10秒超** | その時刻は失敗。サーバーが後から温めても、再試行なしなら救われない |
| **ヒットでも4〜8秒** | 24回 × 数秒でも合計は大きい。負荷時は再びタイムアウトもあり得る |
| **3物質まとめ取得** | 1 HTTP でも APW 内部では物質ごとの全国計算があり得る |

同じ `(県, 時刻)` で **前回予測がすべて成功**していれば、`apw_field.sqlite` ヒットで **ほぼ即** になる。  
**途中でタイムアウトした予測**の後は、体感「毎回遅い／毎回失敗」になりやすい。

---

## 5. 対策案（未実施・別途実装）

### 5.1 APW（`airpollutionwatch-api`）— 効果大

| 優先 | 内容 | 期待効果 |
|------|------|----------|
| 高 | **bbox 内だけ補間**（全国計算をやめる） | 8.5万セル → 数十〜数百セル |
| 高 | IDW の **ベクトル化**（`for` ループ削減） | CPU 時間短縮 |
| 中 | 重い計算の **バックグラウンド化** | クライアント切断後も完了→キャッシュ |
| 中 | キャッシュヒット時 **bbox 分だけ返す**（全国 BLOB 全読みしない） | ヒット時 4〜8秒 → さらに短縮 |
| 低 | 物質ごとの **並列計算** | 3物質まとめ取得の壁時計短縮 |

### 5.2 andersan-api — 補助（すぐ入れられる）

| 優先 | 内容 | 状態 |
|------|------|------|
| 中 | `APW_FIELD_HTTP_TIMEOUT` を **20〜30秒** に | 未実施（env で変更可能） |
| 中 | `API_TILES_MAX_RETRIES` を **2以上** | 未実施（`andersan-api.py`） |
| 済 | 失敗理由の明示（`ApwGridFetchError`） | 2026-05-21 実装済み |
| 低 | 予測用 lookback の **並列取得**（ワーカー占有との兼ね合い要検討） | 未実施 |

### 5.3 運用上の注意

- ログで `reason=timeout` と `Missing observed data` を区別する（後者は HTTP 成功後にテーブルが無い場合）。
- APW 改善後も、初回ミス・24時刻分の合計時間は監視対象。

---

## 6. andersan-api 側の実装済み変更（2026-05-21）

- `andersan.airmonitor.ApwGridFetchError`  
  - `reason`: `timeout`, `http_error`, `connection_error`, `invalid_response`, `empty_field` など  
  - メッセージに `datetime`, `pollutants`, `bbox`, `elapsed`, `timeout` を含む
- `andersan_core.predict`  
  - APW 取得失敗時: `Failed to fetch observed data from APW at ... (delta=...): ...`  
  - 真のデータ欠損時: `Missing observed data at ...`（HTTP は成功したが `None`）

環境変数:

- `APW_FIELD_HTTP_TIMEOUT`（秒、既定 `10`）
- `APW_FIELD_CACHE_SECONDS`（秒、既定 `172800`）

---

## 7. 再現・計測用コマンド（参考）

```bash
# APW 直接（神奈川 bbox、3物質）
curl -w "\ntime_total=%{time_total}\n" -m 60 \
  "http://andersan.net:8089/v1/grid/field?z=12&pollutant=nmhc,nox,ox&datetime=2026-05-21T20:00:00%2B09:00&bbox=138.94,35.13,139.84,35.66&method=idw"

# andersan-api 予測（失敗例）
curl -m 120 "http://127.0.0.1:8000/ox/v0a/kanagawa/2026-05-21T21:00:00+09:00"
```

---

## 8. 関連リンク

- UI/API 契約: `docs/api-contract-for-ui.md`
- APW グリッド API 実装: `airpollutionwatch-api/routers/grid.py`
- クライアント取得: `andersan/airmonitor.py` の `apw_tiles_bbox_`
