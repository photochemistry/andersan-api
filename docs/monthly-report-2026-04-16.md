# 運用ログ集計（数値詳細）

正式な月報（手採取ログ形式）は `/AIR/MONTHLY_REPORT_CHUCK4-2026-05-21.md` を参照。

本ファイルは andersan-api **サーバーログ**（2026-04-16 以降）の数値詳細のみを保持する。

**関連**: [`apw-grid-fetch-performance.md`](apw-grid-fetch-performance.md)

---

（以下、2026-05-21 時点の集計内容）

**対象期間**: 2026-04-16 〜 2026-05-21

| 指標 | 値 |
|------|-----|
| HTTP（ログ記録） | 20,388 |
| 予測 ERROR | 2,800 |
| APW タイムアウト行 | 2,550 |
| HTTP ≥10秒 | 14.0% |

詳細表・日別内訳はアーカイブのため省略。再集計は `logs/andersan-api.log` / `.log.1` / `prefetch.log` に対する grep・スクリプトで可能。
