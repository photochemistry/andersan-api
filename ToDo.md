時刻のうけわたしをdatetime objectに統一しようと思ったのだが、そうするとcacheに記録できなくなる問題が生じたので、やむなくこれまで通り文字列渡しとする。これで数時間を無駄にした。

andersan API 方針メモ:
- `andersan-api2.py` は、`/raw` を含む主要 I/O にタイムアウト制御を入れた次世代案。`/obs` は airpollutionwatch 側で提供する前提で API からは省いている。
- 当面は互換性優先で `andersan-api.py` を運用上の正とする。`api2` は将来の置き換え候補として扱う。
