import sqlite3
import pickle
import json
from datetime import datetime, timedelta
import sys

# 使い方: python3 extract_amedas_json.py 2025-07-06T10:00

def parse_args():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_amedas_json.py YYYY-MM-DDTHH:MM")
        sys.exit(1)
    try:
        target = datetime.fromisoformat(sys.argv[1])
    except Exception:
        print("時刻の形式は YYYY-MM-DDTHH:MM で指定してください")
        sys.exit(1)
    return target

def find_nearest_amedas(target_dt):
    conn = sqlite3.connect('./airpollution.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT key, value, rowid FROM responses")
    candidates = []
    for key, value, rowid in cursor.fetchall():
        try:
            response_data = pickle.loads(value) if isinstance(value, bytes) else value
            if isinstance(response_data, dict):
                url = response_data.get('url', '')
                if 'jma.go.jp/bosai/amedas/data/map/' in url:
                    # URLから時刻を抽出
                    # 例: .../20250706100000.json
                    try:
                        basename = url.split('/')[-1]
                        dtstr = basename.split('.')[0]  # 20250706100000
                        dt = datetime.strptime(dtstr, "%Y%m%d%H%M%S")
                        diff = abs((dt - target_dt).total_seconds())
                        candidates.append((diff, dt, key, value, url))
                    except Exception:
                        continue
        except Exception:
            continue
    conn.close()
    if not candidates:
        print("該当するamedasデータが見つかりませんでした")
        sys.exit(1)
    # 最も近いものを選ぶ
    candidates.sort()
    return candidates[0]  # (diff, dt, key, value, url)

def main():
    target_dt = parse_args()
    diff, dt, key, value, url = find_nearest_amedas(target_dt)
    print(f"指定時刻: {target_dt}")
    print(f"最も近いamedasデータ: {dt} (差分: {diff/60:.1f}分)")
    print(f"URL: {url}")
    # JSONデータを書き出す
    response_data = pickle.loads(value) if isinstance(value, bytes) else value
    content = response_data.get('_content', b'')
    if isinstance(content, bytes):
        content_str = content.decode('utf-8', errors='ignore')
        try:
            content_json = json.loads(content_str)
            outname = f"amedas_{dt.strftime('%Y%m%d%H%M%S')}.json"
            with open(outname, 'w', encoding='utf-8') as f:
                json.dump(content_json, f, ensure_ascii=False, indent=2)
            print(f"JSONデータを書き出しました: {outname}")
        except Exception as e:
            print(f"JSONデコード失敗: {e}")
            print(f"生データをamedas_raw.txtに保存します")
            with open('amedas_raw.txt', 'w', encoding='utf-8') as f:
                f.write(content_str)
    else:
        print("_contentがバイト列ではありません")

if __name__ == "__main__":
    main() 