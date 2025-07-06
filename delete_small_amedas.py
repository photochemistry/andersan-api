import sqlite3
import pickle
import json
from datetime import datetime

def delete_small_amedas():
    conn = sqlite3.connect('./airpollution.sqlite')
    cursor = conn.cursor()
    
    # 削除対象を確認
    cursor.execute("SELECT key, value, rowid FROM responses")
    to_delete = []
    
    for key, value, rowid in cursor.fetchall():
        try:
            response_data = pickle.loads(value) if isinstance(value, bytes) else value
            if isinstance(response_data, dict):
                url = response_data.get('url', '')
                if 'jma.go.jp/bosai/amedas/data/map/' in url:
                    content = response_data.get('_content', b'')
                    if isinstance(content, bytes):
                        content_size = len(content)
                        if content_size < 10240:  # 10kB以下
                            to_delete.append((rowid, key, url, content_size))
        except Exception:
            continue
    
    if not to_delete:
        print("削除対象のデータが見つかりませんでした")
        conn.close()
        return
    
    print(f"削除対象: {len(to_delete)}件")
    for rowid, key, url, size in to_delete:
        print(f"rowid={rowid} size={size} bytes url={url}")
    
    # 確認
    response = input("\n本当に削除しますか？ (y/N): ")
    if response.lower() != 'y':
        print("削除をキャンセルしました")
        conn.close()
        return
    
    # 削除実行
    deleted_count = 0
    for rowid, key, url, size in to_delete:
        try:
            cursor.execute("DELETE FROM responses WHERE rowid = ?", (rowid,))
            deleted_count += 1
            print(f"削除: rowid={rowid} url={url}")
        except Exception as e:
            print(f"削除失敗: rowid={rowid} error={e}")
    
    conn.commit()
    conn.close()
    print(f"\n削除完了: {deleted_count}件")

if __name__ == "__main__":
    delete_small_amedas() 