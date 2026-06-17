#!/usr/bin/env bash
# andersan-api + portforwarder を PM2 で起動・停止・再起動するヘルパー。
# 使い方: ./scripts/pm2-andersan-api.sh {start|stop|restart|status|logs}
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

APPS=(andersan-api andersan-portforwarder)

if ! command -v pm2 >/dev/null 2>&1; then
  echo "pm2 が見つかりません。例: npm install -g pm2" >&2
  exit 1
fi

mkdir -p logs

cmd="${1:-status}"
case "$cmd" in
  start)
    if ss -tln 2>/dev/null | grep -q ':8087 '; then
      echo "ポート 8087 は既に使用中です。手動プロセスを止めてから start してください。" >&2
      ss -tlnp 2>/dev/null | grep ':8087 ' || true
      exit 1
    fi
    pm2 start ecosystem.config.cjs
    pm2 save
    ;;
  stop)
    pm2 stop "${APPS[@]}" || true
    ;;
  restart)
    pm2 restart "${APPS[@]}"
    ;;
  status)
    pm2 status "${APPS[@]}"
    ;;
  logs)
    pm2 logs "${APPS[@]}" --lines 100
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 1
    ;;
esac
