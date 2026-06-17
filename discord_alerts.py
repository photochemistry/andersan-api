"""andersan-api の深刻なシステムエラーを Discord に通知する。"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

ALERT_DISCORD_WEBHOOK_ENV = "ALERT_DISCORD_WEBHOOK_URL"
DISCORD_WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
SEVERE_STATUS_CODES = frozenset({500, 502, 503})
DEFAULT_COOLDOWN_SEC = 300

_pending_alert_detail: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "pending_alert_detail", default=None
)
_lock = threading.Lock()
_last_sent_at: dict[str, float] = {}


def is_severe_status(status_code: int) -> bool:
    return status_code in SEVERE_STATUS_CODES


def set_pending_alert_detail(detail: str) -> None:
    _pending_alert_detail.set(detail)


def clear_pending_alert_detail() -> None:
    _pending_alert_detail.set(None)


def take_pending_alert_detail() -> str | None:
    detail = _pending_alert_detail.get()
    _pending_alert_detail.set(None)
    return detail


def _api_keys_path() -> Path | None:
    base = Path(__file__).resolve().parent
    for candidate in (base / "api_keys.toml", base.parent / "api_keys.toml"):
        if candidate.is_file():
            return candidate
    return None


def _toml_value(key: str) -> str:
    path = _api_keys_path()
    if path is None:
        return ""
    match = re.search(
        rf'^\s*{re.escape(key)}\s*=\s*"([^"]+)"\s*$',
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def discord_webhook_url() -> str:
    env_alert = os.environ.get(ALERT_DISCORD_WEBHOOK_ENV, "").strip()
    env_legacy = os.environ.get(DISCORD_WEBHOOK_ENV, "").strip()
    if env_alert or env_legacy:
        return env_alert or env_legacy
    return _toml_value("alert_discord_webhook") or _toml_value("discord_webhook")


def _cooldown_sec() -> int:
    raw = os.environ.get("ANDERSAN_DISCORD_ALERT_COOLDOWN_SEC", "").strip()
    if not raw:
        return DEFAULT_COOLDOWN_SEC
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_COOLDOWN_SEC


def _cooldown_key(status_code: int, detail: str) -> str:
    return f"{status_code}:{detail[:200]}"


def _should_send(key: str) -> bool:
    cooldown = _cooldown_sec()
    if cooldown == 0:
        return True
    now = time.monotonic()
    with _lock:
        last = _last_sent_at.get(key)
        if last is not None and now - last < cooldown:
            return False
        _last_sent_at[key] = now
        return True


def _headline(status_code: int) -> str:
    return {
        500: "【ALERT】andersan-api 内部エラー",
        502: "【ALERT】andersan-api 上流(APW)取得失敗",
        503: "【ALERT】andersan-api サーバー設定不備",
    }.get(status_code, f"【ALERT】andersan-api HTTP {status_code}")


def _post_discord(content: str) -> None:
    webhook = discord_webhook_url()
    if not webhook:
        return
    payload = json.dumps({"content": content[:2000]}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "andersan-api/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.URLError as e:
        logger.warning("Discord alert failed: %s", e)
    except Exception as e:  # noqa: BLE001 - 通知失敗で API を止めない
        logger.warning("Discord alert failed: %s", e)


def notify_system_error(
    status_code: int,
    detail: str,
    *,
    method: str = "",
    path: str = "",
    context: str = "",
) -> None:
    """深刻な HTTP エラーを Discord に非同期送信する（同一内容はクールダウン）。"""
    if not is_severe_status(status_code):
        return
    key = _cooldown_key(status_code, detail)
    if not _should_send(key):
        return

    lines = [_headline(status_code), f"- HTTP: {status_code}"]
    if method and path:
        lines.append(f"- リクエスト: {method} {path}")
    if context:
        lines.append(f"- コンテキスト: {context}")
    lines.append(f"- 詳細: {detail[:500]}")
    text = "\n".join(lines)
    threading.Thread(target=_post_discord, args=(text,), daemon=True).start()
