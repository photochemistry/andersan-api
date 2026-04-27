#!/usr/bin/env python3
"""
Andersan API のプリフェッチ実行スクリプト。

主用途:
- cron から定期起動して 1 回実行 (`--once`)
- systemd などで常駐し、一定間隔で繰り返し実行 (`--loop`)
"""

import argparse
import datetime as dt
import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable
from urllib import error, parse, request

import pytz


JST = pytz.timezone("Asia/Tokyo")
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "prefetch.log"
logger = logging.getLogger("prefetch")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False


def now_jst() -> dt.datetime:
    return dt.datetime.now(JST)


def to_hour_jst(base: dt.datetime) -> dt.datetime:
    return base.astimezone(JST).replace(minute=0, second=0, microsecond=0)


def hour_to_iso(hour: dt.datetime) -> str:
    return hour.astimezone(JST).isoformat()


def build_targets(
    base_url: str,
    prefectures: Iterable[str],
    models: Iterable[str],
    target_hour: dt.datetime,
    include_tile: bool,
    zoom: int,
) -> list[tuple[str, str]]:
    """(key, url) の一覧を返す。key は algorithm + area + hour。"""
    iso_hour = hour_to_iso(target_hour)
    encoded_hour = parse.quote(iso_hour, safe="")
    targets: list[tuple[str, str]] = []

    for pref in prefectures:
        for model in models:
            key = f"{model}/{pref}/{iso_hour}"
            # /a1 は /ox/a1 と等価だが、モデル一般化のため /ox/{model} を使う
            url = f"{base_url}/ox/{model}/{pref}/{encoded_hour}"
            targets.append((key, url))

        if include_tile:
            key = f"tile-z{zoom}/{pref}/{iso_hour}"
            url = f"{base_url}/tile/{zoom}/{pref}/{encoded_hour}"
            targets.append((key, url))

    return targets


def fetch_json(
    url: str,
    timeout_sec: int,
    *,
    client_id: str,
) -> dict:
    req = request.Request(url=url, method="GET")
    req.add_header("User-Agent", "AndersanPrefetch/1.0")
    req.add_header(
        "X-Andersan-Client",
        os.environ.get("ANDERSAN_PREFETCH_CLIENT", client_id),
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read()
        return json.loads(body.decode("utf-8"))


def fetch_with_retry(
    key: str,
    url: str,
    timeout_sec: int,
    max_retries: int,
    initial_backoff_sec: float,
    *,
    client_id: str,
) -> tuple[bool, str]:
    """指数バックオフ付きでフェッチする。"""
    for attempt in range(max_retries + 1):
        try:
            payload = fetch_json(
                url=url, timeout_sec=timeout_sec, client_id=client_id
            )
            meta = payload.get("meta", {})
            cached_at = meta.get("cached_at", "N/A")
            source_time = meta.get("source_time", "N/A")
            msg = (
                f"OK key={key} cached_at={cached_at} source_time={source_time}"
            )
            return True, msg
        except error.HTTPError as e:
            detail = f"HTTPError {e.code}"
        except error.URLError as e:
            detail = f"URLError {e.reason}"
        except TimeoutError:
            detail = "TimeoutError"
        except Exception as e:  # noqa: BLE001 - ログ用途
            detail = f"{type(e).__name__}: {e}"

        if attempt < max_retries:
            sleep_sec = initial_backoff_sec * (2**attempt)
            logger.warning(
                "key=%s attempt=%s error=%s; retry in %.1fs",
                key,
                attempt + 1,
                detail,
                sleep_sec,
            )
            time.sleep(sleep_sec)
        else:
            return False, f"NG key={key} error={detail}"

    return False, f"NG key={key} error=unexpected"


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_once(args: argparse.Namespace) -> int:
    base = now_jst() + dt.timedelta(hours=args.hour_offset)
    target_hour = to_hour_jst(base)
    targets = build_targets(
        base_url=args.base_url.rstrip("/"),
        prefectures=args.prefectures,
        models=args.models,
        target_hour=target_hour,
        include_tile=args.include_tile,
        zoom=args.zoom,
    )
    logger.info(
        "target_hour=%s targets=%s client_id=%s",
        hour_to_iso(target_hour),
        len(targets),
        args.client_id,
    )

    success_count = 0
    fail_count = 0
    details: list[str] = []
    for key, url in targets:
        ok, msg = fetch_with_retry(
            key=key,
            url=url,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            initial_backoff_sec=args.initial_backoff_sec,
            client_id=args.client_id,
        )
        if ok:
            logger.info("%s url=%s", msg, url)
        else:
            logger.error(msg)
        details.append(msg)
        if ok:
            success_count += 1
        else:
            fail_count += 1

    state = {
        "last_run_at": hour_to_iso(now_jst()),
        "target_hour": hour_to_iso(target_hour),
        "success_count": success_count,
        "fail_count": fail_count,
        "details": details,
    }
    save_state(Path(args.state_file), state)
    logger.info(
        "success=%s fail=%s state=%s",
        success_count,
        fail_count,
        args.state_file,
    )
    return 0 if fail_count == 0 else 1


def sleep_interval_minutes(interval_minutes: int) -> None:
    sleep_sec = max(1.0, float(interval_minutes) * 60.0)
    logger.info("next_run_in_sec=%.1f", sleep_sec)
    time.sleep(sleep_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Andersan API prefetch runner")
    parser.add_argument("--base-url", default="http://127.0.0.1:8087")
    parser.add_argument("--prefectures", nargs="+", default=["kanagawa"])
    parser.add_argument("--models", nargs="+", default=["a1"])
    parser.add_argument("--include-tile", action="store_true")
    parser.add_argument("--zoom", type=int, default=12)
    parser.add_argument("--hour-offset", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--initial-backoff-sec", type=float, default=1.5)
    parser.add_argument("--state-file", default=".prefetch/state.json")
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument(
        "--client-id",
        default="prefetch",
        help="APIログと照合するための X-Andersan-Client（環境変数 ANDERSAN_PREFETCH_CLIENT で上書き可）",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="1回だけ実行（cron向け）")
    mode.add_argument("--loop", action="store_true", help="毎正時+delayで繰り返し実行")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    if args.once:
        return run_once(args)

    while True:
        code = run_once(args)
        # loopモードでは継続実行する。失敗はログに残るので止めない。
        if code != 0:
            logger.warning("prefetch run had failures")
        sleep_interval_minutes(args.interval_minutes)


if __name__ == "__main__":
    raise SystemExit(main())
