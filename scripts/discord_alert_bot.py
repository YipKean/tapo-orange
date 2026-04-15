import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch event logs and send Discord pings for Goblin alerts."
    )
    parser.add_argument(
        "--event-log-dir",
        default="event-log",
        help="Directory containing YYYY-MM-DD.log files (relative to repo root).",
    )
    parser.add_argument(
        "--discord-webhook-url",
        help="Discord webhook URL. Defaults to DISCORD_WEBHOOK_URL from .env/env.",
    )
    parser.add_argument(
        "--discord-user-id",
        help=(
            "Discord user ID(s) to mention. "
            "Use comma-separated values. Defaults to DISCORD_USER_IDS or DISCORD_USER_ID."
        ),
    )
    parser.add_argument(
        "--discord-timeout",
        type=float,
        default=3.0,
        help="Timeout in seconds for Discord webhook requests.",
    )
    parser.add_argument(
        "--discord-user-agent",
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        help="User-Agent header for Discord webhook requests.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="How often to poll the active log file for new lines.",
    )
    parser.add_argument(
        "--start-from-beginning",
        action="store_true",
        help="Read the current day log from the beginning instead of tailing new lines only.",
    )
    return parser.parse_args()


def format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def event_log_file_for_timestamp(event_log_dir: Path, ts: float) -> Path:
    day_name = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    return event_log_dir / f"{day_name}.log"


def append_event_log(event_log_dir: Path, message: str, ts: float | None = None) -> None:
    event_ts = ts if ts is not None else time.time()
    log_path = event_log_file_for_timestamp(event_log_dir, event_ts)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def send_discord_webhook(
    webhook_url: str,
    content: str,
    user_ids: list[str],
    timeout_s: float,
    user_agent: str,
) -> tuple[bool, str]:
    payload: dict[str, object] = {"content": content}
    if user_ids:
        payload["allowed_mentions"] = {"users": user_ids}

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": user_agent,
    }
    req = urllib_request.Request(webhook_url, data=body, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            return True, f"status={status}"
    except urllib_error.HTTPError as exc:
        response_headers = " | ".join(f"{k}: {v}" for k, v in exc.headers.items())
        try:
            response_body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            response_body = ""
        details = f"http_error={exc.code}"
        if response_headers:
            details += f" headers={response_headers}"
        if response_body:
            details += f" body={response_body}"
        return False, details
    except urllib_error.URLError as exc:
        return False, f"url_error={exc.reason}"
    except TimeoutError:
        return False, "timeout"
    except Exception as exc:
        return False, f"error={exc}"


def is_goblin_alert_line(line: str) -> bool:
    text = line.lower()
    return "alert: white-black cat in zone lasted" in text


def parse_user_ids(raw_ids: str) -> list[str]:
    if not raw_ids:
        return []
    tokens = [token.strip() for token in raw_ids.replace(";", ",").split(",")]
    return [token for token in tokens if token]


def build_discord_message(alert_line: str, user_ids: list[str]) -> str:
    mention = " ".join(f"<@{uid}>" for uid in user_ids)
    base = alert_line.strip()
    base = base.replace(
        "] ALERT: white-black cat in zone lasted",
        "] ALERT_GOBLIN: white-black cat in zone lasted",
        1,
    )
    if mention:
        return f"{base} {mention}"
    return base


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    args = parse_args()

    if args.discord_timeout <= 0:
        print("--discord-timeout must be greater than 0.", file=sys.stderr)
        return 1
    if args.poll_seconds <= 0:
        print("--poll-seconds must be greater than 0.", file=sys.stderr)
        return 1

    event_log_dir = (repo_root / args.event_log_dir).resolve()
    event_log_dir.mkdir(parents=True, exist_ok=True)

    webhook_url = args.discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
    raw_user_ids = (
        args.discord_user_id
        or os.environ.get("DISCORD_USER_IDS", "")
        or os.environ.get("DISCORD_USER_ID", "")
    )
    user_ids = parse_user_ids(raw_user_ids)
    if not webhook_url:
        print(
            "Missing Discord webhook URL. Set DISCORD_WEBHOOK_URL in .env or pass --discord-webhook-url.",
            file=sys.stderr,
        )
        return 1

    print(f"[{format_timestamp(time.time())}] Discord bot started. Watching: {event_log_dir}")
    last_seen_day = ""
    file_pos = 0
    sent_signatures: deque[str] = deque(maxlen=200)

    try:
        while True:
            now = time.time()
            day_name = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
            log_path = event_log_dir / f"{day_name}.log"

            if day_name != last_seen_day:
                last_seen_day = day_name
                file_pos = 0
                if not args.start_from_beginning and log_path.exists():
                    file_pos = log_path.stat().st_size

            if not log_path.exists():
                time.sleep(args.poll_seconds)
                continue

            current_size = log_path.stat().st_size
            if current_size < file_pos:
                file_pos = 0

            with log_path.open("r", encoding="utf-8") as log_file:
                log_file.seek(file_pos)
                new_lines = log_file.readlines()
                file_pos = log_file.tell()

            for raw_line in new_lines:
                line = raw_line.strip()
                if not line or not is_goblin_alert_line(line):
                    continue
                if line in sent_signatures:
                    continue
                sent_signatures.append(line)

                content = build_discord_message(line, user_ids)
                ok, details = send_discord_webhook(
                    webhook_url=webhook_url,
                    content=content,
                    user_ids=user_ids,
                    timeout_s=args.discord_timeout,
                    user_agent=args.discord_user_agent,
                )
                status_line = (
                    f"[{format_timestamp(time.time())}] DISCORD_BOT "
                    f"{'OK' if ok else 'FAILED'} {details}"
                )
                print(status_line)
                append_event_log(event_log_dir, status_line)

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print(f"[{format_timestamp(time.time())}] Discord bot stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
