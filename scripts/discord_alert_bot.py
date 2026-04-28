import argparse
import ctypes
import json
import mimetypes
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

CTRL_CLOSE_EVENT = 2
CTRL_LOGOFF_EVENT = 5
CTRL_SHUTDOWN_EVENT = 6

shutdown_ping_lock = threading.Lock()
shutdown_ping_sent = False


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
		description="Watch event logs and send Discord pings for Goblin alerts plus APP_START and APP_END events."
	)
	parser.add_argument(
		"--event-log-dir",
		default="event-log",
		help="Directory containing YYYY-MM-DD.log files (relative to repo root).",
	)
	parser.add_argument(
		"--snapshot-dir",
		default="captures",
		help="Directory containing alert snapshots referenced by snapshot=... log fields.",
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


def build_multipart_body(
	payload: dict[str, object],
	attachment_path: Path,
) -> tuple[bytes, str]:
	boundary = f"----tapo-discord-{time.time_ns()}"
	content_type = (
		mimetypes.guess_type(attachment_path.name)[0]
		or "application/octet-stream"
	)
	payload_json = json.dumps(payload).encode("utf-8")
	file_bytes = attachment_path.read_bytes()

	body_parts: list[bytes] = [
		f"--{boundary}\r\n".encode("utf-8"),
		b'Content-Disposition: form-data; name="payload_json"\r\n',
		b"Content-Type: application/json\r\n\r\n",
		payload_json,
		b"\r\n",
		f"--{boundary}\r\n".encode("utf-8"),
		(
			'Content-Disposition: form-data; name="files[0]"; '
			f'filename="{attachment_path.name}"\r\n'
		).encode("utf-8"),
		f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
		file_bytes,
		b"\r\n",
		f"--{boundary}--\r\n".encode("utf-8"),
	]
	return b"".join(body_parts), boundary


def send_discord_webhook(
	webhook_url: str,
	content: str,
	user_ids: list[str],
	timeout_s: float,
	user_agent: str,
	attachment_path: Path | None = None,
) -> tuple[bool, str]:
	payload: dict[str, object] = {"content": content}
	if user_ids:
		payload["allowed_mentions"] = {"users": user_ids}

	if attachment_path is not None:
		body, boundary = build_multipart_body(payload, attachment_path)
		headers = {
			"Content-Type": f"multipart/form-data; boundary={boundary}",
			"Accept": "application/json, text/plain, */*",
			"User-Agent": user_agent,
		}
	else:
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
			details = f"status={status}"
			if attachment_path is not None:
				details += f" attachment={attachment_path.name}"
			return True, details
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
	return (
		"alert: white-black cat in zone lasted" in text
		or "possible_goblin: cat in zone lasted" in text
	)


def is_standard_alert_line(line: str) -> bool:
	return "alert:" in line.lower()


def is_app_end_line(line: str) -> bool:
	return "app_end:" in line.lower()


def is_app_start_line(line: str) -> bool:
	return "app_start:" in line.lower()


def is_discord_event_line(line: str, include_test_alerts: bool = False) -> bool:
	return (
		is_goblin_alert_line(line)
		or is_app_start_line(line)
		or is_app_end_line(line)
		or (include_test_alerts and is_standard_alert_line(line))
	)


def is_rtsp_lifecycle_line(line: str) -> bool:
	text = line.lower()
	return "source=rtsp stream" in text


def extract_snapshot_name(line: str) -> str:
	match = re.search(r"(?:^|\s)snapshot=([^\s]+)", line)
	return match.group(1) if match else ""


def resolve_snapshot_path(snapshot_dir: Path, line: str) -> Path | None:
	snapshot_name = extract_snapshot_name(line)
	if not snapshot_name:
		return None

	candidate = (snapshot_dir / snapshot_name).resolve()
	try:
		candidate.relative_to(snapshot_dir)
	except ValueError:
		print(
			f"[{format_timestamp(time.time())}] Ignoring unsafe snapshot path: {snapshot_name}",
			file=sys.stderr,
		)
		return None

	if not candidate.is_file():
		print(
			f"[{format_timestamp(time.time())}] Snapshot not found for Discord attachment: {candidate}",
			file=sys.stderr,
		)
		return None

	return candidate


def parse_user_ids(raw_ids: str) -> list[str]:
	if not raw_ids:
		return []
	tokens = [token.strip() for token in raw_ids.replace(";", ",").split(",")]
	return [token for token in tokens if token]


def env_flag_enabled(raw_value: str, default: bool = True) -> bool:
	value = raw_value.strip().lower()
	if not value:
		return default
	return value not in {"0", "false", "no", "off", "disabled"}


def build_discord_message(alert_line: str, user_ids: list[str]) -> str:
	base = alert_line.strip()
	mention = " ".join(f"<@{uid}>" for uid in user_ids)
	if "ALERT: white-black cat in zone lasted" in base:
		base = base.replace(
			"] ALERT: white-black cat in zone lasted",
			"] ALERT_GOBLIN: white-black cat in zone lasted",
			1,
		)
		if mention:
			return f"{base} {mention}"
		return base
	if "APP_END:" in base:
		base = base.replace("] APP_END:", "] RTSP_ENDED:", 1)
		base = f"{base} Please monitor manually."
		if mention:
			return f"{base} {mention}"
		return base
	if "APP_START:" in base:
		base = base.replace("] APP_START:", "] RTSP_STARTED:", 1)
		if mention:
			return f"{base} {mention}"
		return base
	if mention:
		return f"{base} {mention}"
	return base


def send_bot_shutdown_message(
	webhook_url: str,
	timeout_s: float,
	user_agent: str,
) -> tuple[bool, str]:
	content = f"[{format_timestamp(time.time())}] DISCORD_BOT_ENDED: Discord alert bot turning off."
	return send_discord_webhook(
		webhook_url=webhook_url,
		content=content,
		user_ids=[],
		timeout_s=timeout_s,
		user_agent=user_agent,
	)


def send_bot_shutdown_ping_once(
	event_log_dir: Path,
	webhook_url: str,
	timeout_s: float,
	user_agent: str,
	reason: str,
) -> None:
	global shutdown_ping_sent

	with shutdown_ping_lock:
		if shutdown_ping_sent or not webhook_url:
			return
		shutdown_ping_sent = True

	ok, details = send_bot_shutdown_message(
		webhook_url=webhook_url,
		timeout_s=timeout_s,
		user_agent=user_agent,
	)
	status_line = (
		f"[{format_timestamp(time.time())}] DISCORD_BOT_SHUTDOWN_PING "
		f"{'OK' if ok else 'FAILED'} reason={reason} {details}"
	)
	print(status_line, flush=True)
	append_event_log(event_log_dir, status_line)


def register_windows_console_shutdown_handler(
	event_log_dir: Path,
	webhook_url: str,
	timeout_s: float,
	user_agent: str,
) -> object | None:
	if os.name != "nt":
		return None

	def handler(ctrl_type: int) -> bool:
		if ctrl_type not in {CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT}:
			return False
		reason_by_type = {
			CTRL_CLOSE_EVENT: "console_close",
			CTRL_LOGOFF_EVENT: "windows_logoff",
			CTRL_SHUTDOWN_EVENT: "windows_shutdown",
		}
		send_bot_shutdown_ping_once(
			event_log_dir=event_log_dir,
			webhook_url=webhook_url,
			timeout_s=timeout_s,
			user_agent=user_agent,
			reason=reason_by_type[ctrl_type],
		)
		return True

	handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
	handler_ref = handler_type(handler)
	if not ctypes.windll.kernel32.SetConsoleCtrlHandler(handler_ref, True):
		raise ctypes.WinError()
	return handler_ref


def send_bot_startup_message(
	webhook_url: str,
	timeout_s: float,
	user_agent: str,
) -> tuple[bool, str]:
	content = f"[{format_timestamp(time.time())}] DISCORD_BOT_STARTED: Discord alert bot is watching event logs."
	return send_discord_webhook(
		webhook_url=webhook_url,
		content=content,
		user_ids=[],
		timeout_s=timeout_s,
		user_agent=user_agent,
	)


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
	if not env_flag_enabled(os.environ.get("DISCORD_BOT_ENABLED", ""), default=True):
		print(
			f"[{format_timestamp(time.time())}] Discord bot disabled by DISCORD_BOT_ENABLED.",
		)
		return 0
	include_test_alerts = env_flag_enabled(
		os.environ.get("DISCORD_SEND_TEST_ALERTS", ""),
		default=False,
	)

	event_log_dir = (repo_root / args.event_log_dir).resolve()
	event_log_dir.mkdir(parents=True, exist_ok=True)
	snapshot_dir = (repo_root / args.snapshot_dir).resolve()

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
	shutdown_reason = "normal_exit"
	console_shutdown_handler = register_windows_console_shutdown_handler(
		event_log_dir=event_log_dir,
		webhook_url=webhook_url,
		timeout_s=args.discord_timeout,
		user_agent=args.discord_user_agent,
	)
	ok, details = send_bot_startup_message(
		webhook_url=webhook_url,
		timeout_s=args.discord_timeout,
		user_agent=args.discord_user_agent,
	)
	status_line = (
		f"[{format_timestamp(time.time())}] DISCORD_BOT_STARTUP_PING "
		f"{'OK' if ok else 'FAILED'} {details}"
	)
	print(status_line)
	append_event_log(event_log_dir, status_line)

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
				if not line or not is_discord_event_line(line, include_test_alerts):
					continue
				if (is_app_start_line(line) or is_app_end_line(line)) and not is_rtsp_lifecycle_line(line):
					continue
				if line in sent_signatures:
					continue
				sent_signatures.append(line)

				line_lower = line.lower()
				mention_users = (
					user_ids
					if (
						"alert: white-black cat in zone lasted" in line_lower
						or "app_start:" in line_lower
						or "app_end:" in line_lower
						or (include_test_alerts and is_standard_alert_line(line))
					)
					else []
				)
				content = build_discord_message(line, mention_users)
				attachment_path = (
					resolve_snapshot_path(snapshot_dir, line)
					if is_goblin_alert_line(line)
					or (include_test_alerts and is_standard_alert_line(line))
					else None
				)
				ok, details = send_discord_webhook(
					webhook_url=webhook_url,
					content=content,
					user_ids=mention_users,
					timeout_s=args.discord_timeout,
					user_agent=args.discord_user_agent,
					attachment_path=attachment_path,
				)
				status_line = (
					f"[{format_timestamp(time.time())}] DISCORD_BOT "
					f"{'OK' if ok else 'FAILED'} {details}"
				)
				print(status_line)
				append_event_log(event_log_dir, status_line)

			time.sleep(args.poll_seconds)
	except KeyboardInterrupt:
		shutdown_reason = "keyboard_interrupt"
		print(f"[{format_timestamp(time.time())}] Discord bot stopped.")
		return 0
	finally:
		_ = console_shutdown_handler
		send_bot_shutdown_ping_once(
			event_log_dir=event_log_dir,
			webhook_url=webhook_url,
			timeout_s=args.discord_timeout,
			user_agent=args.discord_user_agent,
			reason=shutdown_reason,
		)


if __name__ == "__main__":
	raise SystemExit(main())
