# ðŸ± Cat Monitoring System (Orange vs Goblin)

## 1. Overview

This project aims to detect and prevent **food stealing behavior** by Goblin:

* **Orange** â†’ calm, rightful eater
* **Goblin** â†’ food-motivated, steals food

The system monitors a **Tapo CCTV feed**, detects behavior, and triggers alerts when:

> â— Goblin accesses the food bowl (regardless of Orange)

---

## 2. Goals

### Primary Goal

* Prevent **Goblin from accessing food at any time**

### Secondary Goals

* Minimize false positives
* Run efficiently (idle most of the time)
* Work **without relying on phone presence**
* Be extendable (future AI reasoning layer)

---

## 3. System Architecture

```text
Tapo Camera (RTSP)
        â†“
PC (Python Service)
        â†“
[Idle Mode]
1-2 FPS frame polling
        â†“
Motion Detection (OpenCV)
        â†“
[Trigger Mode]
YOLO Cat Detection
        â†“
Zone Logic Engine
        â†“
Decision
        â†“
Alert (Sound via ESP32 + DFPlayer)
        â†“
Snapshot + Event Log
        â†“
Discord Alert Bot (separate Python process)
        â†“
Discord Webhook Ping
        â†“
(Optional) Save Clip â†’ AI Review (Molmo2)
```

---

## 4. Components

### 4.1 Camera Input

* Source: Tapo CCTV
* Protocol: RTSP
* Current decision: Use Tapo RTSP as the primary MVP camera source.
* Reason: Tapo local RTSP is lower latency and more stable for continuous detection than the temporary Tuya cloud RTSP stream.
* Tuya feeder camera remains a secondary option for experiments because its bowl framing may help cat identification.
* Example:

```
rtsp://<username>:<password>@<ip>:554/stream1
```

---

### 4.2 Idle Detection Loop

* Runs continuously at low cost
* Frequency: ~1-2 FPS
* Current implementation: zone motion polling on the RTSP stream

```python
while True:
    frame = get_frame()
    if motion_detected(frame):
        trigger_detection()
    sleep(1)
```

---

### 4.3 Motion Detection (OpenCV)

* Lightweight frame differencing
* Used to avoid running heavy AI constantly

---

### 4.4 Object Detection (YOLO)

* Model: YOLOv8 (current preferred baseline: `yolov8m.pt`)
* Detect:

  * cats
* Current implementation:

  * configurable cat detection mode (`motion` or `always`, default now `always`)
  * `.onnx` models via OpenCV DNN
  * `.pt` models via Ultralytics / PyTorch
  * GPU inference available through Ultralytics on CUDA
  * optional image-size tuning via `--cat-imgsz`
  * optional IR helper preprocessing via `--cat-preprocess night` / `night-lite`

---

### 4.5 Detection Persistence

* Method: temporal smoothing / short hold window
* Purpose:

  * reduce flickering detections
  * keep cat presence stable across brief missed frames

### 4.6 Evidence Capture

* In cat-detection mode, save one snapshot when cat enters zone (cooldown controlled)
* In motion-only mode, save one snapshot when motion starts
* Append threshold-reaching events to `event-log/YYYY-MM-DD.log`
* Include related snapshot filename in the event log when available
* Optional: save alert clip on dwell trigger via `--save-clip-on-alert --clip-seconds`

---

## 5. Zone System

### 5.1 Bowl Zone (Critical Zone)

* Tight rectangle or polygon around food bowl
* Used to detect **any access to food**

```python
bowl_zone = (x1, y1, x2, y2)
```

or

```python
bowl_zone_polygon = "x1,y1;x2,y2;x3,y3;x4,y4"
```

Current tooling:

* `--zone-polygon` for skewed perspective zones
* `--zone-edit` mouse editor (drag polygon points or entire polygon, press `p` to print values)
* `--cat-zone-overlap` to require minimum box-zone overlap and reduce walk-by edge triggers

---

## 6. Core Detection Logic (Current MVP)

Trigger alert if:

```text
1. Motion starts inside the bowl zone
AND
2. A cat is detected in the bowl zone
AND
3. Cat presence remains active for > X seconds
AND
4. Alert is logged and evidence is saved
```

Current logic notes:

* Cat detection can run on every processed frame (`--cat-detect-mode always`) to avoid dropouts when cat motion is small while eating.
* Dwell trigger still uses `--alert-seconds` (commonly 4s).
* For RTSP, frame-flush/catch-up logic prioritizes low latency (fresh frames) over processing every frame.
* For local video files, playback is throttled without runaway fast-forward.

---

### Rationale

* Use motion as a cheap wake-up gate
* Run heavier cat detection only during active events
* Build a stable alert pipeline before adding Goblin-vs-Orange identity logic

---

### Optional Enhancements

* Goblin-only identity classification
* Hybrid motion + cat presence logic for hard eating angles
* Tracking across detections
* Clip saving on alert

---

## 7. Cat Identification Strategy

### Phase 1 (MVP)

* Current behavior:

  * any cat entering bowl zone is treated as a trigger candidate
  * no Orange vs Goblin identity separation yet

### Phase 2

* Color detection:

  * Orange â†’ orange dominant
  * Goblin â†’ white/black

### Phase 3

* Improve identity with more robust heuristics or a trained classifier

---

## 8. Alert System

### Trigger Action

* Send signal to ESP32
* Play sound via DFPlayer + speaker
* Write structured alerts into `event-log/YYYY-MM-DD.log`
* Separate Discord bot reads Goblin alerts from event log and sends webhook notifications

### Example:

```text
PC â†’ HTTP â†’ ESP32 â†’ DFPlayer â†’ SOUND
```

Discord path:

```text
OpenCV alert log â†’ scripts/discord_alert_bot.py â†’ Discord webhook
```

Current ESP32 API baseline (`tapo-alarm`):

* `GET /health` â†’ health/status check
* `POST /api/alert` â†’ alert trigger endpoint from PC service
* `GET /api/play?track=<n>` â†’ manual DFPlayer track trigger
* `GET /api/ping-pc` â†’ optional reverse ping from ESP32 to PC endpoint

---

## 9. Clip Recording (Optional)

* Save 10â€“15 seconds clip when triggered
* Current implementation already saves snapshots and event logs

---

## 10. AI Review Layer (Future)

Model: Molmo2-8B or similar VLM

---

## 11. Performance Strategy

### Idle Mode

* 1-2 FPS motion polling

### Trigger Mode

* 3-10 FPS practical trigger loop, depending on detector backend
* Prefer GPU-backed Ultralytics for cat detection when available

---

## 12. Hardware Requirements

* RTX 3060
* ESP32
* DFPlayer Mini
* Micro SD (16GB)
* 3W 4Î© speaker

---

## 13. Constraints

* Camera angle must remain fixed
* Bowl position must be static
* Prefer local RTSP over temporary cloud streams for the real-time pipeline

---

## 14. Key Insight

> ðŸŽ¯ This is a **behavior blocking system**, not a full AI understanding system

Focus on:

* fast reaction
* simple logic
* reliable trigger

---

## 15. Development Roadmap

### Phase 1 (MVP)

* [x] RTSP working
* [x] Frame capture
* [x] Bowl zone defined (rectangle and polygon)
* [x] Motion detection in zone
* [x] Snapshot capture
* [x] Event log for dwell alerts
* [x] Basic cat detection
* [x] Zone calibration via mouse editor (`--zone-edit`)
* [x] Optional clip recording on alert dwell threshold
* [x] Goblin-only alert rule
* [ ] Sound trigger

### Phase 2

* [ ] Reduce false positives
* [ ] Add hybrid motion + cat presence logic
* [ ] Add tracking or stronger persistence
* [ ] Improve hard-angle eating detection

### Phase 3

* [ ] Cat identity (Orange vs Goblin)
* [ ] AI review layer

---

## 16. Decision Log

### 2026-04-08

* Proceed with Tapo RTSP first for the MVP.
* Tuya RTSP was validated, but the returned `rtsps://` stream is temporary and had about 2 seconds of latency.
* Tapo appeared to be about 1 second lower latency and is a better fit for fast bowl-access detection.
* Do not optimize for the perfect camera setup yet; first prove the detection pipeline with Tapo.

### 2026-04-09

* Added bowl-zone motion gating, dwell alerts, snapshots, and event logging.
* Added cat-in-zone detection as the next-stage trigger condition.
* Added detector smoothing to reduce flicker from missed detections.
* Switched the preferred GPU path from OpenCV DNN to Ultralytics / PyTorch for actual CUDA inference.

### 2026-04-10

* Current blocker is no longer raw performance.
* Main challenge is detection consistency at difficult feeder eating angles.
* Next likely improvement is hybrid motion + cat presence logic or stronger tracking/persistence before Goblin identity classification.

### 2026-04-11

* Added local footage mode with optional loop support to remove dependency on live cat testing.
* Added RTSP latency handling updates (buffer flushing and resilient retrieve failure handling).
* Changed default detection flow to support always-on cat inference in trigger mode for better eating-angle stability.
* Added `--no-snapshots` to disable evidence images during tuning sessions.
* Added model and tuning controls used in practice: `--cat-imgsz`, `--cat-preprocess`, `--cat-zone-overlap`.

### 2026-04-12

* Completed the current pipeline baseline for bowl monitoring with polygon-zone tooling, cat-enter snapshots, and optional alert clips.
* Event logging now writes daily files to `event-log/YYYY-MM-DD.log`, including lifecycle entries (`APP_START` / `APP_END` with reason and duration).
* Identity heuristic is active for `orange` vs `white_black_dotted`, with on-screen confidence labels.
* Low-light yellow-cast mitigation is active: visible black patch evidence can override orange tint and classify as `white_black_dotted`.
* Current operating mode is stability validation: keep this version unchanged and observe live behavior for 1-2 days before additional tuning or dataset work.

### 2026-04-13 to 2026-04-15

* Strengthened identity stability with temporal smoothing, hysteresis, and conservative Goblin switching.
* Added body-focused ROI filtering (center emphasis plus dark-boundary suppression) to reduce background contamination from door/wall edges.
* Added launch-source tracing in APP_START logs via `launch_origin` (default `manual`, watchdog uses `watchdog_ps1`).
* Updated watchdog defaults for daily use: PowerShell watchdog idle threshold is now 15 minutes.
* Updated alert clip naming to chronological format: `ALERT_YYYY-MM-DD_HHMMSS_{CATNAME}.mp4`.
* Added dual identity alert handling so Orange and Goblin can trigger independently when both persist beyond dwell threshold.
* Added Goblin-only stricter gate (higher confidence, evidence margin, and minimum 3-frame start) to reduce false Goblin alerts.
* Added standalone Discord alert bot (`scripts/discord_alert_bot.py`) that watches `event-log` and posts Goblin-only alerts to Discord webhook (`DISCORD_WEBHOOK_URL`).
* Discord mentions support single or multiple users (`DISCORD_USER_ID` or comma-separated `DISCORD_USER_IDS`).
* Discord message format now mirrors event log style: `[timestamp] ALERT_GOBLIN: white-black cat in zone lasted ... <@user...>`.
* Watchdog launch preset now uses the latest tuned polygon and overlap values (`--zone-polygon "0.4113,0.5238;0.4845,0.5324;0.4821,0.6393;0.4078,0.625"`, `--cat-zone-overlap 0.25`).
* Current operating decision: keep this build running continuously, collect clips/logs, and apply incremental tuning from observed failures.

### 2026-04-16

* Added `tapo-alarm` ESP32 firmware as a LAN HTTP server for sound-trigger integration.
* Confirmed route registration model with Arduino `WebServer` (`server.on(...)`, `server.begin()`, `server.handleClient()` loop).
* Implemented build-time env loading for firmware config via `tapo-alarm/scripts/load_env.py` and generated header `tapo-alarm/include/env_build.h`.
* Env keys now include Wi-Fi credentials and network config (`WIFI_SSID`, `WIFI_PASS`, `PC_PING_URL`, `API_TOKEN`, optional static IP keys).
* Clarified workflow: any `.env` change requires rebuild + re-upload because values are compiled into firmware.
* Diagnosed Wi-Fi failures on home mesh (`AUTH_EXPIRE`) and confirmed stable connection on a dedicated guest SSID.
* Added Wi-Fi diagnostics (scan output, RSSI/auth/channel/BSSID, disconnect reason logging).
* Added AP failover behavior: for matching SSID entries, ESP32 tries each BSSID for 10 seconds before switching to the next candidate and rescanning.
* Added optional static IP support in firmware using env keys (`ESP32_STATIC_IP`, `ESP32_GATEWAY`, `ESP32_SUBNET`, `ESP32_DNS`), with DHCP fallback.
* Current status: end-to-end API path is working; remaining hardware completion is DFPlayer SD card + speaker validation for final sound output.

---
