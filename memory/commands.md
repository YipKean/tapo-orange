# Command Presets

# Notes
# - RTSP source: omit --video and use RTSP_URL from .env (or pass --url).
# - Footage source: use --video <path> and usually --no-snapshots.
# - --snapshot-cooldown 10 means max 1 snapshot every 10 seconds.
# - --cat-zone-overlap helps block walk-by edge-touch triggers.
# - --zone-polygon uses normalized points: "x1,y1;x2,y2;x3,y3;x4,y4".
# - Add --zone-edit to drag the zone with mouse, then press 'p' to print values.
# - Discord bot is now separate from OpenCV. Run scripts\discord_alert_bot.py in parallel.
# - Use --headless for replay/tuning runs without opening an OpenCV window.
# - Use --identity-debug-csv <path> to dump per-frame identity evidence for hard clips.
# - Identity tuning knobs for bad-light Orange/Goblin flicker:
#   --id-goblin-support-conf
#   --id-goblin-support-margin
#   --id-goblin-torso-white-min
#   --id-goblin-periphery-margin-max
#   --id-lock-margin / --id-switch-margin
# - Runtime Goblin lanes:
#   - POSSIBLE_GOBLIN = low-priority evidence event, no user mention
#   - ALERT: white-black cat in zone = confirmed Goblin alert, keeps user mention path
# - Set DISCORD_WEBHOOK_URL and optional DISCORD_USER_ID in .env for the bot.
# - Watchdog can launch a custom command from .env via WATCHDOG_TAPO_COMMAND.
# - Quick watchdog test: set idle threshold to 0 and check interval to 10s.

# Watchdog quick test (auto re-open in ~10 seconds after process exits)
powershell -ExecutionPolicy Bypass -File scripts\watchdog_tapo.ps1 -IdleThresholdSeconds 0 -CheckIntervalSeconds 10

# 1) RTSP production candidate (camera 1 Goblin)
python scripts/tapo_opencv_test.py --zone-polygon "0.31,0.40;0.46,0.40;0.46,0.62;0.31,0.60" --motion-threshold 1.4 --process-fps 6 --snapshot-cooldown 10 --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --device cuda

# 1a) RTSP production + save clip on alert (>4s dwell)
0.4158,0.5114;0.4863,0.5201;0.4814,0.6461;0.4102,0.6294
0.3727,0.5028;0.4453,0.5102;0.4439,0.6140;0.3696,0.6016
0.3588,0.5361;0.4338,0.5435;0.4328,0.6516;0.3588,0.6436

python scripts/tapo_opencv_test.py --zone-polygon "0.4158,0.5114;0.4863,0.5201;0.4814,0.6461;0.4102,0.6294" --motion-threshold 1.4 --process-fps 5 --snapshot-cooldown 10 --alert-seconds 4 --save-clip-on-alert --zone-edit --clip-seconds 10 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --cat-zone-overlap 0.3 --cat-imgsz 1920 --device cuda

# 1d) Start Discord bot separately (for Goblin alert pings from event-log)
python scripts/discord_alert_bot.py --poll-seconds 1.0


# 1b) RTSP zone calibration (drag and print updated zone)
python scripts/tapo_opencv_test.py --zone-polygon "0.31,0.40;0.46,0.40;0.46,0.62;0.31,0.60" --motion-threshold 1.4 --process-fps 10 --snapshot-cooldown 10 --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --zone-edit --device cuda

# 1c) Footage zone calibration (camera 1 Goblin, drag and print updated zone)
python scripts/tapo_opencv_test.py --video captures\1_goblin.mp4 --zone-polygon "0.31,0.40;0.46,0.40;0.46,0.62;0.31,0.60" --motion-threshold 1.4 --process-fps 6 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --zone-edit --device cuda

# 2) Camera 1 footage - Goblin (hard angle, skewed zone)
python scripts/tapo_opencv_test.py --video captures\1_goblin.mp4 --zone-polygon "0.31,0.40;0.46,0.40;0.46,0.62;0.31,0.60" --motion-threshold 1.4 --process-fps 10 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --device cuda

# 3) Camera 1 footage - Orange (same skewed zone)
python scripts/tapo_opencv_test.py --video captures\1_orange.mp4 --zone-polygon "0.31,0.40;0.46,0.40;0.46,0.65;0.31,0.63" --motion-threshold 1.4 --process-fps 10 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.0 --cat-detect-mode always --cat-imgsz 1920 --device cuda

# 4) Camera 1 footage - Orange BW
python scripts/tapo_opencv_test.py --video captures\1_orange_bw.mp4 --zone-polygon "0.31,0.40;0.46,0.40;0.46,0.65;0.31,0.63" --motion-threshold 1.4 --process-fps 10 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.0 --cat-detect-mode always --cat-imgsz 1920 --device cuda

# 5) Camera 2 footage - Orange
python scripts/tapo_opencv_test.py --video captures\2_orange.mp4 --zone 0.28,0.5,0.3,0.3 --motion-threshold 1.4 --process-fps 15 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8s.pt --cat-confidence 0.15 --cat-enter-frames 1 --cat-hold-seconds 0.8 --cat-detect-mode always --cat-imgsz 1600 --device cuda

# 6) Camera 2 footage - Goblin
python scripts/tapo_opencv_test.py --video captures\2_goblin.mp4 --zone 0.28,0.5,0.3,0.3 --motion-threshold 1.4 --process-fps 15 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8s.pt --cat-confidence 0.15 --cat-enter-frames 1 --cat-hold-seconds 0.8 --cat-detect-mode always --cat-imgsz 1600 --device cuda

# 7) Camera 3 footage - Orange (dark)
python scripts/tapo_opencv_test.py --video captures\3_orange_dark.mp4 --zone 0.1,0.5,0.3,0.3 --motion-threshold 1.4 --process-fps 15 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8s.pt --cat-confidence 0.15 --cat-enter-frames 1 --cat-hold-seconds 0.8 --cat-detect-mode always --cat-imgsz 1600 --device cuda

# 8) Camera 3 footage - Goblin
python scripts/tapo_opencv_test.py --video captures\3_goblin.mp4 --zone-polygon "0.2049,0.6059;0.3126,0.6072;0.2994,0.8073;0.2088,0.7826"--motion-threshold 1.4 --process-fps 15 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8s.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --save-clip-on-alert --zone-edit --clip-seconds 10 --cat-imgsz 2080 --device cuda

python scripts/tapo_opencv_test.py --video captures\3_goblin.mp4 --zone 0.1,0.5,0.35,0.3 --motion-threshold 1.4 --process-fps 15 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8s.pt --cat-confidence 0.05 --cat-enter-frames 1 --cat-hold-seconds 1 --cat-detect-mode always --cat-imgsz 1600 --device cuda

# 9) Camera 4 footage - Orange BW (night IR)
python scripts/tapo_opencv_test.py --video captures\4_orange_bw.mp4 --zone 0.4,0.5,0.2,0.3 --motion-threshold 1.4 --process-fps 15 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8s.pt --cat-confidence 0.06 --cat-enter-frames 1 --cat-hold-seconds 2.2 --cat-detect-mode always --cat-imgsz 1280 --cat-preprocess night-lite --device cuda

# 10) Generic footage sanity test
python scripts/tapo_opencv_test.py --video captures\sample_footage.mp4 --zone 0.28,0.4,0.22,0.24 --motion-threshold 1.4 --process-fps 10 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --cat-imgsz 1280 --device cuda

# 11) Identity test (Orange vs Goblin) - fill the video filename yourself
# Example expected runtime output:
# - "Cat entered zone (identity=orange ...)" for orange footage
# - "Cat entered zone (identity=white_black_dotted ...)" for goblin footage
0.4113,0.5238;0.4845,0.5324;0.4821,0.6393;0.4078,0.625
0.4220,0.5269;0.4845,0.5324;0.4838,0.6313;0.4227,0.6226
0.3953,0.5670;0.4533,0.5683;0.4509,0.6782;0.3953,0.6695
0.3852,0.5627;0.4595,0.5683;0.4588,0.6782;0.3828,0.6726

python scripts/tapo_opencv_test.py --zone-polygon "0.3953,0.5670;0.4533,0.5683;0.4509,0.6782;0.3953,0.6695" --zone-edit --save-clip-on-alert --motion-threshold 1.4 --process-fps 5 --snapshot-cooldown 10 --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --device cuda

python scripts/tapo_opencv_test.py --video captures\1_orange.mp4 --zone-polygon "0.31,0.40;0.46,0.40;0.46,0.65;0.31,0.63" --zone-edit --motion-threshold 1.4 --process-fps 5 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --device cuda

python scripts/tapo_opencv_test.py --video captures\1.1_orange_misidentified_20260415.mp4 --zone-polygon "0.31,0.40;0.46,0.40;0.46,0.65;0.31,0.63" --zone-edit --motion-threshold 1.4 --process-fps 5 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --device cuda

# 12) Headless replay for a suspected false Goblin clip with debug CSV
python scripts/tapo_opencv_test.py --video captures\clips\ALERT_2026-04-17_033355_GOBLIN.mp4 --motion-threshold 1.4 --process-fps 5 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --possible-goblin-seconds 2.0 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --device cuda --headless --identity-debug-csv tmp\replay_false_positive.csv

# 13) Headless replay for a trusted Goblin clip
python scripts/tapo_opencv_test.py --video captures\clips\ALERT_2026-04-15_221108_GOBLIN.mp4 --motion-threshold 1.4 --process-fps 5 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --possible-goblin-seconds 2.0 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --device cuda --headless

# 14) Bad-light anti-flicker tuning example
python scripts/tapo_opencv_test.py --video captures\clips\ALERT_2026-04-17_033355_GOBLIN.mp4 --motion-threshold 1.4 --process-fps 5 --snapshot-cooldown 0 --no-snapshots --alert-seconds 4 --cat-model models\yolov8m.pt --cat-confidence 0.08 --cat-enter-frames 1 --cat-hold-seconds 1.5 --possible-goblin-seconds 2.0 --cat-detect-mode always --cat-zone-overlap 0.25 --cat-imgsz 1920 --device cuda --headless --identity-debug-csv tmp\replay_tune.csv --id-goblin-support-conf 0.78 --id-goblin-support-margin 0.24 --id-goblin-torso-white-min 0.13 --id-goblin-periphery-margin-max 0.01 --id-lock-margin 0.13 --id-switch-margin 0.28

1_orange coordinate
--zone-polygon "0.31,0.40;0.46,0.40;0.46,0.65;0.31,0.63"

3 camera
 --zone-polygon "0.2049,0.6059;0.3126,0.6072;0.2994,0.8073;0.2088,0.7826"
