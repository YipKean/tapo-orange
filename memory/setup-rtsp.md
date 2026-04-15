# RTSP Setup (Tapo -> PC)

## Goal

Enable RTSP streaming from a Tapo camera to a PC for video processing.

Current project decision:

* Use Tapo RTSP as the default MVP camera path.
* Tuya feeder RTSP was tested, but it is a temporary cloud stream and added more latency.
* Revisit Tuya only if the tighter feeder framing becomes necessary after the Tapo pipeline is working.

```text
Tapo Camera -> RTSP -> PC (VLC / Python)
```

---

## 1. Enable RTSP in Tapo App

Open the Tapo app.

Go to:

```text
Camera -> Settings -> Advanced Settings -> Camera Account
```

Create a Camera Account:

* Username: for example `admin`
* Password: set a camera-specific password

Note:

* This is not your Tapo app login.
* This account is used for RTSP authentication.

---

## 2. Find Camera IP Address

### Option A - Router

* Log in to your router.
* Find the device list.
* Look for a device similar to `Tapo_Cxxx`.

### Option B - Windows

```powershell
arp -a
```

Look for a local IP like `192.168.x.x`.

### Option C - Mobile App

* Use a network scanner app such as Fing.

### Option D - Tapo App

* Open the camera settings and check network information.

---

## 3. RTSP URL Format

```text
rtsp://USERNAME:PASSWORD@IP:554/stream1
```

Example:

```text
rtsp://admin:123456@192.168.0.101:554/stream1
```

---

## 4. Stream Options

```text
/stream1 -> HD
/stream2 -> lower quality, usually lower load
```

Use `/stream2` if you need a lighter stream for testing.

---

## 5. Test with VLC

Use VLC Media Player.

Steps:

```text
Media -> Open Network Stream
```

Paste the RTSP URL and open it.

---

## 6. Expected Result

* Live camera feed appears.

---

## 7. Troubleshooting

### No video

Check:

* Wrong IP: verify the camera IP address.
* Wrong credentials: use the Camera Account, not the Tapo app login.
* Network issue: PC and camera must be on the same network.
* Firewall issue: allow VLC or port `554`.

---

## 8. Python Test

```python
import cv2

url = "rtsp://admin:123456@192.168.0.101:554/stream1"

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Tapo", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

Repository script:

```powershell
python scripts/tapo_opencv_test.py
```

This script reads `RTSP_URL` from `.env` by default.

Useful tuning flags:

```powershell
python scripts/tapo_opencv_test.py --width 960 --height 540 --zone 0.35,0.45,0.3,0.3 --motion-threshold 1.5 --process-fps 1 --snapshot-cooldown 10
```

`--zone` uses normalized `x,y,w,h` values between `0` and `1`.
`--process-fps 1` matches the planned idle-mode polling rate.
Snapshots are saved under `captures/` when motion first starts.
`--alert-seconds 4` raises a console alert when zone activity persists for 4 seconds.
Alerts that cross the dwell threshold are appended to `captures/events.log`.
When available, each logged alert includes the related snapshot filename.
If `--cat-model` points to a local YOLOv8 ONNX model, the alert condition changes from motion to cat-in-zone dwell.
`--cat-enter-frames` and `--cat-hold-seconds` smooth flickery detections so brief misses do not reset dwell immediately.
`--device cuda` requests OpenCV DNN GPU inference. If CUDA support is unavailable in the installed OpenCV build, the script falls back to CPU.
For actual PyTorch GPU inference, prefer a `.pt` model with Ultralytics, for example `models\yolov8s.pt`.

---

## 9. Notes

* Use `/stream2` if performance is slow.
* Keep camera position fixed for reliable zone detection.
* RTSP requires local network access.

---

## Checklist

* [ ] Camera account created
* [ ] Camera IP found
* [ ] RTSP URL constructed
* [ ] VLC test successful
