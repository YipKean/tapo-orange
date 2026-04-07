# 🎥 RTSP Setup (Tapo → PC)

## 🎯 Goal

Enable RTSP streaming from Tapo camera to PC for video processing.

```text
Tapo Camera → RTSP → PC (VLC / Python)
```

---

## 1. Enable RTSP in Tapo App

Open the **Tapo app**

Go to:

```text
Camera → Settings ⚙️ → Advanced Settings → Camera Account
```

Create a **Camera Account**:

* Username: (e.g. admin)
* Password: (e.g. 123456)

⚠️ Note:

* This is NOT your Tapo login
* This is used for RTSP authentication

---

## 2. Find Camera IP Address

### Option A — Router

* Login to router
* Find device list
* Look for:

```text
Tapo_Cxxx
```

---

### Option B — Windows

```bash
arp -a
```

Look for:

```text
192.168.x.x
```

---

### Option C — Mobile App (Recommended)

Use apps like:

* Fing

---
### Option D — Tapo App setting

Go to Tapo setting and look for Network setting.

---

## 3. RTSP URL Format

```text
rtsp://USERNAME:PASSWORD@IP:554/stream1
```

### Example:

```text
rtsp://admin:123456@192.168.0.101:554/stream1
```

---

## 4. Stream Options

```text
/stream1 → HD (recommended)
/stream2 → lower quality (faster)
```

---

## 5. Test with VLC

Use **VLC media player**

Steps:

```text
Media → Open Network Stream
```

Paste RTSP URL

---

## 6. Expected Result

✅ Live camera feed appears

---

## 7. Troubleshooting

### ❌ No video

Check:

#### 1. Wrong IP

* Verify camera IP

#### 2. Wrong credentials

* Must use Camera Account (not Tapo login)

#### 3. Network issue

* PC and camera must be on same network

#### 4. Firewall

* Allow VLC / port 554

---

## 8. Python Test (Optional)

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

---

## 9. Notes

* Use `/stream2` if performance is slow
* Keep camera position fixed (important for zone detection)
* RTSP requires local network access

---

## ✅ Checklist

* [ ] Camera account created
* [ ] Camera IP found
* [ ] RTSP URL constructed
* [ ] VLC test successful

---
