# 🐱 Cat Monitoring System (Orange vs Goblin)

## 1. Overview

This project aims to detect and prevent **food stealing behavior** by Goblin:

* **Orange** → calm, rightful eater
* **Goblin** → food-motivated, steals food

The system monitors a **Tapo CCTV feed**, detects behavior, and triggers alerts when:

> ❗ Goblin accesses the food bowl (regardless of Orange)

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
        ↓
PC (Python Service)
        ↓
[Idle Mode]
1 FPS frame polling
        ↓
Motion Detection (OpenCV)
        ↓
[Trigger Mode]
YOLO Detection + Tracking
        ↓
Zone Logic Engine
        ↓
Decision
        ↓
Alert (Sound via ESP32 + DFPlayer)
        ↓
(Optional) Save Clip → AI Review (Molmo2)
```

---

## 4. Components

### 4.1 Camera Input

* Source: Tapo CCTV
* Protocol: RTSP
* Example:

```
rtsp://<username>:<password>@<ip>:554/stream1
```

---

### 4.2 Idle Detection Loop

* Runs continuously at low cost
* Frequency: ~1 FPS

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

* Model: YOLOv8 (n or s recommended)
* Detect:

  * cats

---

### 4.5 Tracking (Optional)

* Method: BYTETrack / DeepSORT
* Purpose:

  * maintain identity across frames

---

## 5. Zone System

### 5.1 Bowl Zone (Critical Zone)

* Tight rectangle around food bowl
* Used to detect **any access to food**

```python
bowl_zone = (x1, y1, x2, y2)
```

---

## 6. Core Detection Logic (Simplified)

Trigger alert if:

```text
1. Goblin enters bowl zone
AND
2. Goblin stays > N frames (e.g. 3–5 frames)
```

---

### Rationale

* Focus purely on **blocking Goblin behavior**
* No dependency on Orange detection
* Faster and more reliable

---

### Optional Enhancements

* Detect head overlap instead of full body
* Add cooldown between triggers
* Ignore very short entries (<1–2 frames)

---

## 7. Cat Identification Strategy

### Phase 1 (MVP)

* Assume:

  * any cat entering bowl zone = Goblin

### Phase 2

* Color detection:

  * Orange → orange dominant
  * Goblin → white/black

---

## 8. Alert System

### Trigger Action

* Send signal to ESP32
* Play sound via DFPlayer + speaker

### Example:

```text
PC → HTTP → ESP32 → DFPlayer → SOUND
```

---

## 9. Clip Recording (Optional)

* Save 10–15 seconds clip when triggered

---

## 10. AI Review Layer (Future)

Model: Molmo2-8B or similar VLM

---

## 11. Performance Strategy

### Idle Mode

* 1 FPS

### Trigger Mode

* 10–30 FPS burst

---

## 12. Hardware Requirements

* RTX 3060
* ESP32
* DFPlayer Mini
* Micro SD (16GB)
* 3W 4Ω speaker

---

## 13. Constraints

* Camera angle must remain fixed
* Bowl position must be static

---

## 14. Key Insight

> 🎯 This is a **behavior blocking system**, not a full AI understanding system

Focus on:

* fast reaction
* simple logic
* reliable trigger

---

## 15. Development Roadmap

### Phase 1 (MVP)

* [ ] RTSP working
* [ ] Frame capture
* [ ] Bowl zone defined
* [ ] Basic detection
* [ ] Sound trigger

### Phase 2

* [ ] Reduce false positives
* [ ] Add tracking

### Phase 3

* [ ] Cat identification
* [ ] AI review layer

---