"""MediaPipe Tasks API import testi — collect_data.py öncesi çalıştır."""
import sys
print(f"Python: {sys.version.split()[0]}")

PASS, FAIL, WARN = "[OK]  ", "[FAIL]", "[WARN]"
errors = []

# ── Test 1: mediapipe ────────────────────────────────────────
try:
    import mediapipe as mp
    print(f"{PASS} mediapipe v{mp.__version__}")
except ImportError as e:
    print(f"{FAIL} mediapipe yok → pip install mediapipe")
    errors.append(str(e)); sys.exit(1)

# ── Test 2: Tasks API bileşenleri ───────────────────────────
try:
    BaseOptions        = mp.tasks.BaseOptions
    HandLandmarker     = mp.tasks.vision.HandLandmarker
    HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode
    MpImage            = mp.Image
    MpImageFormat      = mp.ImageFormat
    print(f"{PASS} mp.tasks.vision.HandLandmarker erişilebilir")
except AttributeError as e:
    print(f"{FAIL} Tasks API eksik: {e}")
    errors.append(str(e))

# ── Test 3: opencv ───────────────────────────────────────────
try:
    import cv2
    print(f"{PASS} opencv-python v{cv2.__version__}")
except ImportError:
    print(f"{FAIL} opencv-python yok → pip install opencv-python")
    errors.append("no cv2")

# ── Test 4: numpy ────────────────────────────────────────────
try:
    import numpy as np
    print(f"{PASS} numpy v{np.__version__}")
except ImportError:
    print(f"{FAIL} numpy yok → pip install numpy")
    errors.append("no numpy")

# ── Test 5: Model dosyası ────────────────────────────────────
import os
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
if os.path.isfile(model_path):
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"{PASS} hand_landmarker.task mevcut ({size_mb:.1f} MB)")
else:
    print(f"{WARN} hand_landmarker.task bulunamadı → collect_data.py ilk açılışta indirecek")

# ── Sonuç ────────────────────────────────────────────────────
print()
if errors:
    print(f"✗ {len(errors)} hata bulundu. Yukarıdaki FAIL satırlarını düzelt.")
else:
    print("✓ Tüm testler geçti. 'python collect_data.py' çalıştırabilirsin.")
