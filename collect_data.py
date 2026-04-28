"""
AI-Powered Sign Language Interpreter
=====================================
Module  : Data Collection
Author  : Atakan Yılmaz (AI & Data)
Version : 2.0.0  (MediaPipe Tasks API — v0.10.x+ / Python 3.13 uyumlu)

Açıklama:
    MediaPipe Tasks API kullanarak TSL hareketi veri toplama.
    Eski `mp.solutions` API'si kaldırıldığı için yeni Tasks API
    kullanılmaktadır. Model dosyası ilk çalıştırmada otomatik indirilir.

Klasör Yapısı:
    data/
    ├── MERHABA/
    │   ├── 0.npy   shape=(30, 63)
    │   └── ...
    └── ...

Kullanım:
    python collect_data.py
"""

import cv2
import numpy as np
import os
import time
import urllib.request
import sys

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE — Tasks API (v0.10.x+ / Python 3.13 uyumlu)
# ─────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    BaseOptions         = mp.tasks.BaseOptions
    HandLandmarker      = mp.tasks.vision.HandLandmarker
    HandLandmarkerOpts  = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode   = mp.tasks.vision.RunningMode
    MpImage             = mp.Image
    MpImageFormat       = mp.ImageFormat
except Exception as exc:
    raise SystemExit(
        f"[HATA] MediaPipe yüklenemedi: {exc}\n"
        "Çözüm: pip install mediapipe"
    ) from exc

# ─────────────────────────────────────────────────────────────
#  MODEL DOSYASI — İlk çalıştırmada otomatik indirilir
# ─────────────────────────────────────────────────────────────
_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(_MODEL_DIR, "hand_landmarker.task")
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(model_path: str, url: str) -> None:
    """Model dosyası yoksa Google'dan indir (~25 MB)."""
    if os.path.isfile(model_path):
        return
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"[MODEL] hand_landmarker.task indiriliyor (~25 MB)...")
    print(f"        Kaynak : {url}")
    print(f"        Hedef  : {os.path.abspath(model_path)}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r        [{bar}] %{pct}", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, model_path, reporthook=_progress)
        print(f"\n[MODEL] İndirme tamamlandı.")
    except Exception as exc:
        raise SystemExit(
            f"\n[HATA] Model indirilemedi: {exc}\n"
            "Lütfen modeli manuel olarak şuraya koy:\n"
            f"  {os.path.abspath(model_path)}\n"
            f"İndirme linki: {url}"
        ) from exc


# ─────────────────────────────────────────────────────────────
#  KONFIGÜRASYON
# ─────────────────────────────────────────────────────────────

CLASSES = [
    "MERHABA",
    "TESEKKUR",
    "EVET",
    "HAYIR",
    "LUTFEN",
    "TAMAM",
    "NASIL_SINIZ",
    "ISIM",
]

SEQUENCES_PER_CLASS = 50    # Her sınıf için örnek sayısı
FRAMES_PER_SEQUENCE = 30    # Her örnekteki frame sayısı
DELAY_BETWEEN_SEQ   = 2.0   # Çekimler arası bekleme (saniye)

CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ─────────────────────────────────────────────────────────────
#  RENK PALETİ (BGR)
# ─────────────────────────────────────────────────────────────

C_GREEN   = (0, 220, 100)
C_YELLOW  = (0, 220, 220)
C_RED     = (50, 60, 220)
C_WHITE   = (255, 255, 255)
C_DARK    = (30, 30, 30)
C_CYAN    = (220, 200, 0)

# ─────────────────────────────────────────────────────────────
#  EL BAĞLANTILARI — Manuel çizim için
# ─────────────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),   # Başparmak
    (0, 5),  (5, 6),  (6, 7),  (7, 8),   # İşaret
    (0, 9),  (9, 10), (10, 11),(11, 12),  # Orta
    (0, 13),(13, 14), (14, 15),(15, 16),  # Yüzük
    (0, 17),(17, 18), (18, 19),(19, 20),  # Serçe
    (5, 9), (9, 13), (13, 17),            # Avuç içi
]

FINGERTIP_IDS = {4, 8, 12, 16, 20}  # Parmak uçları


# ─────────────────────────────────────────────────────────────
#  LANDMARK ÇIKARIMI & NORMALİZASYON
# ─────────────────────────────────────────────────────────────

def extract_landmarks(detection_result) -> np.ndarray:
    """
    Tasks API sonuçlarından normalize landmark dizisi çıkar.

    Normalizasyon:
        Tüm koordinatlar bilek (index=0, WRIST) noktasına görece
        hesaplanır → elin kameradaki konumundan bağımsızlık sağlar.

    Dönüş:
        np.ndarray, shape=(63,), dtype=float32
        (El bulunamazsa sıfır vektörü)
    """
    hand_landmarks_list = detection_result.hand_landmarks

    if not hand_landmarks_list:
        return np.zeros(63, dtype=np.float32)

    # İlk eli al
    hand = hand_landmarks_list[0]
    wrist = hand[0]  # index 0 = WRIST

    coords = []
    for lm in hand:
        coords.extend([
            lm.x - wrist.x,   # bağıl x
            lm.y - wrist.y,   # bağıl y
            lm.z - wrist.z,   # bağıl z (derinlik)
        ])

    arr = np.array(coords, dtype=np.float32)

    # Güvenli padding (21 landmark × 3 = 63 değer)
    if len(arr) < 63:
        arr = np.pad(arr, (0, 63 - len(arr)))

    return arr[:63]


# ─────────────────────────────────────────────────────────────
#  LANDMARK ÇİZİMİ (cv2 tabanlı, mp.solutions bağımsız)
# ─────────────────────────────────────────────────────────────

def draw_hand_landmarks(
    frame: np.ndarray,
    detection_result,
) -> None:
    """Tasks API sonuçlarını cv2 ile frame üzerine çiz."""
    h, w = frame.shape[:2]

    for hand in detection_result.hand_landmarks:
        # Koordinatları piksel uzayına çevir
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

        # Bağlantı çizgileri
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, pts[start_idx], pts[end_idx], C_CYAN, 2, cv2.LINE_AA)

        # Landmark noktaları
        for idx, pt in enumerate(pts):
            color  = C_RED if idx in FINGERTIP_IDS else C_WHITE
            radius = 6 if idx in FINGERTIP_IDS else 4
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius, C_DARK, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  KLASÖR YÖNETİMİ
# ─────────────────────────────────────────────────────────────

def create_directory_structure(classes: list, data_dir: str) -> None:
    for cls in classes:
        os.makedirs(os.path.join(data_dir, cls), exist_ok=True)
    print(f"[INFO] Klasör yapısı hazır → {os.path.abspath(data_dir)}")


def get_next_sequence_index(class_dir: str) -> int:
    existing = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
    return len(existing)


# ─────────────────────────────────────────────────────────────
#  UI — HUD & SAYAÇ
# ─────────────────────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    class_name: str,
    seq_idx: int,
    total_seq: int,
    frame_idx: int,
    total_frames: int,
    status: str,
) -> None:
    """Ekrana bilgi bandı ve ilerleme çubuğu çiz."""
    # Üst bant
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (CAMERA_WIDTH, 58), C_DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"Kelime: {class_name}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Ornek: {seq_idx + 1}/{total_seq}",
                (270, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.82, C_YELLOW, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames}",
                (470, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_WHITE, 2, cv2.LINE_AA)

    # Durum ve ilerleme çubuğu (alt)
    st_color = C_YELLOW if status == "HAZIRLAN" else C_GREEN
    cv2.putText(frame, f"[ {status} ]",
                (10, CAMERA_HEIGHT - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.72, st_color, 2, cv2.LINE_AA)

    bx, by, bw, bh = 130, CAMERA_HEIGHT - 20, CAMERA_WIDTH - 140, 10
    fill = int(bw * (frame_idx + 1) / total_frames)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), st_color, -1)


def draw_countdown(frame: np.ndarray, seconds_left: float) -> None:
    """Geri sayım ekranı."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), C_DARK, -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    cv2.putText(frame, "Hazirlan...",
                (CAMERA_WIDTH // 2 - 130, CAMERA_HEIGHT // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, C_YELLOW, 3, cv2.LINE_AA)
    cv2.putText(frame, f"{seconds_left:.1f}",
                (CAMERA_WIDTH // 2 - 35, CAMERA_HEIGHT // 2 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, C_RED, 4, cv2.LINE_AA)


def draw_waiting_screen(frame: np.ndarray, class_name: str) -> None:
    """Sonraki sınıf bekleme ekranı."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (CAMERA_WIDTH, CAMERA_HEIGHT), C_DARK, -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)
    cv2.putText(frame, f"Sonraki: {class_name}",
                (CAMERA_WIDTH // 2 - 195, CAMERA_HEIGHT // 2 - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_GREEN, 3, cv2.LINE_AA)
    cv2.putText(frame, "SPACE=Basla  Q=Cik  S=Atla",
                (CAMERA_WIDTH // 2 - 210, CAMERA_HEIGHT // 2 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_WHITE, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
#  KAMERA
# ─────────────────────────────────────────────────────────────

def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"[HATA] Kamera açılamadı (index={index}).\n"
            "USB kameranın bağlı olduğundan emin ol."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"[INFO] Kamera açıldı — {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ 30fps")
    return cap


# ─────────────────────────────────────────────────────────────
#  TEK SINIF VERİ TOPLAMA
# ─────────────────────────────────────────────────────────────

def collect_class_data(
    cap: cv2.VideoCapture,
    landmarker: HandLandmarker,
    class_name: str,
    class_dir: str,
) -> int:
    """Bir sınıf için tüm sekansları topla. Toplanan sayısını döner."""
    start_seq  = get_next_sequence_index(class_dir)
    target_seq = start_seq + SEQUENCES_PER_CLASS
    collected  = 0
    ts_ms      = 0   # VIDEO modu için monoton artan timestamp

    print(f"\n[START] '{class_name}' — mevcut: {start_seq}, "
          f"hedef: +{SEQUENCES_PER_CLASS}")
    print("        Q=çık  S=bu sınıfı atla")

    for seq_idx in range(start_seq, target_seq):
        frames: list[np.ndarray] = []

        # ── Hazırlık geri sayımı ──────────────────────
        t0 = time.time()
        while time.time() - t0 < DELAY_BETWEEN_SEQ:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            draw_countdown(frame, DELAY_BETWEEN_SEQ - (time.time() - t0))
            cv2.imshow("TSL Veri Toplama", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return collected
            if key == ord("s"):
                print(f"[SKIP] '{class_name}' atlandı.")
                return collected

        # ── Frame kayıt döngüsü ───────────────────────
        for frame_idx in range(FRAMES_PER_SEQUENCE):
            ret, frame = cap.read()
            if not ret:
                frames.append(np.zeros(63, dtype=np.float32))
                continue

            frame = cv2.flip(frame, 1)

            # Tasks API: BGR→RGB → MpImage
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = MpImage(image_format=MpImageFormat.SRGB, data=rgb)

            # VIDEO modunda monoton timestamp zorunlu
            ts_ms += 33   # ~30fps ≈ 33ms/frame
            result = landmarker.detect_for_video(mp_img, ts_ms)

            # Landmark çiz
            draw_hand_landmarks(frame, result)

            # HUD
            draw_hud(frame, class_name,
                     seq_idx - start_seq, SEQUENCES_PER_CLASS,
                     frame_idx, FRAMES_PER_SEQUENCE, "KAYIT")

            cv2.imshow("TSL Veri Toplama", frame)

            # Landmark çıkar ve kaydet
            frames.append(extract_landmarks(result))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return collected

        # ── .npy kaydet ───────────────────────────────
        if len(frames) == FRAMES_PER_SEQUENCE:
            arr = np.array(frames, dtype=np.float32)  # shape: (30, 63)
            save_path = os.path.join(class_dir, f"{seq_idx}.npy")
            np.save(save_path, arr)
            collected += 1
            print(f"  [SAVED] {class_name}/{seq_idx}.npy  "
                  f"shape={arr.shape}  ({collected}/{SEQUENCES_PER_CLASS})")

    return collected


# ─────────────────────────────────────────────────────────────
#  ANA OTURUM
# ─────────────────────────────────────────────────────────────

def run_collection() -> None:
    print("=" * 62)
    print("  TSL Veri Toplama — v2.0  (MediaPipe Tasks API)")
    print("=" * 62)
    print(f"  mediapipe  : v{mp.__version__}")
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  Sınıflar   : {', '.join(CLASSES)}")
    print(f"  Örnek/sınıf: {SEQUENCES_PER_CLASS} × {FRAMES_PER_SEQUENCE} frame")
    print(f"  Kayıt yeri : {os.path.abspath(DATA_DIR)}")
    print("=" * 62)
    print("  KONTROLLER: SPACE=başla  Q=çık  S=sınıfı atla")
    print("=" * 62)

    # Model kontrolü / indir
    ensure_model(MODEL_PATH, MODEL_URL)

    # Klasörler
    create_directory_structure(CLASSES, DATA_DIR)

    # Kamera
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as exc:
        print(exc)
        return

    # HandLandmarker — VIDEO modu (senkron, thread-safe)
    opts = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total_collected = 0

    with HandLandmarker.create_from_options(opts) as landmarker:
        for class_name in CLASSES:
            class_dir = os.path.join(DATA_DIR, class_name)

            # Bekleme ekranı
            print(f"\n>>> Sıradaki sınıf: '{class_name}'")
            print("    Hazır olunca SPACE'e bas...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                draw_waiting_screen(frame, class_name)
                cv2.imshow("TSL Veri Toplama", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if key == ord("q"):
                    print("\n[EXIT] Kullanıcı çıkışı.")
                    cap.release()
                    cv2.destroyAllWindows()
                    _print_summary(total_collected)
                    return
                if key == ord("s"):
                    print(f"[SKIP] '{class_name}' atlandı.")
                    break

            try:
                n = collect_class_data(cap, landmarker, class_name, class_dir)
                total_collected += n
            except Exception as exc:
                print(f"[ERROR] '{class_name}': {exc}")
                continue

    cap.release()
    cv2.destroyAllWindows()
    _print_summary(total_collected)


def _print_summary(total: int) -> None:
    print("\n" + "=" * 62)
    print(f"  TAMAMLANDI — {total} sekans kaydedildi.")
    print(f"  Konum: {os.path.abspath(DATA_DIR)}")
    print("=" * 62)
    print("\n  Sınıf Özeti:")
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
            bar   = "█" * (count // 2) + "░" * max(0, 25 - count // 2)
            print(f"  {cls:<20} {bar}  {count} sekans")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_collection()
