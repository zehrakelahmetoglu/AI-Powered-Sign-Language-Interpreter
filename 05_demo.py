"""Gerçek zamanlı Türk İşaret Dili demo.

WSL'de çalıştır (mediapipe 0.10.9 + Holistic gerektirir):
  python3 05_demo.py

Webcam WSL'de görünmüyorsa → Windows'ta usbipd ile bağla:
  usbipd list
  usbipd bind --busid <BUSID>
  usbipd attach --wsl --busid <BUSID>

Kontroller:
  q     — çıkış
  r     — buffer sıfırla
  SPACE — anlık tahmin
"""
import json
import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.model import SignLSTM

TARGET_FRAMES = 64
INPUT_DIM = 258
TOP_K = 5
INFERENCE_EVERY = 8

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

C_WHITE  = (255, 255, 255)
C_GRAY   = (160, 160, 160)
C_GREEN  = (0, 220, 0)
C_ORANGE = (0, 140, 255)
C_DARK   = (20, 20, 20)
C_YELLOW = (80, 220, 255)
BAR_COLORS = [(0, 230, 0), (0, 200, 80), (0, 170, 120), (0, 140, 160), (0, 110, 200)]


def load_class_map(mapping_path: str = "data/class_mapping_full.json", label_map_path: str = "models/label_map.json") -> dict[int, dict]:
    """Model Index -> Class ID -> Word mapping."""
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)
    with open(label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)
        
    index_to_class_id = {idx: class_str.replace("CLASS_", "") 
                         for class_str, idx in label_map.items() if "CLASS_" in class_str}
    if "MERHABA" in label_map:
        index_to_class_id[label_map["MERHABA"]] = "MERHABA"

    final_map = {}
    for idx, class_id in index_to_class_id.items():
        if class_id in mapping:
            final_map[idx] = mapping[class_id]
        elif class_id == "MERHABA":
            final_map[idx] = {"tr": "merhaba", "en": "hello"}
    return final_map


def extract_keypoints(results) -> np.ndarray:
    """Holistic sonuçlarından 258-boyutlu keypoint vektörü."""
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    
    # Toplanacak tüm noktalar (X ve Y)
    points_x = []
    points_y = []
    
    # Ham verileri topla
    pose_lms = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            pose_lms.append([lm.x, lm.y, lm.z, lm.visibility])
            points_x.append(lm.x); points_y.append(lm.y)
    else:
        pose_lms = [[0,0,0,0]] * 33
        
    lh_lms = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh_lms.append([lm.x, lm.y, lm.z])
            points_x.append(lm.x); points_y.append(lm.y)
    else:
        lh_lms = [[0,0,0]] * 21
        
    rh_lms = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh_lms.append([lm.x, lm.y, lm.z])
            points_x.append(lm.x); points_y.append(lm.y)
    else:
        rh_lms = [[0,0,0]] * 21

    # Normalizasyon: Vücudu 0-1 arasına sığdır (Bounding Box)
    if points_x and points_y:
        min_x, max_x = min(points_x), max(points_x)
        min_y, max_y = min(points_y), max(points_y)
        width = (max_x - min_x) + 1e-6
        height = (max_y - min_y) + 1e-6
        
        # Pose (33 * 4)
        for i, (x, y, z, v) in enumerate(pose_lms):
            vec[i*4 : i*4+4] = [(x - min_x)/width, (y - min_y)/height, z, v]
        # Hands (21 * 3)
        for i, (x, y, z) in enumerate(lh_lms):
            vec[132 + i*3 : 132 + i*3+3] = [(x - min_x)/width, (y - min_y)/height, z]
        for i, (x, y, z) in enumerate(rh_lms):
            vec[195 + i*3 : 195 + i*3+3] = [(x - min_x)/width, (y - min_y)/height, z]
            
    return vec


def run_inference(
    model: torch.nn.Module,
    buffer: deque,
    device: torch.device,
    class_map: dict,
) -> list[tuple[str, str, float]]:
    frames = list(buffer)
    if len(frames) < TARGET_FRAMES:
        pad = [np.zeros(INPUT_DIM, dtype=np.float32)] * (TARGET_FRAMES - len(frames))
        frames = pad + frames
    x = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
    top_idx = np.argsort(probs)[::-1][:TOP_K]
    return [(class_map[i]["tr"], class_map[i]["en"], float(probs[i])) for i in top_idx]


def draw_panel(
    frame: np.ndarray,
    predictions: list[tuple[str, str, float]],
    buffer_fill: float,
    panel_w: int = 330,
) -> None:
    h, w = frame.shape[:2]
    x0 = w - panel_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "TSL - Isaretler", (x0 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 2)
    cv2.line(frame, (x0 + 10, 38), (w - 10, 38), (80, 80, 80), 1)

    bar_x, bar_y, bar_w = x0 + 10, 50, panel_w - 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 50, 50), -1)
    filled_px = int(bar_w * min(buffer_fill, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_px, bar_y + 12),
                  C_GREEN if buffer_fill >= 1.0 else C_ORANGE, -1)
    status = "HAZIR" if buffer_fill >= 1.0 else f"Dolduruluyor %{int(buffer_fill * 100)}"
    cv2.putText(frame, status, (bar_x, bar_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)

    if not predictions:
        cv2.putText(frame, "64 frame dolunca baslar...", (x0 + 10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
        return

    for rank, (tr, en, conf) in enumerate(predictions):
        y = 100 + rank * 72
        if rank == 0:
            cv2.rectangle(frame, (x0 + 5, y - 20), (w - 5, y + 50), (30, 55, 30), -1)
        text_color = C_YELLOW if rank == 0 else C_WHITE
        cv2.putText(frame, f"#{rank + 1}  {tr.upper()}", (x0 + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58 if rank == 0 else 0.50,
                    text_color, 2 if rank == 0 else 1)
        cv2.putText(frame, f"({en})   {conf * 100:.1f}%", (x0 + 12, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GRAY, 1)
        bw = panel_w - 25
        by = y + 30
        cv2.rectangle(frame, (x0 + 12, by), (x0 + 12 + bw, by + 7), (50, 50, 50), -1)
        cv2.rectangle(frame, (x0 + 12, by), (x0 + 12 + int(bw * conf), by + 7),
                      BAR_COLORS[rank], -1)

    cv2.putText(frame, "q:cikis  r:sifirla  SPACE:tahmin", (x0 + 8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1)


def main() -> None:
    import mediapipe as mp_lib

    ckpt_path = Path("checkpoints/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError("Checkpoint bulunamadı: checkpoints/best_model.pt")

    class_map = load_class_map()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Cihaz: %s", device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model = SignLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=saved.get("hidden_dim", 512),
        num_layers=saved.get("num_layers", 3),
        num_classes=saved.get("num_classes", 226),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Model yüklendi | epoch=%d | val_acc=%.3f", ckpt["epoch"], ckpt["val_acc"])

    holistic = mp_lib.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw   = mp_lib.solutions.drawing_utils
    mp_styles = mp_lib.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError(
            "Webcam açılamadı (/dev/video0).\n"
            "Windows PowerShell'de (admin) şunu çalıştır:\n"
            "  usbipd list\n"
            "  usbipd bind --busid <BUSID>\n"
            "  usbipd attach --wsl --busid <BUSID>"
        )

    import time

    CONFIDENCE_THRESHOLD = 0.20  # altında "emin değilim" der
    COUNTDOWN_SEC = 3

    buffer: deque[np.ndarray] = deque(maxlen=TARGET_FRAMES)
    predictions: list[tuple[str, str, float]] = []
    recording = False
    recorded_frames = 0
    countdown_start: float | None = None  # geri sayım başlangıcı
    log.info("Demo başladı. SPACE=kayıt başlat  q=çıkış")

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame okunamadı, çıkılıyor.")
            break

        # Kamerayı yatayda çevir (Modelin eğitim verisiyle uyumlu olması için ŞART)
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks,
                mp_lib.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, results.left_hand_landmarks,
                mp_lib.solutions.holistic.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(121, 44, 250), thickness=2),
            )
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, results.right_hand_landmarks,
                mp_lib.solutions.holistic.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2),
            )

        kp = extract_keypoints(results)
        now = time.time()

        # Geri sayım aşaması
        if countdown_start is not None and not recording:
            elapsed = now - countdown_start
            remaining = COUNTDOWN_SEC - elapsed
            if remaining <= 0:
                # Geri sayım bitti → kayıt başla
                buffer.clear()
                recorded_frames = 0
                recording = True
                countdown_start = None
            else:
                # Büyük sayı göster
                n = int(remaining) + 1
                h, w = frame.shape[:2]
                cv2.putText(frame, str(n), (w // 2 - 30, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 80, 255), 8)
                cv2.putText(frame, "Hazirlanin...", (w // 2 - 100, h // 2 + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Kayıt aşaması
        if recording:
            buffer.append(kp)
            recorded_frames += 1
            if recorded_frames >= 64:  # Standart 64 frame
                predictions = run_inference(model, buffer, device, class_map)
                log.info("Tahmin: %s", predictions[0][0])
                recording = False

        # Son tahmini her zaman çiz
        draw_panel(frame, predictions, recorded_frames / 64 if recording else (1.0 if predictions else 0.0))

        if recording:
            pct = min(1.0, recorded_frames / 64)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 20), (30, 30, 30), -1)
            cv2.rectangle(frame, (20, h - 50), (20 + int((w - 40) * pct), h - 20), (0, 200, 0), -1)
            cv2.putText(frame, "KAYDEDILIYOR...", (w // 2 - 80, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif countdown_start is None:
            cv2.putText(frame, "SPACE: islaret kaydet (3 sn geri sayim)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Sonuç panelinde düşük confidence uyarısı
        if predictions and predictions[0][2] < CONFIDENCE_THRESHOLD:
            predictions_display = [("? EMIN DEGILIM ?", "", predictions[0][2])] + list(predictions[1:])
        else:
            predictions_display = predictions

        draw_panel(frame, predictions_display,
                   recorded_frames / TARGET_FRAMES if recording else (1.0 if predictions else 0.0))
        cv2.imshow("TSL Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and not recording and countdown_start is None:
            countdown_start = now
            log.info("%d saniye geri sayım başladı.", COUNTDOWN_SEC)

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    log.info("Demo kapatıldı.")


if __name__ == "__main__":
    main()
