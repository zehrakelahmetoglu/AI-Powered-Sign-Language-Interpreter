"""Gemini Vision API ile gerçek zamanlı TSL demo.

Kurulum:
  pip install google-generativeai pillow opencv-python

Çalıştırma (Windows PowerShell):
  $env:GEMINI_API_KEY = "YOUR_KEY_HERE"
  python 05_demo_gemini.py

Kontroller:
  SPACE — kayıt başlat (3 sn geri sayım → ~2 sn kayıt → Gemini'ye gönder)
  q     — çıkış
"""
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import google.generativeai as genai

COUNTDOWN_SEC = 3
RECORD_FRAMES = 48       # kaydedilecek toplam frame
SEND_FRAMES = 8          # Gemini'ye gönderilecek anahtar frame sayısı
GEMINI_MODEL = "gemini-2.0-flash"

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
C_RED    = (0, 0, 220)


def load_class_list(path: str = "data/class_mapping_full.json") -> tuple[list[str], list[str]]:
    """TR ve EN sınıf listelerini döner."""
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    tr_list = [mapping[str(i)]["tr"] for i in range(len(mapping))]
    en_list = [mapping[str(i)]["en"] for i in range(len(mapping))]
    return tr_list, en_list


def build_prompt(tr_classes: list[str]) -> str:
    class_list_str = ", ".join(tr_classes)
    return (
        "Bu görüntü dizisinde bir kişi Türk İşaret Dili (TİD / TSL) işareti yapmaktadır. "
        "Görüntüler sıralı kareler olup hareketi göstermektedir.\n\n"
        f"Aşağıdaki {len(tr_classes)} Türkçe kelimeden hangisini işaret ettiğini belirle:\n"
        f"{class_list_str}\n\n"
        "Kurallar:\n"
        "1. Yalnızca yukarıdaki listeden BİR kelime yaz.\n"
        "2. Hiçbir açıklama, noktalama veya ek kelime ekleme.\n"
        "3. Emin değilsen en yakın kelimeyi seç.\n\n"
        "Cevap:"
    )


def frames_to_pil(frames: list[np.ndarray], n: int) -> list[PIL.Image.Image]:
    """Eşit aralıklı n frame seç ve PIL Image listesine dönüştür."""
    if len(frames) <= n:
        indices = list(range(len(frames)))
    else:
        indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    result = []
    for i in indices:
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        result.append(PIL.Image.fromarray(rgb))
    return result


def query_gemini(
    model: genai.GenerativeModel,
    frames: list[np.ndarray],
    prompt: str,
    tr_classes: list[str],
    en_map: dict[str, str],
) -> tuple[str, str]:
    """Gemini'ye frame listesini gönderir, (tr, en) döner."""
    pil_images = frames_to_pil(frames, SEND_FRAMES)
    contents = [prompt] + pil_images

    try:
        response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=20,
            ),
        )
        raw = response.text.strip().lower().replace(".", "").replace(",", "")
        # En yakın eşleşmeyi bul
        tr_lower = [t.lower() for t in tr_classes]
        if raw in tr_lower:
            idx = tr_lower.index(raw)
            return tr_classes[idx], en_map[tr_classes[idx]]
        # Kısmi eşleşme dene
        for i, t in enumerate(tr_lower):
            if raw in t or t in raw:
                return tr_classes[i], en_map[tr_classes[i]]
        return raw, "?"
    except Exception as e:
        log.error("Gemini hatası: %s", e)
        return "HATA", str(e)[:40]


def draw_panel(
    frame: np.ndarray,
    result: tuple[str, str] | None,
    status: str,
    panel_w: int = 340,
) -> None:
    h, w = frame.shape[:2]
    x0 = w - panel_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, "TSL  x  Gemini", (x0 + 10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_WHITE, 2)
    cv2.line(frame, (x0 + 10, 40), (w - 10, 40), (80, 80, 80), 1)

    # Durum satırı
    cv2.putText(frame, status, (x0 + 10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_GRAY, 1)

    if result:
        tr_word, en_word = result
        # Büyük TR kelimesi
        cv2.rectangle(frame, (x0 + 5, 90), (w - 5, 160), (30, 60, 30), -1)
        font_scale = max(0.5, min(1.0, 10 / max(len(tr_word), 1)))
        cv2.putText(frame, tr_word.upper(), (x0 + 12, 138),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, C_YELLOW, 2)
        # EN karşılığı
        cv2.putText(frame, f"({en_word})", (x0 + 12, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_GRAY, 1)

        cv2.putText(frame, "Gemini tahmini", (x0 + 12, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 180, 100), 1)

    # Alt yardım metni
    cv2.putText(frame, "SPACE: kaydet   q: cikis", (x0 + 8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)


def main() -> None:
    # API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        api_key = input("Gemini API key: ").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY gerekli.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    log.info("Gemini modeli: %s", GEMINI_MODEL)

    tr_classes, en_classes = load_class_list()
    en_map = dict(zip(tr_classes, en_classes))
    prompt = build_prompt(tr_classes)
    log.info("%d sınıf yüklendi.", len(tr_classes))

    # Webcam — Windows'ta 0, WSL'de usbipd sonrası 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Webcam açılamadı (index=0).")
    log.info("Webcam açıldı. SPACE ile kayıt başlat.")

    recorded_frames: list[np.ndarray] = []
    result: tuple[str, str] | None = None
    countdown_start: float | None = None
    recording = False
    waiting_gemini = False

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame okunamadı.")
            break

        now = time.time()
        h, w = frame.shape[:2]

        # ── Geri sayım ──────────────────────────────────────────────────────
        if countdown_start is not None and not recording:
            elapsed = now - countdown_start
            remaining = COUNTDOWN_SEC - elapsed
            if remaining <= 0:
                recorded_frames = []
                recording = True
                countdown_start = None
            else:
                n = int(remaining) + 1
                cv2.putText(frame, str(n), (w // 2 - 50, h // 2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 60, 220), 10)
                cv2.putText(frame, "Hazirlanin...", (w // 2 - 110, h // 2 + 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_WHITE, 2)

        # ── Kayıt ───────────────────────────────────────────────────────────
        if recording:
            recorded_frames.append(frame.copy())
            pct = len(recorded_frames) / RECORD_FRAMES
            bar_w = w - 20
            cv2.rectangle(frame, (10, 10), (10 + int(bar_w * pct), 32), C_RED, -1)
            cv2.rectangle(frame, (10, 10), (w - 10, 32), C_WHITE, 2)
            cv2.putText(frame, f"KAYIT  {int(pct*100)}%", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1)

            if len(recorded_frames) >= RECORD_FRAMES:
                recording = False
                waiting_gemini = True
                status = "Gemini'ye gonderiliyor..."
                draw_panel(frame, result, status)
                cv2.imshow("TSL x Gemini", frame)
                cv2.waitKey(1)

                log.info("Gemini'ye %d frame gönderiliyor...", SEND_FRAMES)
                result = query_gemini(model, recorded_frames, prompt, tr_classes, en_map)
                log.info("Gemini tahmini: %s (%s)", result[0], result[1])
                waiting_gemini = False

        # ── Panel ───────────────────────────────────────────────────────────
        if waiting_gemini:
            status = "Gemini'ye gonderiliyor..."
        elif recording:
            status = "Islaret yapiniz..."
        elif countdown_start is not None:
            status = "Hazirlanin..."
        elif result:
            status = "Tamamlandi. SPACE ile yeniden dene."
        else:
            status = "SPACE'e bas, islaret yap."

        draw_panel(frame, result, status)
        cv2.imshow("TSL x Gemini", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" ") and not recording and countdown_start is None and not waiting_gemini:
            countdown_start = now
            log.info("Geri sayım başladı.")

    cap.release()
    cv2.destroyAllWindows()
    log.info("Demo kapatıldı.")


if __name__ == "__main__":
    main()
