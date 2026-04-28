"""
AI-Powered Sign Language Interpreter
=====================================
Module  : Local Dataset Processor
Author  : Atakan Yılmaz and Zeynep Ötegen (AI & Data)
Version : 1.1.0 (Optimizasyon Yaması - Zeynep)

Açıklama:
    Yerel diskteki ChaLearn / Local dataset videolarından (_color.mp4)
    MediaPipe Tasks API ile el landmark'ları çıkarır ve (30, 63)
    boyutunda .npy dosyaları olarak kaydeder.
    
    YENİ: Etiketler klasör isimlerinden değil, train_labels.csv dosyasından okunur.
"""

import os
import sys
import json
import time
import logging
import argparse
import urllib.request
import csv  # CSV okuma işlemi için eklendi
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE — Tasks API
# ─────────────────────────────────────────────────────────────

try:
    import mediapipe as mp
    BaseOptions        = mp.tasks.BaseOptions
    HandLandmarker     = mp.tasks.vision.HandLandmarker
    HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode
    MpImage            = mp.Image
    MpImageFormat      = mp.ImageFormat
except Exception as exc:
    raise SystemExit(
        f"[HATA] MediaPipe yüklenemedi: {exc}\n"
        "Çözüm: pip install mediapipe"
    ) from exc


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

DEFAULT_SOURCE  = Path("/media/atakan/UBUNTU 22_0/Sign_Language_Data")
PROJECT_ROOT    = Path(__file__).resolve().parent
DATA_DIR        = PROJECT_ROOT / "data"
MODEL_DIR       = PROJECT_ROOT / "models"
LOG_DIR         = PROJECT_ROOT / "logs"
MODEL_PATH      = MODEL_DIR / "hand_landmarker.task"
MODEL_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

FRAMES_PER_SEQ  = 30        # Hedef frame sayısı (collect_data ile aynı)
MIN_HAND_CONF   = 0.55      # El algılama minimum güven
MIN_TRACK_CONF  = 0.50      # Takip minimum güven
TARGET_FPS      = 15        # Frame örnekleme FPS'i

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / "local_processor.log",
            mode="a", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("local_processor")


# ─────────────────────────────────────────────────────────────
#  MODEL İNDİR
# ─────────────────────────────────────────────────────────────

def ensure_model() -> None:
    if MODEL_PATH.is_file():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("hand_landmarker.task indiriliyor (~25 MB)...")

    def _progress(block, bsize, total):
        pct = min(100, block * bsize * 100 // total)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] %{pct}", end="", flush=True)

    urllib.request.urlretrieve(str(MODEL_URL), str(MODEL_PATH), reporthook=_progress)
    print()
    log.info("Model indirme tamamlandı.")


# ─────────────────────────────────────────────────────────────
#  VİDEO TARAMA VE CSV EŞLEŞTİRME — Zeynep Optimizasyonu
# ─────────────────────────────────────────────────────────────

def scan_local_videos(source_dir: Path) -> list[dict]:
    """
    Kaynak dizindeki train_labels.csv dosyasını okur.
    Ardından dizindeki tüm _color.mp4 videolarını bulup CSV'deki etiketle eşleştirir.
    """
    videos = []
    if not source_dir.is_dir():
        log.error(f"Kaynak dizin bulunamadı: {source_dir}")
        return videos

    # 1. Rosetta Taşı'nı (CSV) Bul
    csv_path = source_dir / "train_labels.csv"
    if not csv_path.is_file():
        csv_path = source_dir.parent / "train_labels.csv"

    if not csv_path.is_file():
        log.error(f"KRİTİK HATA: train_labels.csv dosyası bulunamadı! Beklenen yer: {csv_path}")
        return videos

    # 2. CSV'yi Oku ve Şifreleri Çöz
    label_map = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                video_id = row[0].strip()  # Örn: 'signer0_sample1'
                label_id = row[1].strip()  # Örn: '41'
                label_map[video_id] = label_id

    log.info(f"CSV dosyasından {len(label_map)} adet etiket şifresi çözüldü.")

    # 3. Videoları Bul ve Eşleştir (rglob sayesinde alt klasörlere takılmaz)
    for video_file in sorted(source_dir.rglob("*_color.mp4")):
        stem = video_file.stem  # Örn: 'signer0_sample1_color'
        video_id = stem.replace("_color", "") # '_color' ekini atıp saf ID'yi buluyoruz

        label_num = label_map.get(video_id)

        if not label_num:
            continue

        videos.append({
            "path":  video_file,
            "label": f"CLASS_{label_num}", # Çıktı klasörleri karışmasın diye CLASS_41 formatı
            "stem":  stem,
        })

    log.info(
        f"Tarama tamamlandı: {len(videos)} _color.mp4 dosyası eşleşti "
        f"({len(set(v['label'] for v in videos))} farklı sınıf)"
    )
    return videos


# ─────────────────────────────────────────────────────────────
#  TEMPORAL RESAMPLING — Sabit 30 Frame
# ─────────────────────────────────────────────────────────────

def read_and_resample_frames(
    video_path: Path,
    target_frames: int = FRAMES_PER_SEQ,
    target_fps: int = TARGET_FPS,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Video açılamadı: {video_path.name}")
        return []

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = max(1, int(round(orig_fps / target_fps)))

    all_frames = []
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % skip != 0:
            continue

        all_frames.append(bgr)

    cap.release()

    if not all_frames:
        return []

    n = len(all_frames)

    if n == target_frames:
        return all_frames

    indices = np.linspace(0, n - 1, target_frames, dtype=int)
    resampled = [all_frames[i] for i in indices]

    return resampled


# ─────────────────────────────────────────────────────────────
#  LANDMARK ÇIKARIMI — Wrist-relative 63-d vektör
# ─────────────────────────────────────────────────────────────

def landmarks_to_vector(detection_result) -> np.ndarray:
    if not detection_result.hand_landmarks:
        return np.zeros(63, dtype=np.float32)

    hand  = detection_result.hand_landmarks[0]
    wrist = hand[0]

    coords = []
    for lm in hand:
        coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    arr = np.array(coords, dtype=np.float32)
    if len(arr) < 63:
        arr = np.pad(arr, (0, 63 - len(arr)))
    return arr[:63]


def extract_landmarks_fixed(
    frames: list[np.ndarray],
    landmarker,
) -> np.ndarray:
    vectors = []
    ts_ms = 0

    for bgr in frames:
        ts_ms += int(1000 / TARGET_FPS)

        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = MpImage(image_format=MpImageFormat.SRGB, data=rgb)

        try:
            result = landmarker.detect_for_video(mp_img, ts_ms)
            vec    = landmarks_to_vector(result)
        except Exception:
            vec = np.zeros(63, dtype=np.float32)

        vectors.append(vec)

    return np.array(vectors, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
#  TEK VİDEO İŞLEME — Bağımsız (multiprocessing uyumlu)
# ─────────────────────────────────────────────────────────────

def process_single_video(
    video_path: Path,
    label: str,
    output_name: str,
    data_dir: Path,
    model_path: Path,
) -> dict:
    out_dir  = data_dir / label
    out_file = out_dir / f"{output_name}.npy"

    if out_file.is_file():
        return {
            "status": "skip",
            "label":  label,
            "file":   str(out_file),
            "msg":    "zaten mevcut",
        }

    try:
        frames = read_and_resample_frames(video_path)
        if not frames or len(frames) != FRAMES_PER_SEQ:
            return {
                "status": "error",
                "label":  label,
                "file":   str(video_path),
                "msg":    f"Frame okunamadı ({len(frames) if frames else 0})",
            }

        opts = HandLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=MIN_HAND_CONF,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=MIN_TRACK_CONF,
        )

        with HandLandmarker.create_from_options(opts) as landmarker:
            sequence = extract_landmarks_fixed(frames, landmarker)

        nonzero = np.count_nonzero(np.any(sequence != 0, axis=1))
        if nonzero < FRAMES_PER_SEQ // 2:
            return {
                "status": "error",
                "label":  label,
                "file":   str(video_path),
                "msg":    f"Yetersiz el tespiti ({nonzero}/{FRAMES_PER_SEQ})",
            }

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_file), sequence)

        return {
            "status": "ok",
            "label":  label,
            "file":   str(out_file),
            "msg":    f"ok ({nonzero}/{FRAMES_PER_SEQ})",
        }

    except Exception as exc:
        return {
            "status": "error",
            "label":  label,
            "file":   str(video_path),
            "msg":    str(exc),
        }


# ─────────────────────────────────────────────────────────────
#  DOĞRULAMA RAPORU
# ─────────────────────────────────────────────────────────────

def print_verification_report(data_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("  Dataset Doğrulama Raporu")
    print("=" * 70)

    if not data_dir.is_dir():
        log.error(f"'{data_dir}' bulunamadı.")
        return

    total = 0
    ok_count = 0
    bad_count = 0

    print(f"  {'Sınıf':<30} {'Sekans':>8}  {'Shape':>12}  Durum")
    print("  " + "-" * 64)

    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        npys = sorted(cls_dir.glob("*.npy"))
        if not npys:
            continue

        try:
            sample = np.load(str(npys[0]))
            shape  = str(sample.shape)
            ok     = sample.shape == (FRAMES_PER_SEQ, 63)
        except Exception as exc:
            shape = str(exc)[:20]
            ok    = False

        count = len(npys)
        total += count

        if ok:
            ok_count += count
            icon = "✓"
            status = "OK"
        else:
            bad_count += count
            icon = "✗"
            status = "HATA"

        print(f"  {icon} {cls_dir.name:<28} {count:>8}  {shape:>12}  {status}")

    print("  " + "-" * 64)
    print(f"  TOPLAM: {total} sekans | OK: {ok_count} | Hatalı: {bad_count}")
    print(f"  Sınıf sayısı: {len(list(data_dir.iterdir()))}\n")


# ─────────────────────────────────────────────────────────────
#  ANA PIPELINE
# ─────────────────────────────────────────────────────────────

def run_processor(
    source_dir: Path,
    data_dir: Path        = DATA_DIR,
    limit: Optional[int]  = None,
    workers: int          = 1,
    skip_existing: bool   = True,
) -> None:
    t_start = time.time()

    print("=" * 70)
    print("  Local Dataset Processor — ChaLearn _color.mp4 → .npy")
    print("=" * 70)

    ensure_model()

    videos = scan_local_videos(source_dir)
    if not videos:
        log.error("İşlenecek video bulunamadı!")
        return

    if limit:
        videos = videos[:limit]
        log.info(f"Limit uygulandı: ilk {limit} video işlenecek")

    total     = len(videos)
    ok_count  = 0
    skip_count = 0
    err_count = 0
    error_log = []

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="İşleniyor", unit="video",
                    bar_format="{l_bar}{bar:40}{r_bar}")
    except ImportError:
        pbar = None

    if workers <= 1:
        opts = HandLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH.resolve())),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=MIN_HAND_CONF,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=MIN_TRACK_CONF,
        )

        with HandLandmarker.create_from_options(opts) as landmarker:
            for i, video in enumerate(videos):
                out_dir  = data_dir / video["label"]
                out_file = out_dir / f"{video['stem']}.npy"

                if skip_existing and out_file.is_file():
                    skip_count += 1
                    if pbar: pbar.update(1)
                    continue

                try:
                    frames = read_and_resample_frames(video["path"])
                    if not frames or len(frames) != FRAMES_PER_SEQ:
                        err_count += 1
                        if pbar: pbar.update(1)
                        continue

                    sequence = extract_landmarks_fixed(frames, landmarker)

                    nonzero = np.count_nonzero(np.any(sequence != 0, axis=1))
                    if nonzero < FRAMES_PER_SEQ // 2:
                        err_count += 1
                        if pbar: pbar.update(1)
                        continue

                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(str(out_file), sequence)
                    ok_count += 1

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(ok=ok_count, skip=skip_count, err=err_count)

                except Exception as exc:
                    err_count += 1
                    if pbar: pbar.update(1)

    else:
        log.info(f"Paralel mod: {workers} worker başlatılıyor...")
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for video in videos:
                future = executor.submit(
                    process_single_video,
                    video["path"], video["label"], video["stem"],
                    data_dir, MODEL_PATH.resolve(),
                )
                futures[future] = video

            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "ok":
                    ok_count += 1
                elif result["status"] == "skip":
                    skip_count += 1
                else:
                    err_count += 1

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, skip=skip_count, err=err_count)

    if pbar:
        pbar.close()

    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("  İŞLEM TAMAMLANDI")
    print("=" * 70)
    print(f"  Toplam video     : {total}")
    print(f"  Başarılı         : {ok_count}")
    print(f"  Atlanan (mevcut) : {skip_count}")
    print(f"  Başarısız        : {err_count}")
    print(f"  Süre             : {elapsed:.1f}s ({elapsed/60:.1f} dk)")
    print(f"  Hız              : {total / max(elapsed, 1):.1f} video/s")
    print("=" * 70)

    print("\n  Sınıf bazlı dağılım (en çok veri olanlar):")
    class_counts = {}
    for cls_dir in sorted(data_dir.iterdir()):
        if cls_dir.is_dir():
            cnt = len(list(cls_dir.glob("*.npy")))
            if cnt > 0:
                class_counts[cls_dir.name] = cnt

    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:20]:
        bar = "█" * min(cnt, 50) + "░" * max(0, 10 - cnt)
        print(f"  {cls:<28} {bar} {cnt}")

    print()
    print_verification_report(data_dir)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Local Dataset Processor")
    parser.add_argument("--source", type=str, required=True, help="Kaynak video dizini")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Çıktı dizini")
    parser.add_argument("--limit", type=int, default=None, help="İşlenecek max video")
    parser.add_argument("--workers", type=int, default=1, help="Paralel worker sayısı")
    parser.add_argument("--verify", action="store_true", help="Sadece doğrula")
    parser.add_argument("--no-skip", action="store_true", help="Mevcut dosyaları yeniden işle")

    args = parser.parse_args()

    if args.verify:
        print_verification_report(Path(args.data_dir))
        return

    run_processor(
        source_dir    = Path(args.source),
        data_dir      = Path(args.data_dir),
        limit         = args.limit,
        workers       = args.workers,
        skip_existing = not args.no_skip,
    )


if __name__ == "__main__":
    main()
