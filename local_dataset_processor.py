"""
AI-Powered Sign Language Interpreter
=====================================
Module  : Local Dataset Processor
Author  : Atakan Yılmaz and Zeynep Ötegen (AI & Data)
Version : 1.0.0

Açıklama:
    Yerel diskteki ChaLearn / Local dataset videolarından (_color.mp4)
    MediaPipe Tasks API ile el landmark'ları çıkarır ve (30, 63)
    boyutunda .npy dosyaları olarak kaydeder.

    Çıktı formatı collect_data.py ve youtube_dataset_builder.py ile
    birebir uyumludur → direkt model eğitimine girer.

Çıktı Yapısı:
    data/
    ├── MERHABA/
    │   ├── webcam_0.npy     shape=(30, 63)  ← mevcut veriler
    │   ├── local_vid1.npy   shape=(30, 63)  ← bu script'in çıktısı
    │   └── ...
    └── ...

Kullanım:
    python local_dataset_processor.py                            # tüm videoları işle
    python local_dataset_processor.py --limit 100                # ilk 100 video
    python local_dataset_processor.py --source /başka/yol/       # farklı kaynak
    python local_dataset_processor.py --workers 4                # paralel worker
    python local_dataset_processor.py --verify                   # sadece doğrulama
    python local_dataset_processor.py --no-skip                  # mevcut .npy'leri yeniden işle

Not:
    _depth.mp4 dosyaları otomatik olarak filtrelenir.
    Sadece _color.mp4 uzantılı dosyalar işlenir.
"""

import os
import sys
import json
import time
import logging
import argparse
import urllib.request
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
    """Model dosyası yoksa Google'dan indir (~25 MB)."""
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
#  VİDEO TARAMA — Sadece _color.mp4
# ─────────────────────────────────────────────────────────────

def scan_local_videos(source_dir: Path) -> list[dict]:
    """
    Kaynak dizini tarar ve işlenecek video listesini döner.

    Yapı beklentisi:
        source_dir/
        ├── SINIF_ADI/
        │   ├── video1_color.mp4
        │   ├── video1_depth.mp4   ← atlanır
        │   └── video2_color.mp4
        └── BASKA_SINIF/
            └── ...

    Her alt klasör adı doğrudan label olarak kullanılır.
    Sadece _color.mp4 uzantılı dosyalar hedeflenir.
    """
    videos = []

    if not source_dir.is_dir():
        log.error(f"Kaynak dizin bulunamadı: {source_dir}")
        return videos

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name.upper().strip()
        if not label:
            continue

        # _color.mp4 dosyalarını bul
        for video_file in sorted(class_dir.iterdir()):
            fname = video_file.name.lower()

            # Sadece _color.mp4 uzantılı dosyaları al
            if not fname.endswith("_color.mp4"):
                continue

            videos.append({
                "path":  video_file,
                "label": label,
                "stem":  video_file.stem,  # video_adi_color
            })

    log.info(
        f"Tarama tamamlandı: {len(videos)} _color.mp4 dosyası bulundu "
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
    """
    Videoyu okur ve tam olarak target_frames adet frame'e
    temporal resampling uygular.

    np.linspace ile uniform temporal sampling yapılır:
    - Kısa vidyolar: Frame'ler tekrarlanarak uzatılır
    - Uzun videolar: Eşit aralıklarla örneklenir

    Bu, zero-padding'den daha iyi temporal bilgi korur.

    Dönüş:
        List[np.ndarray] uzunluk=target_frames, her eleman BGR frame
        Boş liste: video açılamadı
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Video açılamadı: {video_path.name}")
        return []

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = max(1, int(round(orig_fps / target_fps)))

    # Tüm frame'leri (downsampled) oku
    all_frames = []
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # FPS downsample
        if frame_idx % skip != 0:
            continue

        all_frames.append(bgr)

    cap.release()

    if not all_frames:
        log.warning(f"Hiç frame okunamadı: {video_path.name}")
        return []

    n = len(all_frames)

    if n == target_frames:
        return all_frames

    # np.linspace ile uniform temporal resampling
    indices = np.linspace(0, n - 1, target_frames, dtype=int)
    resampled = [all_frames[i] for i in indices]

    return resampled


# ─────────────────────────────────────────────────────────────
#  LANDMARK ÇIKARIMI — Wrist-relative 63-d vektör
# ─────────────────────────────────────────────────────────────

def landmarks_to_vector(detection_result) -> np.ndarray:
    """
    Tasks API sonucundan wrist-relative 63-d vektör çıkar.
    collect_data.py / youtube_dataset_builder.py ile birebir aynı mantık.
    """
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
    """
    Tam olarak FRAMES_PER_SEQ adet frame'den (30, 63) landmark dizisi çıkar.

    Dönüş:
        np.ndarray shape=(30, 63) dtype=float32
        El bulunamayan frame'ler sıfır vektörü olur.
    """
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

    return np.array(vectors, dtype=np.float32)  # (30, 63)


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
    """
    Tek bir videoyu işler: okuma → resampling → landmark → kayıt.

    ProcessPoolExecutor'da çalışacağı için kendi landmarker'ını oluşturur.
    Her worker bağımsızdır — thread-safety sorunu yok.

    Dönüş:
        dict: {"status": "ok"|"skip"|"error", "label": ..., "file": ..., "msg": ...}
    """
    out_dir  = data_dir / label
    out_file = out_dir / f"{output_name}.npy"

    # Zaten varsa atla
    if out_file.is_file():
        return {
            "status": "skip",
            "label":  label,
            "file":   str(out_file),
            "msg":    "zaten mevcut",
        }

    try:
        # 1. Frame'leri oku ve resample et
        frames = read_and_resample_frames(video_path)
        if not frames or len(frames) != FRAMES_PER_SEQ:
            return {
                "status": "error",
                "label":  label,
                "file":   str(video_path),
                "msg":    f"Frame okunamadı veya yetersiz ({len(frames) if frames else 0})",
            }

        # 2. Landmarker oluştur (her process kendi instance'ını kullanır)
        opts = HandLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=MIN_HAND_CONF,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=MIN_TRACK_CONF,
        )

        with HandLandmarker.create_from_options(opts) as landmarker:
            # 3. Landmark çıkar
            sequence = extract_landmarks_fixed(frames, landmarker)

        # 4. Kalite kontrolü — en az %50 frame'de el bulunmalı
        nonzero = np.count_nonzero(np.any(sequence != 0, axis=1))
        if nonzero < FRAMES_PER_SEQ // 2:
            return {
                "status": "error",
                "label":  label,
                "file":   str(video_path),
                "msg":    f"Yetersiz el tespiti ({nonzero}/{FRAMES_PER_SEQ} frame)",
            }

        # 5. Kaydet
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(out_file), sequence)

        return {
            "status": "ok",
            "label":  label,
            "file":   str(out_file),
            "msg":    f"shape={sequence.shape}, el_frame={nonzero}/{FRAMES_PER_SEQ}",
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
    """data/ altındaki .npy dosyalarını doğrula ve rapor yaz."""
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
            status = "SHAPE HATASI"

        print(f"  {icon} {cls_dir.name:<28} {count:>8}  {shape:>12}  {status}")

    print("  " + "-" * 64)
    print(f"  TOPLAM: {total} sekans | OK: {ok_count} | Hatalı: {bad_count}")
    print(f"  Sınıf sayısı: {len(list(data_dir.iterdir()))}\n")


# ─────────────────────────────────────────────────────────────
#  ANA PIPELINE
# ─────────────────────────────────────────────────────────────

def run_processor(
    source_dir: Path      = DEFAULT_SOURCE,
    data_dir: Path        = DATA_DIR,
    limit: Optional[int]  = None,
    workers: int           = 1,
    skip_existing: bool    = True,
) -> None:
    """
    Ana işleme pipeline'ı.

    1. Kaynak dizini tara → _color.mp4 listesi
    2. Her video için: okuma → resampling → landmark → .npy kayıt
    3. Sonuç raporu yazdır
    """
    t_start = time.time()

    print("=" * 70)
    print("  Local Dataset Processor — ChaLearn _color.mp4 → .npy")
    print("=" * 70)
    print(f"  mediapipe   : v{mp.__version__}")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  Kaynak      : {source_dir}")
    print(f"  Çıktı       : {data_dir}")
    print(f"  Workers     : {workers}")
    print(f"  Frame/seq   : {FRAMES_PER_SEQ} | FPS hedef: {TARGET_FPS}")
    print(f"  Skip mevcut : {'Evet' if skip_existing else 'Hayır'}")
    print("=" * 70)

    # Model kontrol
    ensure_model()

    # Video listesi
    videos = scan_local_videos(source_dir)
    if not videos:
        log.error("İşlenecek video bulunamadı!")
        return

    if limit:
        videos = videos[:limit]
        log.info(f"Limit uygulandı: ilk {limit} video işlenecek")

    # İlerleme sayaçları
    total     = len(videos)
    ok_count  = 0
    skip_count = 0
    err_count = 0

    # Hata logları
    error_log = []

    # tqdm opsiyonel
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="İşleniyor", unit="video",
                    bar_format="{l_bar}{bar:40}{r_bar}")
    except ImportError:
        pbar = None
        log.info("(tqdm kurulu değil, basit ilerleme gösteriliyor)")

    if workers <= 1:
        # ── Sıralı İşleme (debug dostu) ─────────────────────
        # Tek landmarker oluştur, tüm videolarda kullan (daha verimli)
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

                # Skip kontrolü
                if skip_existing and out_file.is_file():
                    skip_count += 1
                    if pbar:
                        pbar.update(1)
                    else:
                        _print_progress(i + 1, total, video["label"], "ATLA")
                    continue

                try:
                    # Frame oku ve resample et
                    frames = read_and_resample_frames(video["path"])
                    if not frames or len(frames) != FRAMES_PER_SEQ:
                        err_count += 1
                        error_log.append({
                            "file": str(video["path"]),
                            "label": video["label"],
                            "error": f"Yetersiz frame: {len(frames) if frames else 0}",
                        })
                        if pbar:
                            pbar.update(1)
                        else:
                            _print_progress(i + 1, total, video["label"], "HATA (frame)")
                        continue

                    # Landmark çıkar
                    sequence = extract_landmarks_fixed(frames, landmarker)

                    # Kalite kontrolü
                    nonzero = np.count_nonzero(np.any(sequence != 0, axis=1))
                    if nonzero < FRAMES_PER_SEQ // 2:
                        err_count += 1
                        error_log.append({
                            "file": str(video["path"]),
                            "label": video["label"],
                            "error": f"Yetersiz el tespiti: {nonzero}/{FRAMES_PER_SEQ}",
                        })
                        if pbar:
                            pbar.update(1)
                        else:
                            _print_progress(i + 1, total, video["label"],
                                            f"HATA (el:{nonzero}/{FRAMES_PER_SEQ})")
                        continue

                    # Kaydet
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(str(out_file), sequence)
                    ok_count += 1

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(ok=ok_count, skip=skip_count, err=err_count)
                    else:
                        _print_progress(i + 1, total, video["label"],
                                        f"OK (el:{nonzero}/{FRAMES_PER_SEQ})")

                except Exception as exc:
                    err_count += 1
                    error_log.append({
                        "file": str(video["path"]),
                        "label": video["label"],
                        "error": str(exc),
                    })
                    if pbar:
                        pbar.update(1)
                    else:
                        _print_progress(i + 1, total, video["label"], f"HATA: {exc}")

    else:
        # ── Paralel İşleme (ProcessPoolExecutor) ─────────────
        log.info(f"Paralel mod: {workers} worker başlatılıyor...")

        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for video in videos:
                future = executor.submit(
                    process_single_video,
                    video["path"],
                    video["label"],
                    video["stem"],
                    data_dir,
                    MODEL_PATH.resolve(),
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
                    error_log.append({
                        "file":  result["file"],
                        "label": result["label"],
                        "error": result["msg"],
                    })

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, skip=skip_count, err=err_count)

    if pbar:
        pbar.close()

    elapsed = time.time() - t_start

    # ── Hata logunu JSON'a yaz ────────────────────────────
    if error_log:
        err_path = LOG_DIR / "processor_errors.json"
        with open(str(err_path), "w", encoding="utf-8") as f:
            json.dump(error_log, f, ensure_ascii=False, indent=2)
        log.info(f"Hata detayları: {err_path}")

    # ── Özet Rapor ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  İŞLEM TAMAMLANDI")
    print("=" * 70)
    print(f"  Toplam video     : {total}")
    print(f"  Başarılı         : {ok_count}")
    print(f"  Atlanan (mevcut) : {skip_count}")
    print(f"  Başarısız        : {err_count}")
    print(f"  Süre             : {elapsed:.1f}s ({elapsed/60:.1f} dk)")
    print(f"  Hız              : {total / max(elapsed, 1):.1f} video/s")
    print(f"  Çıktı dizin      : {data_dir.resolve()}")
    print("=" * 70)

    # Sınıf bazlı istatistik
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


def _print_progress(current: int, total: int, label: str, status: str) -> None:
    """tqdm yokken basit ilerleme çıktısı."""
    pct = current * 100 // total
    print(f"  [{current:>5}/{total}] ({pct:>3}%) {label:<28} {status}")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Dataset Processor — ChaLearn _color.mp4 → .npy",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source", type=str, default=str(DEFAULT_SOURCE),
        help=f"Kaynak video dizini (varsayılan: {DEFAULT_SOURCE})"
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help=f"Çıktı dizini (varsayılan: {DATA_DIR})"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="İşlenecek max video sayısı (test için)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Paralel worker sayısı (varsayılan: 1, sıralı)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Sadece mevcut dataset'i doğrula, işleme yapma"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Mevcut .npy dosyalarını yeniden işle"
    )

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
