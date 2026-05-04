"""MediaPipe Holistic ile video keypoint extraction. Her video → (64, 258) .npy dosyası."""
import argparse
import logging
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET_FRAMES = 64
INPUT_DIM = 258  # 33*4 + 21*3 + 21*3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Modül-level global: her worker process kendi holistic örneğini tutar
_holistic = None


def _init_worker(model_complexity: int, min_conf: float) -> None:
    """Her worker process başladığında MediaPipe Holistic başlatır."""
    global _holistic
    import mediapipe as mp_lib
    _holistic = mp_lib.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )


def _extract_frame_keypoints(results) -> np.ndarray:
    """Bir frame'in MediaPipe sonuçlarından 258-boyutlu vektör çıkarır."""
    vec = np.zeros(INPUT_DIM, dtype=np.float32)

    # Pose: 33 × 4 = 132
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            vec[i * 4: i * 4 + 4] = [lm.x, lm.y, lm.z, lm.visibility]

    # Sol el: 21 × 3 = 63  (offset 132)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            base = 132 + i * 3
            vec[base: base + 3] = [lm.x, lm.y, lm.z]

    # Sağ el: 21 × 3 = 63  (offset 195)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            base = 195 + i * 3
            vec[base: base + 3] = [lm.x, lm.y, lm.z]

    return vec


def _process_video(task: tuple) -> tuple[str, bool, str]:
    """Tek bir videoyu işler. (sample_id, başarılı_mı, hata_mesajı) döner."""
    import cv2

    sample_id, video_path, out_path, target_frames = task

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return sample_id, False, f"Video açılamadı: {video_path}"

    frames_raw: list[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if len(frames_raw) == 0:
        return sample_id, False, f"Frame okunamadı: {video_path}"

    # Frame sayısı normalizasyonu
    n = len(frames_raw)
    if n >= target_frames:
        indices = np.linspace(0, n - 1, target_frames, dtype=int)
        selected = [frames_raw[i] for i in indices]
    else:
        selected = frames_raw  # padding ile tamamlanacak

    keypoints = np.zeros((target_frames, INPUT_DIM), dtype=np.float32)
    for t, frame_rgb in enumerate(selected):
        results = _holistic.process(frame_rgb)
        keypoints[t] = _extract_frame_keypoints(results)
    # Kısa videolarda kalan satırlar sıfır kalır (padding)

    np.save(str(out_path), keypoints)
    return sample_id, True, ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MediaPipe keypoint extraction")
    p.add_argument(
        "--data-dir",
        default="/mnt/c/Users/Atakan/OneDrive/Desktop/Sign_Language_Data",
        help="Ham veri dizini (video alt klasörü ve etiket CSV'si burada)",
    )
    p.add_argument(
        "--video-subdir",
        default="train",
        help="Video dosyalarının bulunduğu alt klasör (train, val veya test)",
    )
    p.add_argument(
        "--labels-csv",
        default="data/train_labels_full.csv",
        help="Örnek ID ve sınıf etiketlerini içeren CSV",
    )
    p.add_argument("--out-dir", default="keypoints_full", help="Keypoint çıktı dizini")
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() - 1),
        help="Paralel worker sayısı",
    )
    p.add_argument("--target-frames", type=int, default=TARGET_FRAMES)
    p.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe model karmaşıklığı (0=hızlı, 2=doğru)",
    )
    p.add_argument("--min-conf", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    labels_csv = Path(args.labels_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Etiket CSV bulunamadı: {labels_csv}\n"
            "Önce '01_select_top_classes.py' çalıştırın."
        )

    train_dir = data_dir / args.video_subdir
    if not train_dir.exists():
        raise FileNotFoundError(f"Video dizini bulunamadı: {train_dir}")

    df = pd.read_csv(str(labels_csv), header=None, names=["sample_id", "class_id"])
    sample_ids = df["sample_id"].tolist()
    log.info("CSV'den %d örnek okundu.", len(sample_ids))

    # Dizinleri bir kez tara → set ile O(1) varlık kontrolü (NTFS/WSL üzerinde kritik)
    log.info("Video dizini taranıyor: %s", train_dir)
    available_videos: set[str] = {p.stem for p in train_dir.iterdir() if p.suffix == ".mp4"}
    log.info("Dizinde %d .mp4 dosyası bulundu.", len(available_videos))

    log.info("Keypoint dizini taranıyor: %s", out_dir)
    done_keypoints: set[str] = {p.stem for p in out_dir.iterdir() if p.suffix == ".npy"}
    log.info("Mevcut keypoint: %d", len(done_keypoints))

    tasks: list[tuple] = []
    skipped = 0
    missing_video = 0

    for sid in sample_ids:
        if sid in done_keypoints:
            skipped += 1
            continue
        video_stem = f"{sid}_color"
        if video_stem not in available_videos:
            missing_video += 1
            continue
        video_path = train_dir / f"{sid}_color.mp4"
        out_path = out_dir / f"{sid}.npy"
        tasks.append((sid, video_path, out_path, args.target_frames))

    log.info(
        "İşlenecek: %d | Zaten mevcut (atlandı): %d | Video eksik: %d",
        len(tasks),
        skipped,
        missing_video,
    )

    if not tasks:
        log.info("Tüm keypoint'ler zaten mevcut. Çıkılıyor.")
        return

    log.info("%d worker ile başlatılıyor (model_complexity=%d)...", args.workers, args.model_complexity)

    errors: list[str] = []
    success = 0

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.model_complexity, args.min_conf),
    ) as pool:
        with tqdm(total=len(tasks), unit="video", dynamic_ncols=True) as pbar:
            for sid, ok, msg in pool.imap_unordered(_process_video, tasks, chunksize=4):
                if ok:
                    success += 1
                else:
                    errors.append(f"{sid}: {msg}")
                    log.warning("HATA — %s: %s", sid, msg)
                pbar.update()

    log.info("Tamamlandı: %d başarılı, %d hata.", success, len(errors))
    if errors:
        err_log = out_dir / "extraction_errors.txt"
        err_log.write_text("\n".join(errors), encoding="utf-8")
        log.info("Hata listesi kaydedildi: %s", err_log)


if __name__ == "__main__":
    main()
