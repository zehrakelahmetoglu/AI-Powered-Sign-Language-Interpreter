"""
AI-Powered Sign Language Interpreter
=====================================
Module  : YouTube Dataset Builder
Author  : Atakan Yılmaz (AI & Data)
Version : 1.0.0

Açıklama:
    @isaretdiliegitimi5504 YouTube kanalından TSL videolarını indirir,
    MediaPipe Tasks API ile landmark çıkarır ve .npy formatında kaydeder.
    Çıktı formatı collect_data.py ile aynıdır → direkt model eğitimine girer.

Çıktı Yapısı:
    data/
    ├── BILMEK/
    │   ├── 0.npy   shape=(30, 63)  ← her sekans
    │   └── 1.npy
    ├── CALISMA/
    │   └── ...
    └── ...

Kullanım:
    python youtube_dataset_builder.py               # tüm kanal
    python youtube_dataset_builder.py --limit 20    # ilk 20 video
    python youtube_dataset_builder.py --cats alfabe sayilar
    python youtube_dataset_builder.py --verify      # sadece doğrulama raporu
"""

import os
import re
import sys
import json
import time
import shutil
import argparse
import subprocess
import unicodedata
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

# ─────────────────────────────────────────────────────────────
#  YT-DLP EXECUTABLE — venv vs sistem Python sorunu
# ─────────────────────────────────────────────────────────────

def _find_ytdlp_cmd() -> list:
    """
    yt-dlp'yi bul: venv içinde olmayabilir, sistem Python da denenir.
    """
    # 1. PATH'te yt-dlp.exe / yt-dlp var mı?
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]

    # 2. Python executable'lar ile -m yt_dlp dene
    candidates = [sys.executable, "python", "python3", "py"]
    for py in candidates:
        if py != sys.executable and not shutil.which(py):
            continue
        try:
            r = subprocess.run(
                [py, "-m", "yt_dlp", "--version"],
                capture_output=True, timeout=10
            )
            if r.returncode == 0:
                return [py, "-m", "yt_dlp"]
        except Exception:
            continue

    # 3. Windows Store Python scripts klasörü (sabit yol)
    localappdata = os.environ.get("LOCALAPPDATA", "")
    store_scripts = Path(localappdata) / "Packages"
    if store_scripts.exists():
        for pkg in store_scripts.iterdir():
            if "Python" in pkg.name:
                yt = pkg / "LocalCache" / "local-packages" / \
                     "Python313" / "Scripts" / "yt-dlp.exe"
                if yt.is_file():
                    return [str(yt)]

    raise RuntimeError(
        "[HATA] yt-dlp bulunamadi!\n"
        "Cozum: pip install yt-dlp"
    )

_YTDLP_CMD: list = []   # lazy init

def _ytdlp() -> list:
    global _YTDLP_CMD
    if not _YTDLP_CMD:
        _YTDLP_CMD = _find_ytdlp_cmd()
        print(f"[YT-DLP] {' '.join(_YTDLP_CMD)}")
    return _YTDLP_CMD

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

CHANNEL_URL     = "https://www.youtube.com/@isaretdiliegitimi5504/videos"
PROJECT_ROOT    = Path(__file__).resolve().parent
DATA_DIR        = PROJECT_ROOT / "data"
MODEL_DIR       = PROJECT_ROOT / "models"
TMP_DIR         = PROJECT_ROOT / "tmp_videos"
MODEL_PATH      = MODEL_DIR / "hand_landmarker.task"
MODEL_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

FRAMES_PER_SEQ  = 30        # collect_data.py ile aynı
MIN_HAND_CONF   = 0.55      # daha düşük = daha fazla frame geçer
MIN_TRACK_CONF  = 0.50
MIN_FRAMES_KEEP = 15        # sekans kabul için min anlamlı frame
TARGET_FPS      = 15        # video FPS'i bu değere normalize et (hız)

# ─────────────────────────────────────────────────────────────
#  TÜRKÇE KARAKTERLERİ NORMALİZE ET → klasör adı
# ─────────────────────────────────────────────────────────────

_TR_MAP = str.maketrans("çğışöüÇĞİŞÖÜ", "cgisouCGISOUcgisouCGISOU"[: 12])

# Derleme/özet videoları atla (tek sınıf değil, birden fazla içerik)
_SKIP_KEYWORDS = {"tumu", "tum", "hepsi", "ozet", "fenerbahce",
                  "yiyecekler", "iller", "alfabetumu", "iceceklertumu",
                  "icecekler_tumu", "biz_siz_o_tumu"}

def _should_skip(name: str) -> bool:
    """Derleme/özet videoları atla."""
    n = name.lower().replace("_", "")
    if n in _SKIP_KEYWORDS:
        return True
    # "_TUMU" ile biten derleme videoları (İçecekler Tümü, Alfabe Tümü vb.)
    if name.endswith("_TUMU") or name.endswith("TUMU"):
        return True
    return False

# "İşaret Dili/Dİli Eğitimi" pattern
_ISARET_PAT = re.compile(
    r"[Ii][sşŞS]aret\s+[Dd][iİIı][lL][iİIı]\s+[Ee][gğĞG][iİIı]timi",
    re.IGNORECASE
)

# "Sayılar X(kelime)" → kelimeyi çıkar: "Sayılar 14(ondört)" → "ONDORT"
_SAYILAR_PAT = re.compile(r"[Ss]ay[ıi]lar\s+[\d.]+\(?([^)]*)\)?")


def normalize_classname(raw: str) -> str:
    """
    Her iki başlık formatını doğru şekilde ayrıştırır:

    Format A: "Bilmek -- İşaret Dili Eğitimi"           → BILMEK
    Format B: "İşaret Dili Eğitimi -- bana"             → BANA
    Format C: "İşaret Dili Eğitimi -- Sayılar 10(on)"   → ON
    Format D: "İşaret Dİli Eğitimi -- sen -- İşaret..." → SEN
    Format E: "Boş 2 -- İşaret Dili Eğitimi"            → BOS
    """
    t = raw.strip()

    # Birden fazla "--" içeren başlıkları temizle (Format D)
    parts = [p.strip() for p in t.split("--")]

    # "İşaret Dili Eğitimi" olan parçaları çıkar
    word_parts = [p for p in parts if not _ISARET_PAT.fullmatch(p.strip())]

    if not word_parts:
        return ""  # Tamamen kanal adından ibaret → atla

    # İlk anlamlı parçayı al
    name = word_parts[0].strip()

    # "Sayılar 10(on)" → "on", "Sayılar 14(ondort)" → "ondort"
    m = _SAYILAR_PAT.match(name)
    if m:
        inner = m.group(1).strip()
        name = inner if inner else re.sub(r"[^0-9]", "", name)

    # Varyant numarasını sil: "Boş 2" → "Boş", "siz2" → "siz"
    name = re.sub(r"\s+\d+$", "", name).strip()
    name = re.sub(r"(\D)\d+$", r"\1", name).strip()  # "siz2" → "siz"

    # TR karakterleri ASCII'ye çevir
    name = name.translate(_TR_MAP)

    # NFKD → combining karakterleri düşür
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")

    # Sadece alfanumerik + boşluk → boşlukları _ yap → büyüt
    name = re.sub(r"[^a-zA-Z0-9\s_]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    name = name.upper()

    # Derleme/atlanacak videoları filtrele
    if _should_skip(name) or not name:
        return ""

    return name


# ─────────────────────────────────────────────────────────────
#  MODEL İNDİR
# ─────────────────────────────────────────────────────────────

def ensure_model() -> None:
    if MODEL_PATH.is_file():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] hand_landmarker.task indiriliyor (~25 MB)...")

    def _progress(block, bsize, total):
        pct = min(100, block * bsize * 100 // total)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] %{pct}", end="", flush=True)

    urllib.request.urlretrieve(str(MODEL_URL), str(MODEL_PATH), reporthook=_progress)
    print("\n[MODEL] Tamamlandı.")


# ─────────────────────────────────────────────────────────────
#  CHANNEL VİDEO LİSTESİ — yt-dlp
# ─────────────────────────────────────────────────────────────

def fetch_video_list() -> list[dict]:
    """
    Kanalın tüm video metadata'sını yt-dlp ile çek.
    stdout encoding sorunlarını önlemek için temp dosya kullanır.
    """
    print(f"[YT] Video listesi alınıyor: {CHANNEL_URL}")

    tmp_json = Path(os.environ.get("TEMP", "/tmp")) / "yt_tsl_playlist.json"

    # yt-dlp çıktısını doğrudan temp dosyaya yönlendir
    with open(str(tmp_json), "w", encoding="utf-8") as fout:
        result = subprocess.run(
            _ytdlp() + [
             "--flat-playlist", "--dump-json",
             "--no-warnings", "--no-progress",
             CHANNEL_URL],
            stdout=fout,
            stderr=subprocess.PIPE,
            timeout=120,
        )

    # Stderr'den hata kontrolü
    if result.returncode != 0:
        stderr_txt = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        print(f"[WARN] yt-dlp returncode={result.returncode}")
        if stderr_txt.strip():
            print(f"[WARN] yt-dlp stderr:\n{stderr_txt[:500]}")

    # Temp dosyayı oku
    videos = []
    try:
        with open(str(tmp_json), "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    v = json.loads(line)
                    videos.append({
                        "title"   : v.get("title", ""),
                        "id"      : v.get("id", ""),
                        "duration": v.get("duration", 0) or 0,
                        "url"     : f"https://www.youtube.com/watch?v={v.get('id','')}",
                    })
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print("[HATA] yt-dlp çıktı dosyası oluşturulamadı.")
    finally:
        try:
            tmp_json.unlink()
        except Exception:
            pass

    if not videos:
        print("[HATA] Video listesi boş geldi. yt-dlp kurulu mu?")
        print(f"       Test komutu: python -m yt_dlp --flat-playlist --dump-json \"{CHANNEL_URL}\"")

    print(f"[YT] {len(videos)} video bulundu.")
    return videos


def download_video(video_id: str, out_path: Path) -> bool:
    """Tek bir videoyu 480p mp4 olarak indir. Başarıysa True."""
    try:
        result = subprocess.run(
            _ytdlp() + [
             "-f", "mp4[height<=480]/best[height<=480]/best",
             "--no-warnings", "--quiet",
             "--no-progress",
             "-o", str(out_path),
             f"https://www.youtube.com/watch?v={video_id}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=90,
        )
        # yt-dlp bazen .mp4 yerine farklı uzantı ekler, kontrol et
        if out_path.is_file():
            return True
        # Olası uzantı varyasyonlarını dene
        for ext in [".mp4", ".webm", ".mkv"]:
            alt = out_path.with_suffix(ext)
            if alt.is_file():
                alt.rename(out_path)
                return True
        return False
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Zaman aşımı ({video_id})")
        return False
    except Exception as exc:
        print(f"  [WARN] İndirme hatası ({video_id}): {exc}")
        return False


# ─────────────────────────────────────────────────────────────
#  LANDMARK ÇIKARIMI
# ─────────────────────────────────────────────────────────────

def extract_landmarks_from_video(
    video_path: Path,
    landmarker,
    target_fps: int = TARGET_FPS,
) -> list[np.ndarray]:
    """
    Video dosyasından landmark dizilerini çıkar.

    Strateji:
    - Frame'leri TARGET_FPS'e downsample ederek işle (hız için)
    - Her FRAMES_PER_SEQ frame → 1 sekans
    - El bulunamayan frame → sıfır vektörü (padding)
    - MIN_FRAMES_KEEP'ten az anlamlı frame içeren sekanslar atılır

    Dönüş:
        List[np.ndarray]  her eleman shape=(30, 63)
    """
    MpImage       = mp.Image
    MpImageFormat = mp.ImageFormat

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    orig_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    skip      = max(1, int(round(orig_fps / target_fps)))  # kaçta bir frame al

    all_vectors: list[np.ndarray] = []
    frame_idx  = 0
    ts_ms      = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % skip != 0:
            continue

        ts_ms += int(1000 / target_fps)

        # BGR → RGB → MpImage
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = MpImage(image_format=MpImageFormat.SRGB, data=rgb)

        try:
            result  = landmarker.detect_for_video(mp_img, ts_ms)
            vec     = _landmarks_to_vector(result)
        except Exception:
            vec = np.zeros(63, dtype=np.float32)

        all_vectors.append(vec)

    cap.release()

    # Sekans böl
    sequences = _split_into_sequences(all_vectors)
    return sequences


def _landmarks_to_vector(detection_result) -> np.ndarray:
    """Tasks API sonucundan wrist-relative 63-d vektör çıkar."""
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


def _split_into_sequences(vectors: list[np.ndarray]) -> list[np.ndarray]:
    """Frame vektörlerini FRAMES_PER_SEQ uzunluğunda sekasanlara böl."""
    n   = len(vectors)
    if n == 0:
        return []

    sequences = []

    # Tam sekanslar
    for start in range(0, n - FRAMES_PER_SEQ + 1, FRAMES_PER_SEQ):
        chunk = vectors[start: start + FRAMES_PER_SEQ]
        nonzero = sum(1 for v in chunk if np.any(v != 0))
        if nonzero >= MIN_FRAMES_KEEP:
            sequences.append(np.array(chunk, dtype=np.float32))

    # Kısa video: tek sekans için padding uygula
    if not sequences and n >= MIN_FRAMES_KEEP:
        chunk = vectors.copy()
        # Son frame'i tekrarla → 30 frame'e tamamla
        while len(chunk) < FRAMES_PER_SEQ:
            chunk.append(chunk[-1])
        sequences.append(np.array(chunk[:FRAMES_PER_SEQ], dtype=np.float32))

    return sequences


# ─────────────────────────────────────────────────────────────
#  DOĞRULAMA RAPORU
# ─────────────────────────────────────────────────────────────

def print_verification_report() -> None:
    print("\n" + "=" * 65)
    print("  Dataset Doğrulama Raporu")
    print("=" * 65)

    if not DATA_DIR.is_dir():
        print(f"[HATA] '{DATA_DIR}' bulunamadı. Önce builder'ı çalıştır.")
        return

    classes = sorted(DATA_DIR.iterdir())
    total   = 0
    print(f"  {'Sınıf':<28} {'Sekans':>8}  {'Shape':>12}  Durum")
    print("  " + "-" * 60)

    for cls_dir in classes:
        if not cls_dir.is_dir():
            continue
        npys = sorted(cls_dir.glob("*.npy"))
        if not npys:
            continue
        try:
            sample = np.load(str(npys[0]))
            shape  = str(sample.shape)
            ok     = "OK" if sample.shape == (FRAMES_PER_SEQ, 63) else "SHAPE HATASI"
        except Exception as exc:
            shape = str(exc)
            ok    = "BOZUK"

        total += len(npys)
        status_icon = "✓" if ok == "OK" else "✗"
        print(f"  {status_icon} {cls_dir.name:<26} {len(npys):>8}  {shape:>12}  {ok}")

    print("  " + "-" * 60)
    print(f"  TOPLAM: {len(classes)} sınıf, {total} sekans\n")


# ─────────────────────────────────────────────────────────────
#  KATEGORİ FİLTRESİ
# ─────────────────────────────────────────────────────────────

_CAT_KEYWORDS = {
    "alfabe"  : ["Harfi", "Alfabe"],
    "sayilar" : ["Sayılar", "Sayılar"],
    "icecekler": ["Çay", "Kahve", "Su ", "Soda", "Bira", "Rakı", "Şarap",
                   "Kola", "Ayran", "Süt", "Gazoz", "İçecekler", "Limonata",
                   "Nescafe", "Oralet", "Sahlep", "Votka", "Şampanya", "Şalgam",
                   "Çorba", "Meyve Suyu", "Sıkma", "Soğuk Çay", "Sıcak Çikolata",
                   "İçki"],
    "aylar"   : ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
                  "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"],
    "aile"    : ["Anne", "Baba", "Abla", "Abi", "Kardeş", "Dede", "Nine",
                  "Aile", "Torun", "Komşu", "Kuzen", "Damat", "Gelin", "Koca",
                  "Çocuk", "Bebek", "Kız ", "Erkek", "Kadın", "Bayan", "Bay ",
                  "Adam", "Oğul", "Yiğen", "Teyze", "Amca", "Hala", "Dayı",
                  "Arkadaş", "Akraba", "Ahbap", "Misafir", "Sevgili", "Dost",
                  "Genç", "Yaşlı", "İhtiyar"],
    "zamirler": ["-- ben", "-- sen", "-- o ", "-- biz", "-- siz", "-- onlar",
                  "-- bana", "-- sana", "-- onun", "-- bizim", "-- senin",
                  "-- şu ", "-- onların"],
}


def filter_by_category(videos: list[dict], cats: list[str]) -> list[dict]:
    cats_lower = [c.lower() for c in cats]
    result     = []
    for v in videos:
        t = v["title"]
        for cat in cats_lower:
            if cat == "fiiller":
                matched = any(
                    any(kw in t for kw in kws)
                    for kws in _CAT_KEYWORDS.values()
                )
                if not matched:
                    result.append(v)
                    break
            elif cat in _CAT_KEYWORDS:
                if any(kw in t for kw in _CAT_KEYWORDS[cat]):
                    result.append(v)
                    break
    return result


# ─────────────────────────────────────────────────────────────
#  ANA PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    limit     : Optional[int]   = None,
    categories: Optional[list]  = None,
    skip_existing: bool          = True,
) -> None:
    print("=" * 65)
    print("  YouTube TSL Dataset Builder")
    print("=" * 65)
    print(f"  mediapipe  : v{mp.__version__}")
    print(f"  Çıktı dizin: {DATA_DIR}")
    print(f"  FPS hedef  : {TARGET_FPS} | Seq uzun: {FRAMES_PER_SEQ}")
    print("=" * 65)

    # Model
    ensure_model()

    # Video listesi
    videos = fetch_video_list()
    if categories:
        videos = filter_by_category(videos, categories)
        print(f"[FİLTRE] Kategori={categories} → {len(videos)} video")
    if limit:
        videos = videos[:limit]
        print(f"[LİMİT]  İlk {limit} video işlenecek")

    # Klasörler
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # MediaPipe Landmarker
    BaseOptions        = mp.tasks.BaseOptions
    HandLandmarker     = mp.tasks.vision.HandLandmarker
    HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode

    opts = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH.resolve())),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=MIN_HAND_CONF,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    total_saved  = 0
    total_skip   = 0
    total_fail   = 0
    class_counts: dict[str, int] = {}

    with HandLandmarker.create_from_options(opts) as landmarker:
        for i, video in enumerate(videos):
            class_name = normalize_classname(video["title"])
            if not class_name:
                total_skip += 1
                continue

            class_dir = DATA_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Mevcut sekans sayısı
            existing = list(class_dir.glob("*.npy"))

            if skip_existing and len(existing) >= 3:
                print(f"  [{i+1:3}/{len(videos)}] {class_name:<28} ATLA (mevcut: {len(existing)} sek)")
                total_skip += 1
                continue

            # Video indir
            vid_path = TMP_DIR / f"{video['id']}.mp4"
            print(f"  [{i+1:3}/{len(videos)}] {class_name:<28} İndiriliyor...", end="", flush=True)

            if not vid_path.is_file():
                ok = download_video(video["id"], vid_path)
                if not ok:
                    print(" BAŞARISIZ")
                    total_fail += 1
                    continue

            # Landmark çıkar
            sequences = extract_landmarks_from_video(vid_path, landmarker)

            # Temizle (RAM & disk)
            try:
                vid_path.unlink()
            except Exception:
                pass

            if not sequences:
                print(f" {len(existing)} sek mevcut | 0 yeni (el bulunamadı)")
                total_fail += 1
                continue

            # Kaydet
            start_idx = len(list(class_dir.glob("*.npy")))
            for j, seq in enumerate(sequences):
                save_path = class_dir / f"{start_idx + j}.npy"
                np.save(str(save_path), seq)

            saved = len(sequences)
            total_saved += saved
            class_counts[class_name] = class_counts.get(class_name, 0) + saved
            dur = int(video["duration"])
            print(f" +{saved} sek  [{dur}s video]  toplam={start_idx+saved}")

    # Geçici klasörü temizle
    if TMP_DIR.exists():
        shutil.rmtree(str(TMP_DIR), ignore_errors=True)

    # Özet
    print("\n" + "=" * 65)
    print(f"  TAMAMLANDI")
    print(f"  Kaydedilen : {total_saved} sekans")
    print(f"  Atlanan    : {total_skip} video (mevcut)")
    print(f"  Başarısız  : {total_fail} video")
    print(f"  Konum      : {DATA_DIR.resolve()}")
    print("=" * 65)
    print("\n  En çok sekans toplanan sınıflar:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * cnt + "░" * max(0, 10 - cnt)
        print(f"  {cls:<28} {bar} {cnt}")

    print_verification_report()


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube TSL Dataset Builder",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="İşlenecek max video sayısı (test için)"
    )
    parser.add_argument(
        "--cats", nargs="+",
        choices=["alfabe", "sayilar", "icecekler", "aylar", "aile", "zamirler", "fiiller"],
        metavar="KATEGORİ",
        help="Sadece belirli kategorileri işle\n"
             "Seçenekler: alfabe sayilar icecekler aylar aile zamirler fiiller"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Sadece mevcut dataset'i doğrula, indirme yapma"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Mevcut sekansları yeniden indir (üstüne ekle)"
    )
    args = parser.parse_args()

    if args.verify:
        print_verification_report()
        return

    run_pipeline(
        limit          = args.limit,
        categories     = args.cats,
        skip_existing  = not args.no_skip,
    )


if __name__ == "__main__":
    main()
