# AI-Powered Sign Language Interpreter — Proje Bağlamı

## Proje Özeti
Türk İşaret Dili (TSL) hareketlerini gerçek zamanlı olarak metne çeviren masaüstü uygulama.
Dil: Python 3.13 | Mimari: MediaPipe Tasks API → LSTM model → FastAPI → PyQt5 UI

## Ekip
- **Atakan Yılmaz** — Scrum Master, **AI & Data** (sen)
- Elif Nur Günay — Backend
- Sevda Tuba Ehlibeyt — Frontend
- Zeynep Ötegen — Frontend
- Zehra Kelahmetoğlu — Performance & Testing

## Klasör Yapısı
```
TSL_Project/
├── data/                  # .npy sekans dosyaları (30 frame × 63 koordinat)
│   ├── MERHABA/0.npy ...
│   └── ...               # 255 sınıf, ~291 sekans (YouTube'dan)
├── data_collection/
│   ├── collect_data.py    # Webcam'den veri toplama (MediaPipe Tasks API)
│   ├── youtube_dataset_builder.py  # YouTube kanalından otomatik veri
│   ├── verify_data.py     # Dataset doğrulama raporu
│   └── _test_import.py    # Kurulum testi
├── models/
│   └── hand_landmarker.task  # MediaPipe model dosyası (~25MB)
├── venv/                  # Python sanal ortamı
└── requirements.txt
```

## Kritik Teknik Notlar

### MediaPipe (ÇOK ÖNEMLİ)
- mediapipe v0.10.32 + Python 3.13 → `mp.solutions` ve `mediapipe.python` **KALDIRILDI**
- Sadece **Tasks API** kullanılmalı:
  ```python
  import mediapipe as mp
  HandLandmarker = mp.tasks.vision.HandLandmarker
  HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode
  # detect_for_video(mp_img, timestamp_ms) — VIDEO modunda monoton ts zorunlu
  ```
- Model dosyası: `models/hand_landmarker.task` (ilk çalıştırmada otomatik indir)

### Veri Formatı
- Her sekans: `numpy array shape=(30, 63), dtype=float32`
- 30 frame × 21 landmark × 3 koordinat (x, y, z)
- **Normalizasyon**: tüm koordinatlar wrist (index=0) noktasına göre bağıl

### Dataset Durumu (Mart 2026)
- 255 sınıf, 291 sekans — YouTube kanalından: https://www.youtube.com/@isaretdiliegitimi5504
- Sorun: çoğu sınıf 1 sekans → model eğitimi için yetersiz
- **Sonraki adım**: collect_data.py ile webcam'den ek veri topla (min 30 seq/sınıf)

## Şu Anki Hafta: Hafta 3 — Core Coding
Yol haritasına göre yapılacaklar:
- [ ] Model mimarisi seç (LSTM önerilen — landmark sekansları için ideal)
- [ ] train.py yaz (PyTorch LSTM)
- [ ] İlk eğitimi çalıştır
- [ ] mAP hesapla

## Model Mimarisi Önerisi
```
Input: (batch, 30, 63)  ← 30 frame, 63 koordinat
→ LSTM(hidden=128, layers=2, dropout=0.3)
→ FC(128 → num_classes)
→ Softmax
```
Alternatif: YOLOv8 (bounding box tabanlı, daha ağır)

## Önemli Kararlar (Bu Konuşmadan)
1. YOLOv8 yerine **MediaPipe + LSTM** seçildi (daha hafif, real-time için uygun)
2. YouTube dataset builder yazıldı — `youtube_dataset_builder.py`
3. Veri az → hybrid yaklaşım: YouTube + webcam toplama
4. Hedef kelimeler: MERHABA, TESEKKUR, EVET, HAYIR, LUTFEN, TAMAM, NASIL_SINIZ, ISIM + daha fazlası

## Stack
```
torch, torchvision     # Model eğitimi
mediapipe==0.10.32     # El landmark tespiti (Tasks API)
opencv-python          # Kamera
numpy, pandas          # Veri işleme
scikit-learn           # Metrikler, K-Fold
matplotlib, seaborn    # Görselleştirme
fastapi, uvicorn       # API
PyQt5                  # Desktop UI
onnx, onnxruntime      # Model optimizasyonu
sqlite3                # Log kayıtları
```

## Sık Kullanılan Komutlar
```bash
# Venv aktifleştir
venv\Scripts\activate

# Kurulum testi
python data_collection/_test_import.py

# Webcam'den veri topla
python data_collection/collect_data.py

# YouTube'dan veri çek
python data_collection/youtube_dataset_builder.py --limit 5

# Dataset doğrula
python data_collection/verify_data.py
```
