# AI-Powered Sign Language Interpreter (Local Dataset Pipeline)

Bu depo, İşaret Dili Tercümanı projesi için **yerel video veri setlerinin işlenmesi** ve **LSTM model eğitimi** süreçlerine odaklanmaktadır.

## Kapsam

- **Yerel Video İşleme**: ChaLearn/Local dataset videolarından MediaPipe landmark dizileri çıkarımı.
- **LSTM Model Eğitimi**: RTX 4060 için optimize edilmiş, mixed-precision destekli eğitim pipeline'ı.
- **Webcam Veri Toplama**: MediaPipe ile anlık webcam üzerinden veri seti oluşturma.

## Repo Yapısı

- `local_dataset_processor.py`: Yerel `_color.mp4` videolarını tam 30 frame'e resample eder ve `(30, 63)` landmark dizileri üretir.
- `train_lstm.py`: RTX 4060 için optimize edilmiş (mixed precision, cuDNN LSTM) Keras eğitim scripti.
- `collect_data.py`: Webcam üzerinden elle veri toplamak için kullanılır.
- `_test_import.py`: MediaPipe ve OpenCV ortam kontrolü.
- `data/`: `.npy` formatındaki landmark veri setlerinin saklandığı kök dizin.
- `models/`: MediaPipe `.task` modelleri ve eğitilen Keras modellerinin saklandığı dizin.

## Kurulum

1. Gereksinimleri yükleyin:

```bash
pip install -r requirements.txt
# Ek olarak TensorFlow (CUDA destekli) önerilir:
pip install tensorflow[and-cuda]
```

2. Ortamı test edin:

```bash
python _test_import.py
```

## Kullanım

### 1. Yerel Videoları İşleme
Harici diskteki veya yerel dizindeki videoları (ChaLearn formatında) `.npy` dosyalarına dönüştürmek için:

```bash
python local_dataset_processor.py --source "/path/to/videos" --workers 4
```

### 2. LSTM Model Eğitimi
İşlenmiş verilerle RTX 4060 üzerinde eğitim başlatmak için:

```bash
python train_lstm.py --epochs 100 --batch-size 64
```

### 3. Webcam ile Veri Toplama
```bash
python collect_data.py
```

## Veri Formatı

- **Dosya Tipi**: `.npy`
- **Boyut (Shape)**: `(30, 63)` (30 frame x 21 landmark x 3 koordinat)
- **Normalizasyon**: Bilek (wrist) noktasına göre bağıl koordinatlar.

## Ekip

- Zehra Kelahmetoglu - Frontend
- Atakan Yilmaz - Data
- Zeynep Otegen - Optimization and Documentation
- Elifnur Gunay - Test and Maintenance
- Sevda Tuba Ehlibeyt - Backend

## Lisans
MIT
