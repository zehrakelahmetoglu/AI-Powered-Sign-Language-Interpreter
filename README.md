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

3-Install the required libraries: ```pip install -r requirements.txt```

4-Run the application: ```python main.py```

### Usage

1- Be sure you are on the project file path

2- Type ```python main.py```  on the terminal and press enter

3- You can see the split-screen with your view

4- You can start the program with pressing "Kamera Başlat" button  

5- Press 'Durdur' button to quit the program

### Application Interface

<img width="1887" height="875" alt="Ekran görüntüsü 2026-04-28 223224" src="https://github.com/user-attachments/assets/47ae8791-388d-4fa5-9d3e-9bd74995ba39" />
<img width="1881" height="864" alt="Ekran görüntüsü 2026-04-28 223200" src="https://github.com/user-attachments/assets/20aaf309-5cdf-4e40-9e46-bf0f1897d81e" />


### Contribution Guidelines

1-Fork the repository.

2-Create a new branch for your feature: ``` git checkout -b feature/NewFeature ```

3-Commit your changes and open a Pull Request.

### License 

MIT License
For more information , please check the LICENSE document

### Credits/Team

-Zehra Kelahmetoğlu - Frontend

-Atakan Yılmaz - Data

-Zeynep Ötegen - Data and Documentation

-Elifnur Günay -Test and Maintenance

-Sevda Tuba Ehlibeyt - Backend
