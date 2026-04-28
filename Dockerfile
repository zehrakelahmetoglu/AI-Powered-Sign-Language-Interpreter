# 1. Adım: Python 3.10-slim kullanarak daha güncel ve hafif bir temel seçiyoruz
FROM python:3.10-slim

# 2. Adım: Çalışma klasörünü belirle
WORKDIR /app

# 3. Adım: OpenCV'nin (cv2) çalışması için ŞART olan sistem kütüphanelerini kur
# (Buradaki libgl1-mesa-glx ve libglib2.0-0 hataları önlemek için çok önemlidir)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Adım: Önce sadece requirements.txt kopyala (Bu adım hız kazandırır)
COPY requirements.txt .

# 5. Adım: Kütüphaneleri yükle
RUN pip install --no-cache-dir -r requirements.txt

# 6. Adım: Tüm proje dosyalarını (main.py, models/ vb.) içeri kopyala
COPY . .

# 7. Adım: Dış dünya ile iletişim portu
EXPOSE 8000

# 8. Adım: Uygulamayı başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
