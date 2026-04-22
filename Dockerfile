# 1. Adım: Temel işletim sistemi olarak Python 3.9 kullan
FROM python:3.9-slim

# 2. Adım: Docker'ın içindeki çalışma klasörünü belirle
WORKDIR /app

# 3. Adım: Sistem güncellemelerini ve OpenCV için gereken kütüphaneleri kur
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Adım: Alışveriş listemizi (requirements.txt) içeri kopyala
COPY requirements.txt .

# 5. Adım: Listedeki kütüphaneleri yükle
RUN pip install --no-cache-dir -r requirements.txt

# 6. Adım: Kalan tüm kodlarını (main.py vb.) içeri kopyala
COPY . .

# 7. Adım: Uygulamayı dış dünyaya aç (8000 portu)
EXPOSE 8000

# 8. Adım: Sunucuyu başlat!
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]