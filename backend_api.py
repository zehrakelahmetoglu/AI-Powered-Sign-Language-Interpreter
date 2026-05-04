from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2 
import numpy as np
import base64
import os
import time
import sqlite3
import random

app = FastAPI(title="İşaret Dili Çevirici - Hibrit Backend")

# --- 🛰️ CORS AYARLARI (Arayüz ile iletişimi sağlar) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class ImageRequest(BaseModel):
    image_data: str

class TestRequest(BaseModel):
    oturum_id: str

@app.get("/")
def home():
    return {"mesaj": "İşaret Dili API Çalışıyor - Video ve Fotoğraf Hazır!"}

# --- 🛠️ TEST ENTEGRASYON KÖPRÜSÜ (Arayüzdeki test butonu buraya istek atacak) ---
@app.post("/api/tahmin_yap")
def tahmin_yap_ve_kaydet(request: TestRequest):
    """Model gelene kadar arayüz ve veritabanı testlerini yapmamızı sağlar."""
    sahte_kelimeler = ["MERHABA", "HASTA", "AĞRI", "VAR", "YOK", "DOKTOR", "İLAÇ"]
    tahmin = random.choice(sahte_kelimeler)
    guven_skoru = round(random.uniform(85.0, 99.9), 2)
    gecikme_ms = round(random.uniform(20.0, 50.0), 2)
    
    # Senin veritabanına loglama işlemi!
    try:
        conn = sqlite3.connect("signspeak_live.sqlite3")
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Hasta_Ceviri_Loglari (oturum_id, modelin_tahmini, guven_skoru, gecikme_ms)
            VALUES (?, ?, ?, ?)
        ''', (request.oturum_id, tahmin, guven_skoru, gecikme_ms))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Veritabanı Hatası: {e}")

    return {"tahmin": tahmin, "guven": guven_skoru, "type": "TEST"}


# 📸 FOTOĞRAF KISMI (Canlı Kamera Akışı İçin)
@app.post("/tahmin-et")
async def predict_image(request: ImageRequest):
    try:
        header, encoded = request.image_data.split(",", 1) if "," in request.image_data else ("", request.image_data)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Görüntü çözümlenemedi.")

        # Atakan'ın modeli buraya gelecek (Şimdilik test verisi dönüyoruz)
        return {"prediction": "Merhaba", "type": "IMAGE"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Resim işlenirken hata oluştu: {str(e)}")

# 🎥 VİDEO KISMI (Video Yükleme İçin)
@app.post("/video-tahmin-et")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir video dosyası yükleyin.")

    temp_name = f"temp_{int(time.time())}_{file.filename}"
    
    try:
        with open(temp_name, "wb") as f:
            content = await file.read()
            f.write(content)

        video = cv2.VideoCapture(temp_name)
        kare_sayisi = 0
        okunan_kareler = []

        if not video.isOpened():
            raise ValueError("Video dosyası açılamadı.")

        while video.isOpened():
            basari, kare = video.read()
            if not basari: break
            
            if kare_sayisi % 30 == 0:
                okunan_kareler.append(f"Kare_{kare_sayisi}")
            kare_sayisi += 1

        video.release() 
        
        return {
            "filename": file.filename,
            "islenen_kare_sayisi": len(okunan_kareler),
            "mesaj": "Video başarıyla analiz edildi.",
            "type": "VIDEO_PARCALAMA_TESTI"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video işlenirken bir hata oluştu: {str(e)}")
    
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)