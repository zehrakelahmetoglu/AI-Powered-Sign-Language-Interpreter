from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2 
import numpy as np
import base64
import os
import time

app = FastAPI(title="İşaret Dili Çevirici - Hibrit Backend")

# --- 🛰️ CORS AYARLARI ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class ImageRequest(BaseModel):
    image_data: str

@app.get("/")
def home():
    return {"mesaj": "İşaret Dili API Çalışıyor - Video ve Fotoğraf Hazır!"}

# 📸 FOTOĞRAF KISMI (TRY-EXCEPT EKLENDİ)
@app.post("/tahmin-et")
async def predict_image(request: ImageRequest):
    try:
        header, encoded = request.image_data.split(",", 1) if "," in request.image_data else ("", request.image_data)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Görüntü çözümlenemedi.")

        # Atakan'ın modeli buraya gelecek
        return {"prediction": "Merhaba", "type": "IMAGE"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Resim işlenirken hata oluştu: {str(e)}")

# 🎥 VİDEO KISMI (TRY-EXCEPT-FINALLY EKLENDİ)
@app.post("/video-tahmin-et")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir video dosyası yükleyin.")

    # Çakışmaları önlemek için dosya ismine zaman ekledik
    temp_name = f"temp_{int(time.time())}_{file.filename}"
    
    try:
        # 1. Videoyu geçici olarak kaydet
        with open(temp_name, "wb") as f:
            content = await file.read()
            f.write(content)

        # 2. OpenCV ile video analizi
        video = cv2.VideoCapture(temp_name)
        kare_sayisi = 0
        okunan_kareler = []

        if not video.isOpened():
            raise ValueError("Video dosyası açılamadı.")

        while video.isOpened():
            basari, kare = video.read()
            if not basari:
                break
            
            # Her 30 karede bir işlem yap (Saniyede 1 kare gibi)
            if kare_sayisi % 30 == 0:
                okunan_kareler.append(f"Kare_{kare_sayisi}")
            kare_sayisi += 1

        video.release() # Analiz bitti, dosyayı serbest bırak
        
        return {
            "filename": file.filename,
            "islenen_kare_sayisi": len(okunan_kareler),
            "mesaj": "Video başarıyla analiz edildi.",
            "type": "VIDEO_PARCALAMA_TESTI"
        }

    except Exception as e:
        # Bir hata olursa kullanıcıya bildir
        raise HTTPException(status_code=500, detail=f"Video işlenirken bir hata oluştu: {str(e)}")
    
    finally:
        # HATA OLSA DA OLMASA DA: Geçici dosyayı sil (Sunucuyu temiz tut)
        if os.path.exists(temp_name):
            os.remove(temp_name)
            print(f"🗑️ Geçici dosya silindi: {temp_name}")
