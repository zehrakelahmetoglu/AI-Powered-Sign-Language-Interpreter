from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import cv2  #OpenCV kütüphanesi
import numpy as np
import base64
import os  #dosya silme/oluşturma
import time

app = FastAPI(title="İşaret Dili Çevirici - Hibrit Backend")

class ImageRequest(BaseModel):
    image_data: str

@app.get("/")
def home():
    return {"mesaj": "İşaret Dili API Çalışıyor - Video ve Fotoğraf Hazır!"}

# 📸 FOTOĞRAF KISMI 
@app.post("/tahmin-et")
async def predict_image(request: ImageRequest):
    try:
        hadeer, encoded = request.image_data.split(",", 1) if "," in request.image_data else ("", request.image_data)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return {"prediction": "Merhaba", "type": "IMAGE"}
    except Exception:
        raise HTTPException(status_code=400, detail="Resim işlenemedi!")

# 🎥 VİDEO KISMI (İşte burayı 'akıllandırdık')
@app.post("/video-tahmin-et")
async def predict_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir video dosyası yükleyin.")

    # 1. Videoyu işleyebilmek için geçici bir dosya olarak kaydedelim
    temp_name = f"temp_{file.filename}"
    with open(temp_name, "wb") as f:
        f.write(await file.read())

    # 2. OpenCV ile videonun içine giriyoruz
    video = cv2.VideoCapture(temp_name)
    kare_sayisi = 0
    okunan_kareler = []

    print(f"🎬 Video analiz ediliyor: {file.filename}")

    while video.isOpened():
        basari, kare = video.read()
        if not basari:
            break
        
        # Saniyede yaklaşık 1 kare alalım (Her 30 karede bir)
        if kare_sayisi % 30 == 0:
            # Burada 'kare' artık bir fotoğraf gibi elimizde!
            print(f"✅ Kare {kare_sayisi} yakalandı ve işlenmeye hazır.")
            okunan_kareler.append(f"Kare_{kare_sayisi}")

        kare_sayisi += 1

    video.release()
    
    # 3. İşimiz bitti, geçici dosyayı silelim (Bilgisayarın dolmasın)
    if os.path.exists(temp_name):
        os.remove(temp_name)

    return {
        "filename": file.filename,
        "islenen_kare_sayisi": len(okunan_kareler),
        "mesaj": "Video başarıyla parçalandı. Model gelince harfler burada görünecek!",
        "type": "VIDEO_PARCALAMA_TESTI"
    }