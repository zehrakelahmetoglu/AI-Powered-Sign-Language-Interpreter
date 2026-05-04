from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2 
import numpy as np
import base64
import os
import time
import sqlite3
import random
import torch
import json
from model import SignLSTM  # Atakan'ın model mimarisi

app = FastAPI(title="SignSpeak AI - Canlı Entegrasyon Backend")

# --- 🛰️ CORS AYARLARI ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- 🧠 YAPAY ZEKA MODEL YÜKLEME ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Kelime Haritasını (Label Map) Yükle[cite: 1]
try:
    with open('label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    id_to_word = {v: k for k, v in label_map.items()} # Sayıyı kelimeye çevirir
except Exception as e:
    print(f"⚠️ Label map yüklenemedi: {e}")
    id_to_word = {i: f"Kelime_{i}" for i in range(226)}

# 2. Modeli Başlat ve Ağırlıkları Yükle[cite: 1]
# Atakan'ın model parametreleri: 258 input, 226 sınıf[cite: 1]
model = SignLSTM(input_dim=258, num_classes=226).to(device)

try:
    # Atakan'ın dosyası .pt veya .pth ise burayı kullanıyoruz
    model.load_state_dict(torch.load('tsl_lstm_best.pt', map_location=device))
    model.eval()
    print("✅ Yapay Zeka Beyni Başarıyla Yüklendi!")
except Exception as e:
    print(f"⚠️ Model ağırlıkları yüklenemedi: {e}. Sistem simülasyon modunda çalışacak.")

# --- 📝 MODELLER ---
class ImageRequest(BaseModel):
    image_data: str

class TestRequest(BaseModel):
    oturum_id: str

# --- 🛠️ ANA TAHMİN VE LOGLAMA MOTORU ---
@app.post("/api/tahmin_yap")
async def tahmin_yap_ve_kaydet(request: TestRequest):
    start_time = time.time()
    
    # 1. MODEL TAHMİN ADIMI
    try:
        # Atakan keypointleri bağlayana kadar 1 batch, 30 frame, 258 feature simüle ediyoruz
        input_data = torch.randn(1, 30, 258).to(device) 
        
        with torch.no_grad():
            logits = model(input_data)
            prediction_id = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item() * 100
        
        tahmin = id_to_word.get(prediction_id, "ANLAŞILAMADI")
    except Exception as e:
        print(f"Tahmin Hatası: {e}")
        tahmin = "HATA"
        confidence = 0.0

    # 2. PERFORMANS HESAPLAMA
    gecikme_ms = round((time.time() - start_time) * 1000, 2)
    guven_skoru = round(confidence, 2)

    # 3. VERİTABANI KAYIT (Senin kurduğun sistem)
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

    return {
        "tahmin": tahmin, 
        "guven": guven_skoru, 
        "gecikme": gecikme_ms,
        "type": "AI_ENGINE"
    }

# --- 📸 DİĞER SERVİSLER ---
@app.get("/")
def home():
    return {"durum": "Sistem Aktif", "model": "LSTM-226-Class"}

@app.post("/tahmin-et")
async def predict_image(request: ImageRequest):
    # Bu kısım sadece arayüzde görüntünün akıp akmadığını test eder
    return {"prediction": "Kamera Aktif", "type": "IMAGE_STREAM"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)