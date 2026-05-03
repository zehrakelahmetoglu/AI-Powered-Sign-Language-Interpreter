# AI-Based Sign Language Interpreter - API Documentation

## Overview
This document outlines the RESTful API endpoints for the AI-Based Sign Language Interpreter backend. The API is built using **FastAPI**, ensuring high-speed processing to simulate real-time communication. The system currently supports a 50-word vocabulary (hospital jargon) tailored for the Minimum Viable Product (MVP).

## Base URL
All API requests should be directed to the base URL (default local environment):
`http://localhost:8000`

---

## 1. Health Check

Checks if the API and backend services are up and running.

* **URL:** `/`
* **Method:** `GET`
* **Response Content-Type:** `application/json`

### Success Response
* **Code:** 200 OK
* **Body:**
```json
{
  "mesaj": "İşaret Dili API Çalışıyor - Video ve Fotoğraf Hazır!"
}
```

---

## 2. Predict Sign (Frame)

This endpoint is the core of the real-time split-screen application. It receives a single frame (image) captured by the patient's webcam, processes it through the AI model, and returns the predicted text for the receptionist.

* **URL:** `/tahmin-et`
* **Method:** `POST`
* **Content-Type:** `application/json`

### Request Payload
The frontend must send the captured frame as a Base64 encoded string inside a JSON object.

```json
{
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

### Success Response
Returns the predicted sign language translation.

* **Code:** 200 OK
* **Body:**
```json
{
  "prediction": "Merhaba",
  "type": "IMAGE"
}
```

### Error Response
* **Code:** 400 Bad Request (If the payload is missing or invalid)
* **Body:**
```json
{
  "detail": "Resim işlenirken hata oluştu: Görüntü çözümlenemedi."
}
```

---

## 3. Predict Sign (Video)

This endpoint processes short, pre-recorded videos. The backend extracts frames from the uploaded video and runs predictions. *(Note: This is an additional feature alongside the primary real-time MVP).*

* **URL:** `/video-tahmin-et`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`

### Request Payload
Content-Type: multipart/form-data

Body: file (Binary video file)

  
Key: file
Type: File (Seçilen video dosyası)

### Success Response
* **Code:** 200 OK
* **Body:**
```json
{
  "filename": "video.mp4",
  "islenen_kare_sayisi": 10,
  "mesaj": "Video başarıyla analiz edildi.",
  "type": "VIDEO_PARCALAMA_TESTI"
}
```

