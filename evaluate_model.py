import sqlite3
from datetime import datetime

DB_NAME = "db.sqlite3" 

def setup_database():
    """Test sonuçlarını loglayacağımız veritabanı altyapısını kurar."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 1. TABLO: Hangi testi, nereden yapıyoruz?
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Test_Senaryolari (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tarih TEXT,
        test_kaynagi TEXT,    -- 'Canli_Kamera' veya 'YouTube_Video'
        ortam_notu TEXT,      -- Örn: 'Ters Isik', '2 Metre Mesafe', 'YouTube in-the-wild'
        model_versiyonu TEXT  -- Örn: 'Baseline_v1', 'Buyuk_Veriseti_v2'
    )
    ''')

    # 2. TABLO: Modelin sınav kağıdı (Anlık tahminler)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Tahmin_Loglari (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        senaryo_id INTEGER,
        gercek_kelime TEXT,
        model_tahmini TEXT,
        guven_skoru REAL,     -- Modelin tahminden % kaç emin olduğu (Confidence)
        gecikme_ms REAL,      -- Sistemin hızı (Latency)
        FOREIGN KEY(senaryo_id) REFERENCES Test_Senaryolari(id)
    )
    ''')

    conn.commit()
    conn.close()
    print(f"[BAŞARILI] '{DB_NAME}' test veritabanı başarıyla kuruldu ve tablolar hazırlandı!")

if __name__ == "__main__":
    setup_database()