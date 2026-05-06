import sqlite3

def canli_veritabani_kur():
    conn = sqlite3.connect("signspeak_live.sqlite3")
    cursor = conn.cursor()

    # ESKİ TABLOYU SİL (Garantiye alalım)
    cursor.execute("DROP TABLE IF EXISTS Hasta_Ceviri_Loglari")

    # YENİ VE TAM TABLOYU KUR
    cursor.execute('''
    CREATE TABLE Hasta_Ceviri_Loglari (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        oturum_id TEXT NOT NULL,        
        modelin_tahmini TEXT NOT NULL,  
        guven_skoru REAL,               
        gecikme_ms REAL,                
        dogru_cevrildi_mi BOOLEAN DEFAULT 1,      
        tarih TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
    )
    ''')

    conn.commit()
    conn.close()
    print("[ZAFER] Veritabanı TÜM sütunlarla (oturum_id dahil) sıfırdan kuruldu!")

if __name__ == "__main__":
    canli_veritabani_kur()