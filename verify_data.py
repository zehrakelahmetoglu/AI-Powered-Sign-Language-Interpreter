import sqlite3

try:
    conn = sqlite3.connect("signspeak_live.sqlite3")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Hasta_Ceviri_Loglari")
    veriler = cursor.fetchall()
    
    print(f"\n--- TOPLAM {len(veriler)} KAYIT BULUNDU ---")
    for satir in veriler:
        print(satir)
    conn.close()
except Exception as e:
    print(f"Hata: {e}")