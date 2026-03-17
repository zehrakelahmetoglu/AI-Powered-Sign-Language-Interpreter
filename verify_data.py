"""
AI-Powered Sign Language Interpreter
=====================================
Module  : Dataset Doğrulama
Author  : Atakan Yılmaz (AI & Data)

Açıklama:
    Toplanan .npy verilerinin bütünlüğünü, shape'ini ve
    içeriğini hızlıca kontrol eder. collect_data.py
    çalıştırıldıktan sonra kullan.

Kullanım:
    python verify_data.py
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def verify_dataset(data_dir: str) -> None:
    print("=" * 55)
    print("  Dataset Doğrulama Raporu")
    print("=" * 55)

    if not os.path.isdir(data_dir):
        print(f"[HATA] '{data_dir}' bulunamadı. Önce collect_data.py çalıştır.")
        return

    classes = sorted(os.listdir(data_dir))
    if not classes:
        print("[HATA] Hiç sınıf klasörü yok.")
        return

    total_ok    = 0
    total_bad   = 0

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
        ok, bad = 0, 0

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                arr = np.load(fpath)
                # Beklenen shape: (30, 63)
                if arr.shape == (30, 63) and not np.any(np.isnan(arr)):
                    ok += 1
                else:
                    print(f"  [WARN] {cls}/{fname} — beklenmeyen shape: {arr.shape}")
                    bad += 1
            except Exception as exc:
                print(f"  [ERR]  {cls}/{fname} — {exc}")
                bad += 1

        status = "✓" if bad == 0 else "✗"
        bar    = "█" * ok + "░" * max(0, 50 - ok)
        print(f"  {status} {cls:<20} [{bar[:25]}]  {ok} OK  {bad} BAD")

        total_ok  += ok
        total_bad += bad

    print("-" * 55)
    print(f"  Toplam geçerli sekans  : {total_ok}")
    print(f"  Toplam hatalı sekans   : {total_bad}")
    print(f"  Tahmini memory (float32): "
          f"{total_ok * 30 * 63 * 4 / 1024 / 1024:.2f} MB")
    print("=" * 55)


if __name__ == "__main__":
    verify_dataset(DATA_DIR)
