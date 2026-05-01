"""Tam AUTSL 226-sınıf dataset hazırlığı.

Girdi: train_labels.csv  +  SignList_ClassId_TR_EN.csv
Çıktı (data/ altına):
  - train_labels_full.csv    (sample_id, class_id)
  - val_labels_full.csv      (varsa, validation_labels.csv'den)
  - class_mapping_full.json  ({class_id: {tr, en}})
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tam AUTSL dataset hazırlığı")
    p.add_argument(
        "--data-dir",
        default="/mnt/c/Users/Atakan/OneDrive/Desktop/Sign_Language_Data",
        help="Ham veri dizini (train_labels.csv ve SignList burada)",
    )
    p.add_argument(
        "--sign-list",
        default=None,
        help="SignList_ClassId_TR_EN.csv yolu (belirtilmezse --data-dir içinde aranır)",
    )
    p.add_argument("--out-dir", default="data", help="Çıktı dizini")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Train etiketleri ──────────────────────────────────────────────────────
    train_csv = data_dir / "train_labels.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train etiket CSV bulunamadı: {train_csv}")

    df_train = pd.read_csv(str(train_csv), header=None, names=["sample_id", "class_id"])
    log.info("Train: %d örnek, %d benzersiz sınıf", len(df_train), df_train["class_id"].nunique())

    out_train = out_dir / "train_labels_full.csv"
    df_train.to_csv(str(out_train), header=False, index=False)
    log.info("Kaydedildi: %s", out_train)

    # ── Validation etiketleri (varsa) ─────────────────────────────────────────
    for val_name in ("validation_labels.csv", "val_labels.csv"):
        val_csv = data_dir / val_name
        if val_csv.exists():
            df_val = pd.read_csv(str(val_csv), header=None, names=["sample_id", "class_id"])
            log.info("Val: %d örnek", len(df_val))
            out_val = out_dir / "val_labels_full.csv"
            df_val.to_csv(str(out_val), header=False, index=False)
            log.info("Kaydedildi: %s", out_val)
            break
    else:
        log.warning(
            "validation_labels.csv bulunamadı. "
            "validation_labels.zip dosyasını extract edin ve tekrar çalıştırın."
        )

    # ── Sınıf mapping ─────────────────────────────────────────────────────────
    sign_list_path = Path(args.sign_list) if args.sign_list else data_dir / "SignList_ClassId_TR_EN.csv"
    if not sign_list_path.exists():
        log.warning("SignList CSV bulunamadı: %s — mapping atlanıyor.", sign_list_path)
        return

    df_signs = pd.read_csv(str(sign_list_path))
    # Beklenen sütunlar: ClassId, TR, EN
    mapping: dict[int, dict[str, str]] = {}
    for _, row in df_signs.iterrows():
        mapping[int(row["ClassId"])] = {
            "tr": str(row["TR"]),
            "en": str(row["EN"]),
        }

    out_mapping = out_dir / "class_mapping_full.json"
    with open(out_mapping, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    log.info("Sınıf mapping kaydedildi: %s (%d sınıf)", out_mapping, len(mapping))

    # ── Özet ─────────────────────────────────────────────────────────────────
    all_ids = sorted(df_train["class_id"].unique())
    log.info(
        "\n=== ÖZET ===\n"
        "  Train örnekleri : %d\n"
        "  Sınıf sayısı    : %d\n"
        "  Class ID aralığı: %d – %d",
        len(df_train),
        len(all_ids),
        min(all_ids),
        max(all_ids),
    )


if __name__ == "__main__":
    main()
