"""
AI-Powered Sign Language Interpreter
=====================================
Module  : LSTM Model Trainer
Author  : Atakan Yılmaz and Zeynep Ötegen (AI & Data)
Version : 1.0.0

Açıklama:
    data/ altındaki .npy landmark dizilerini (30, 63) kullanarak
    Keras/TensorFlow LSTM modelini eğitir.

    RTX 4060 (8 GB VRAM) için optimize edilmiştir:
    - Mixed Precision (float16) → Tensor Core kullanımı
    - cuDNN-fused LSTM katmanları
    - tf.data pipeline ile GPU beslemesi
    - Dinamik GPU bellek büyütme

Model Mimarisi:
    Input(30, 63) → LSTM(128) → Dropout → LSTM(64) → Dropout
    → Dense(64, relu) → Dropout → Dense(num_classes, softmax)

Çıktılar:
    models/tsl_lstm_best.keras    ← en iyi model
    models/label_map.json         ← sınıf → index eşleşmesi
    logs/lstm_training/           ← TensorBoard logları
    logs/training_history.json    ← metrik geçmişi
    logs/confusion_matrix.png     ← confusion matrix görselleştirme
    logs/training_curves.png      ← loss/accuracy grafikleri

Kullanım:
    python train_lstm.py                                # varsayılan ayarlarla
    python train_lstm.py --epochs 200 --batch-size 128  # özelleştirilmiş
    python train_lstm.py --data-dir /başka/data/        # farklı veri dizini
    python train_lstm.py --resume models/tsl_lstm_best.keras  # devam
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────
#  GPU KONFIGÜRASYONU (import öncesi çağrılmalı)
# ─────────────────────────────────────────────────────────────

def configure_gpu():
    """
    TensorFlow GPU ayarları:
    - Dinamik bellek büyütme (tüm VRAM'i işgal etme)
    - Mixed precision (Tensor Core kullanımı)
    """
    import tensorflow as tf

    # GPU bellek büyütme
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] {len(gpus)} GPU bulundu: {[g.name for g in gpus]}")
    else:
        print("[GPU] GPU bulunamadı! CPU ile eğitim yapılacak.")
        print("      CUDA/cuDNN kurulu değilse: pip install tensorflow[and-cuda]")

    # Mixed Precision — RTX 4060 Tensor Core kullanımı
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[GPU] Mixed Precision (float16) aktif → Tensor Core kullanılacak")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")


configure_gpu()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# sklearn opsiyonel (classification report + confusion matrix)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn kurulu değil → train/test split numpy ile yapılacak")

# matplotlib opsiyonel (grafikler)
try:
    import matplotlib
    matplotlib.use("Agg")  # GUI olmadan kaydet
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib kurulu değil → grafikler kaydedilmeyecek")


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"
LOG_DIR      = PROJECT_ROOT / "logs"

FRAMES_PER_SEQ = 30
FEATURES_DIM   = 63
INPUT_SHAPE    = (FRAMES_PER_SEQ, FEATURES_DIM)  # (30, 63)

# Minimum sınıf başına sekans sayısı (az verili sınıfları filtrele)
MIN_SAMPLES_PER_CLASS = 3


# ─────────────────────────────────────────────────────────────
#  VERİ YÜKLEME
# ─────────────────────────────────────────────────────────────

def load_dataset(
    data_dir: Path,
    min_samples: int = MIN_SAMPLES_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    data/ altındaki tüm .npy dosyalarını yükler.

    Dönüş:
        X: np.ndarray shape=(N, 30, 63)
        y: np.ndarray shape=(N,) — integer class indices
        labels: list[str] — sınıf adları (index sırası)
    """
    print("\n" + "=" * 60)
    print("  Veri Yükleme")
    print("=" * 60)

    # Sınıfları ve dosyaları topla
    class_data = {}
    for cls_dir in sorted(data_dir.iterdir()):
        if not cls_dir.is_dir():
            continue

        npys = sorted(cls_dir.glob("*.npy"))
        if len(npys) < min_samples:
            print(f"  [ATLA] {cls_dir.name:<28} ({len(npys)} sekans < {min_samples} min)")
            continue

        sequences = []
        for npy in npys:
            try:
                arr = np.load(str(npy))
                if arr.shape == (FRAMES_PER_SEQ, FEATURES_DIM):
                    sequences.append(arr)
                else:
                    print(f"  [WARN] Shape uyumsuz: {npy.name} → {arr.shape}")
            except Exception as exc:
                print(f"  [WARN] Yükleme hatası: {npy.name} → {exc}")

        if len(sequences) >= min_samples:
            class_data[cls_dir.name] = sequences

    if not class_data:
        raise ValueError(
            f"Yeterli veri bulunamadı!\n"
            f"  data_dir: {data_dir}\n"
            f"  min_samples: {min_samples}\n"
            "  Önce local_dataset_processor.py veya collect_data.py çalıştırın."
        )

    # Sıralı label listesi
    labels = sorted(class_data.keys())
    label_to_idx = {name: idx for idx, name in enumerate(labels)}

    # Numpy dizilerine birleştir
    X_list = []
    y_list = []
    for name in labels:
        seqs = class_data[name]
        X_list.extend(seqs)
        y_list.extend([label_to_idx[name]] * len(seqs))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\n  Yüklenen Sınıflar:")
    for name in labels:
        cnt = len(class_data[name])
        bar = "█" * min(cnt, 40) + "░" * max(0, 10 - cnt)
        print(f"  {name:<28} {bar} {cnt}")

    print(f"\n  Toplam : {X.shape[0]} sekans, {len(labels)} sınıf")
    print(f"  Shape  : X={X.shape}, y={y.shape}")
    print("=" * 60)

    return X, y, labels


# ─────────────────────────────────────────────────────────────
#  VERİ BÖLME & tf.data PIPELINE
# ─────────────────────────────────────────────────────────────

def create_datasets(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    batch_size: int = 64,
    val_split: float = 0.15,
) -> tuple:
    """
    Stratified train/val split + tf.data.Dataset oluştur.

    Dönüş:
        train_ds, val_ds, X_val, y_val_int (metrikler için)
    """
    # One-hot encode
    y_onehot = keras.utils.to_categorical(y, num_classes=num_classes)

    # Stratified split
    if HAS_SKLEARN:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_onehot,
            test_size=val_split,
            stratify=y,
            random_state=42,
        )
        y_val_int = np.argmax(y_val, axis=1)
    else:
        # Basit random split
        n = len(X)
        indices = np.random.RandomState(42).permutation(n)
        split = int(n * (1 - val_split))
        train_idx, val_idx = indices[:split], indices[split:]
        X_train, y_train = X[train_idx], y_onehot[train_idx]
        X_val, y_val     = X[val_idx],   y_onehot[val_idx]
        y_val_int = np.argmax(y_val, axis=1)

    print(f"\n  Train : {X_train.shape[0]} sekans")
    print(f"  Val   : {X_val.shape[0]} sekans")
    print(f"  Batch : {batch_size}")

    # tf.data pipeline — GPU'yu verimli beslemek için
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=min(len(X_train), 10000), seed=42)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, X_val, y_val_int


# ─────────────────────────────────────────────────────────────
#  MODEL — Bidirectional LSTM + Dropout
# ─────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    input_shape: tuple = INPUT_SHAPE,
    lstm1_units: int = 128,
    lstm2_units: int = 64,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    LSTM sequence classifier.

    Mimari:
        Input(30, 63)
        → LSTM(128, return_sequences=True)  → Dropout(0.3)
        → LSTM(64)                          → Dropout(0.3)
        → Dense(64, relu)                   → Dropout(0.2)
        → Dense(num_classes, softmax)       — float32 (mixed precision uyumu)

    cuDNN LSTM otomatik aktif olur:
        - activation='tanh', recurrent_activation='sigmoid' (varsayılan)
        - unroll=False (varsayılan)
        - use_bias=True (varsayılan)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape, name="landmark_input"),

        # LSTM Katman 1 — sequence → sequence
        layers.LSTM(
            lstm1_units,
            return_sequences=True,
            name="lstm_1",
        ),
        layers.Dropout(dropout_rate, name="dropout_1"),

        # LSTM Katman 2 — sequence → vector
        layers.LSTM(
            lstm2_units,
            return_sequences=False,
            name="lstm_2",
        ),
        layers.Dropout(dropout_rate, name="dropout_2"),

        # Dense katmanlar
        layers.Dense(dense_units, activation="relu", name="dense_1"),
        layers.Dropout(dropout_rate * 0.66, name="dropout_3"),

        # Çıkış — mixed precision için float32 zorunlu
        layers.Dense(num_classes, dtype="float32", name="output"),
        layers.Activation("softmax", dtype="float32", name="softmax"),
    ], name="tsl_lstm")

    # Cosine decay learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ─────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────

def get_callbacks(
    model_save_path: Path,
    log_dir: Path,
    patience: int = 15,
) -> list:
    """Eğitim callback'leri."""
    cb_list = [
        # En iyi modeli kaydet
        callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        # Erken durdurma
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),

        # LR azaltma (plato)
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),

        # TensorBoard
        callbacks.TensorBoard(
            log_dir=str(log_dir / "lstm_training"),
            histogram_freq=1,
            write_graph=True,
        ),
    ]
    return cb_list


# ─────────────────────────────────────────────────────────────
#  EĞİTİM SONRASI ANALİZ
# ─────────────────────────────────────────────────────────────

def save_training_history(history, log_dir: Path) -> None:
    """Eğitim geçmişini JSON olarak kaydet."""
    history_dict = {}
    for key, val in history.history.items():
        history_dict[key] = [float(v) for v in val]

    path = log_dir / "training_history.json"
    with open(str(path), "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[KAYIT] Eğitim geçmişi: {path}")


def plot_training_curves(history, log_dir: Path) -> None:
    """Eğitim/doğrulama loss ve accuracy grafiklerini kaydet."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history.history["loss"], label="Train Loss", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Eğitim ve Doğrulama Kaybı")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Eğitim ve Doğrulama Doğruluğu")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = log_dir / "training_curves.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"[KAYIT] Eğitim grafikleri: {path}")


def evaluate_and_report(
    model: keras.Model,
    X_val: np.ndarray,
    y_val_int: np.ndarray,
    labels: list[str],
    log_dir: Path,
) -> None:
    """Classification report + confusion matrix."""
    # Tahmin
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Classification Report
    if HAS_SKLEARN:
        print("\n" + "=" * 60)
        print("  Classification Report")
        print("=" * 60)
        report = classification_report(
            y_val_int, y_pred,
            target_names=labels,
            zero_division=0,
        )
        print(report)

        # JSON olarak da kaydet
        report_dict = classification_report(
            y_val_int, y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        with open(str(log_dir / "classification_report.json"), "w") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

    # Confusion Matrix
    if HAS_SKLEARN and HAS_MPL:
        cm = confusion_matrix(y_val_int, y_pred)

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.4),
                                        max(8, len(labels) * 0.35)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=14)
        plt.colorbar(im, ax=ax)

        # Sınıf etiketlerini göster (çok sınıf varsa döndür)
        if len(labels) <= 30:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)

        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        plt.tight_layout()

        path = log_dir / "confusion_matrix.png"
        plt.savefig(str(path), dpi=150)
        plt.close()
        print(f"[KAYIT] Confusion matrix: {path}")

    # Basit doğruluk
    accuracy = np.mean(y_pred == y_val_int) * 100
    print(f"\n  Doğrulama Doğruluğu: {accuracy:.2f}%")


# ─────────────────────────────────────────────────────────────
#  ANA EĞİTİM PIPELINE
# ─────────────────────────────────────────────────────────────

def run_training(
    data_dir: Path          = DATA_DIR,
    model_dir: Path         = MODEL_DIR,
    log_dir: Path           = LOG_DIR,
    epochs: int             = 100,
    batch_size: int         = 64,
    learning_rate: float    = 1e-3,
    patience: int           = 15,
    min_samples: int        = MIN_SAMPLES_PER_CLASS,
    resume_path: Optional[str] = None,
) -> None:
    """Ana eğitim pipeline'ı."""
    t_start = time.time()

    print("=" * 60)
    print("  TSL LSTM Model Eğitimi")
    print("=" * 60)
    print(f"  TensorFlow  : {tf.__version__}")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  GPU(lar)    : {[g.name for g in tf.config.list_physical_devices('GPU')]}")
    print(f"  Precision   : {tf.keras.mixed_precision.global_policy().name}")
    print(f"  Veri dizini : {data_dir}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  LR          : {learning_rate}")
    print(f"  Patience    : {patience}")
    print("=" * 60)

    # Dizinler
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Veri yükle
    X, y, labels = load_dataset(data_dir, min_samples)
    num_classes = len(labels)

    # 2. Label map kaydet
    label_map = {name: int(idx) for idx, name in enumerate(labels)}
    label_map_path = model_dir / "label_map.json"
    with open(str(label_map_path), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"[KAYIT] Label map: {label_map_path} ({num_classes} sınıf)")

    # 3. tf.data pipeline
    train_ds, val_ds, X_val, y_val_int = create_datasets(
        X, y, num_classes, batch_size
    )

    # 4. Model oluştur / yükle
    model_save_path = model_dir / "tsl_lstm_best.keras"

    if resume_path and Path(resume_path).is_file():
        print(f"\n[RESUME] Model yükleniyor: {resume_path}")
        model = keras.models.load_model(resume_path)
        # LR'yi özelleştirilmiş değerle güncelle
        model.optimizer.learning_rate.assign(learning_rate)
    else:
        model = build_model(
            num_classes=num_classes,
            learning_rate=learning_rate,
        )

    # 5. Callbacks
    cb_list = get_callbacks(model_save_path, log_dir, patience)

    # 6. Eğitim
    print("\n" + "=" * 60)
    print("  EĞİTİM BAŞLIYOR")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        verbose=1,
    )

    elapsed = time.time() - t_start

    # 7. Sonuç kayıt
    save_training_history(history, log_dir)
    plot_training_curves(history, log_dir)

    # 8. En iyi modeli yükle ve değerlendir
    if model_save_path.is_file():
        print(f"\n[KAYIT] En iyi model yükleniyor: {model_save_path}")
        best_model = keras.models.load_model(str(model_save_path))
    else:
        best_model = model

    evaluate_and_report(best_model, X_val, y_val_int, labels, log_dir)

    # 9. Özet
    best_val_acc = max(history.history.get("val_accuracy", [0]))
    best_val_loss = min(history.history.get("val_loss", [float("inf")]))
    total_epochs = len(history.history["loss"])

    print("\n" + "=" * 60)
    print("  EĞİTİM TAMAMLANDI")
    print("=" * 60)
    print(f"  Toplam epoch     : {total_epochs}")
    print(f"  En iyi val_acc   : {best_val_acc:.4f} ({best_val_acc * 100:.2f}%)")
    print(f"  En iyi val_loss  : {best_val_loss:.4f}")
    print(f"  Süre             : {elapsed:.1f}s ({elapsed / 60:.1f} dk)")
    print(f"  Model            : {model_save_path}")
    print(f"  Label map        : {label_map_path}")
    print(f"  TensorBoard      : tensorboard --logdir {log_dir / 'lstm_training'}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSL LSTM Model Trainer — RTX 4060 Optimized",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help=f"Veri dizini (varsayılan: {DATA_DIR})"
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(MODEL_DIR),
        help=f"Model kayıt dizini (varsayılan: {MODEL_DIR})"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Maksimum epoch sayısı (varsayılan: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch boyutu (varsayılan: 64, RTX 4060 için optimal)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Öğrenme hızı (varsayılan: 0.001)"
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="EarlyStopping sabır (varsayılan: 15)"
    )
    parser.add_argument(
        "--min-samples", type=int, default=MIN_SAMPLES_PER_CLASS,
        help=f"Min sekans/sınıf (varsayılan: {MIN_SAMPLES_PER_CLASS})"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Önceki modelden devam et (keras dosya yolu)"
    )

    args = parser.parse_args()

    run_training(
        data_dir      = Path(args.data_dir),
        model_dir     = Path(args.model_dir),
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        learning_rate = args.lr,
        patience      = args.patience,
        min_samples   = args.min_samples,
        resume_path   = args.resume,
    )


if __name__ == "__main__":
    main()
