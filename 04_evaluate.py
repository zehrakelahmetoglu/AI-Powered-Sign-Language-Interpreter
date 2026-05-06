"""best_model.pt yükleyip validation seti üzerinde tam değerlendirme yapar.

Resmi AUTSL val split:
  python 04_evaluate.py \
    --val-labels-csv data/val_labels_full.csv \
    --val-keypoints-dir keypoints_val

Random split (fallback):
  python 04_evaluate.py
"""
import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from utils.dataset import load_official_splits, load_splits
from utils.model import SignLSTM

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model değerlendirme")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--labels-csv", default="data/train_labels_full.csv")
    p.add_argument("--keypoints-dir", default="keypoints_full")
    p.add_argument("--val-labels-csv", default=None,
                   help="Resmi AUTSL val CSV (belirtilirse resmi split kullanılır)")
    p.add_argument("--val-keypoints-dir", default="keypoints_val")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-ratio", type=float, default=0.15)
    return p.parse_args()


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets, all_top5 = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            top5 = logits.topk(5, dim=1).indices.cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())
            all_top5.append(top5)

    return (
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_top5, axis=0),
    )


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, num_classes: int) -> None:
    fig_size = max(14, num_classes // 3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=(num_classes <= 20),
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
    )
    ax.set_xlabel("Tahmin Edilen Sınıf")
    ax.set_ylabel("Gerçek Sınıf")
    ax.set_title(f"Confusion Matrix — {num_classes} Sınıf (Validation Set)")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    log.info("Confusion matrix kaydedildi: %s", out_path)


def top5_accuracy(all_preds_top5: np.ndarray, all_targets: np.ndarray) -> float:
    correct = sum(t in top5 for top5, t in zip(all_preds_top5, all_targets))
    return correct / len(all_targets)


def most_confused_pairs(cm: np.ndarray, n: int = 10) -> list[tuple[int, int, int]]:
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat = cm_no_diag.flatten()
    top_idx = np.argsort(flat)[::-1][:n]
    nc = cm.shape[0]
    return [(idx // nc, idx % nc, int(flat[idx])) for idx in top_idx]


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {ckpt_path}\n"
            "Önce '03_train_lstm.py' çalıştırın."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Cihaz: %s", device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    saved_args = ckpt.get("args", {})
    num_classes = saved_args.get("num_classes", 226)
    hidden_dim = saved_args.get("hidden_dim", 512)
    num_layers = saved_args.get("num_layers", 3)

    log.info("Checkpoint epoch: %d | val_acc: %.3f", ckpt["epoch"], ckpt["val_acc"])

    model = SignLSTM(
        input_dim=258,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    log.info("Model yüklendi (%d sınıf).", num_classes)

    # ── Validation dataset ────────────────────────────────────────────────────
    use_official = (
        args.val_labels_csv is not None
        and Path(args.val_labels_csv).exists()
        and Path(args.val_keypoints_dir).exists()
    )

    if use_official:
        log.info("Resmi AUTSL val seti kullanılıyor.")
        _, val_ds = load_official_splits(
            train_csv=args.labels_csv,
            train_kp_dir=args.keypoints_dir,
            val_csv=args.val_labels_csv,
            val_kp_dir=args.val_keypoints_dir,
        )
    else:
        log.info("Resmi val seti yok — random split ile değerlendirme yapılıyor.")
        _, val_ds = load_splits(args.labels_csv, args.keypoints_dir, val_ratio=args.val_ratio)

    log.info("Validation seti: %d örnek", len(val_ds))

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_preds, all_targets, all_top5 = run_inference(model, val_loader, device)

    top1 = (all_preds == all_targets).mean()
    top5 = top5_accuracy(all_top5, all_targets)
    log.info("Top-1 Accuracy: %.4f  (%.2f%%)", top1, top1 * 100)
    log.info("Top-5 Accuracy: %.4f  (%.2f%%)", top5, top5 * 100)

    report = classification_report(all_targets, all_preds, digits=4)
    log.info("\n%s", report)
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info("Sınıflandırma raporu kaydedildi: %s", report_path)

    cm = confusion_matrix(all_targets, all_preds)
    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path, num_classes)

    confused = most_confused_pairs(cm, n=10)
    log.info("En çok karıştırılan 10 çift (gerçek → tahmin: sayı):")
    for true_cls, pred_cls, count in confused:
        log.info("  Sınıf %3d → Sınıf %3d : %d kez", true_cls, pred_cls, count)


if __name__ == "__main__":
    main()
