"""Bidirectional LSTM eğitimi: keypoints_full/ → checkpoints/best_model.pt

Resmi AUTSL split (önerilen):
  python 03_train_lstm.py \
    --val-labels-csv data/val_labels_full.csv \
    --val-keypoints-dir keypoints_val

Random split (fallback, val seti yoksa):
  python 03_train_lstm.py
"""
import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import load_official_splits, load_splits
from utils.model import SignLSTM

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()


class EarlyStopping:
    def __init__(self, patience: int = 15) -> None:
        self.patience = patience
        self.counter = 0
        self.best_val_acc = 0.0

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Epoch fonksiyonları ────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    clip_norm: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_top5 += top_k_accuracy(logits, y, k=5) * x.size(0)
            total_samples += x.size(0)

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        total_top5 / total_samples,
    )


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SignLSTM tam dataset eğitimi")

    # Veri
    p.add_argument("--labels-csv", default="data/train_labels_full.csv")
    p.add_argument("--keypoints-dir", default="keypoints_full")
    p.add_argument("--val-labels-csv", default=None,
                   help="Resmi AUTSL val CSV (belirtilirse resmi split kullanılır)")
    p.add_argument("--val-keypoints-dir", default="keypoints_val",
                   help="Val keypoint dizini (--val-labels-csv ile birlikte kullanılır)")

    # Dizinler
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--runs-dir", default="runs")

    # Eğitim
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.15,
                   help="Random split oranı (resmi val yoksa kullanılır)")

    # Model
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-classes", type=int, default=226)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Cihaz ─────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        log.warning("CUDA bulunamadı, CPU kullanılıyor (eğitim çok yavaş olabilir).")
        device = torch.device("cpu")

    # ── Dizinler ──────────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ── Veri ──────────────────────────────────────────────────────────────────
    use_official_split = (
        args.val_labels_csv is not None
        and Path(args.val_labels_csv).exists()
        and Path(args.val_keypoints_dir).exists()
    )

    if use_official_split:
        log.info("Resmi AUTSL train/val split kullanılıyor.")
        train_ds, val_ds = load_official_splits(
            train_csv=args.labels_csv,
            train_kp_dir=args.keypoints_dir,
            val_csv=args.val_labels_csv,
            val_kp_dir=args.val_keypoints_dir,
        )
    else:
        log.info("Resmi val seti bulunamadı — random %.0f%% split kullanılıyor.", args.val_ratio * 100)
        train_ds, val_ds = load_splits(args.labels_csv, args.keypoints_dir, val_ratio=args.val_ratio)

    log.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SignLSTM(
        input_dim=258,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
    ).to(device)
    log.info(
        "Model: hidden=%d, layers=%d, classes=%d | Parametre: %s",
        args.hidden_dim, args.num_layers, args.num_classes,
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    # ── Optimizasyon ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    early_stop = EarlyStopping(patience=args.early_stop_patience)

    # ── Loglama ───────────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(runs_dir))
    log_csv_path = ckpt_dir / "training_log.csv"
    log_csv_fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5_acc", "lr"]
    with open(log_csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_csv_fields).writeheader()

    best_val_acc = 0.0

    # ── Eğitim döngüsü ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device, args.clip_norm
        )
        val_loss, val_acc, val_top5 = val_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_acc)

        log.info(
            "Epoch %3d/%d | TrainLoss=%.4f TrainAcc=%.3f | ValLoss=%.4f ValAcc=%.3f ValTop5=%.3f | LR=%.2e",
            epoch, args.epochs,
            train_loss, train_acc,
            val_loss, val_acc, val_top5,
            current_lr,
        )

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("acc", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("val_top5_acc", val_top5, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        with open(log_csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_csv_fields).writerow({
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "val_top5_acc": f"{val_top5:.6f}",
                "lr": f"{current_lr:.2e}",
            })

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            },
            ckpt_dir / "last_model.pt",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                ckpt_dir / "best_model.pt",
            )
            log.info("  → Yeni en iyi model kaydedildi (val_acc=%.3f)", best_val_acc)

        if early_stop.step(val_acc):
            log.info("Early stopping tetiklendi (patience=%d). Eğitim durdu.", args.early_stop_patience)
            break

    writer.close()
    log.info("Eğitim tamamlandı. En iyi val_acc: %.3f", best_val_acc)
    log.info("Checkpoint: %s", ckpt_dir / "best_model.pt")


if __name__ == "__main__":
    main()
