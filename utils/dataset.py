from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.augmentation import apply_augmentation

SEED = 42


class AUTSLKeypointDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        keypoints_dir: Path | str,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.keypoints_dir = Path(keypoints_dir)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample_id, label = self.samples[idx]
        kp_path = self.keypoints_dir / f"{sample_id}.npy"
        if not kp_path.exists():
            raise FileNotFoundError(f"Keypoint dosyası bulunamadı: {kp_path}")
        x = np.load(str(kp_path)).astype(np.float32)
        if self.augment:
            x = apply_augmentation(x)
        return torch.from_numpy(x), label


def _load_csv_samples(
    csv_path: Path,
    keypoints_dir: Path,
    split_name: str,
) -> list[tuple[str, int]]:
    """CSV'den örnek listesi oluşturur; mevcut olmayan keypoint'leri uyarı vererek atlar."""
    import logging
    log = logging.getLogger(__name__)

    df = pd.read_csv(str(csv_path), header=None, names=["sample_id", "class_id"])
    mask = df["sample_id"].apply(lambda sid: (keypoints_dir / f"{sid}.npy").exists())
    n_missing = (~mask).sum()
    if n_missing > 0:
        log.warning("%s: %d örnek için keypoint dosyası bulunamadı, atlanıyor.", split_name, n_missing)
    df = df[mask].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            f"{split_name} için hiç geçerli keypoint örneği bulunamadı. "
            "02_extract_keypoints.py çalıştırıldı mı?"
        )
    return list(zip(df["sample_id"], df["class_id"].astype(int)))


def load_official_splits(
    train_csv: Path | str,
    train_kp_dir: Path | str,
    val_csv: Path | str,
    val_kp_dir: Path | str,
) -> tuple["AUTSLKeypointDataset", "AUTSLKeypointDataset"]:
    """AUTSL resmi train/val split'ini kullanır (user-independent bölünme)."""
    train_csv = Path(train_csv)
    train_kp_dir = Path(train_kp_dir)
    val_csv = Path(val_csv)
    val_kp_dir = Path(val_kp_dir)

    for p in (train_csv, val_csv):
        if not p.exists():
            raise FileNotFoundError(f"Etiket CSV bulunamadı: {p}")
    for p in (train_kp_dir, val_kp_dir):
        if not p.exists():
            raise FileNotFoundError(f"Keypoint dizini bulunamadı: {p}")

    train_samples = _load_csv_samples(train_csv, train_kp_dir, "Train")
    val_samples = _load_csv_samples(val_csv, val_kp_dir, "Val")

    train_ds = AUTSLKeypointDataset(train_samples, train_kp_dir, augment=True)
    val_ds = AUTSLKeypointDataset(val_samples, val_kp_dir, augment=False)
    return train_ds, val_ds


def load_splits(
    csv_path: Path | str,
    keypoints_dir: Path | str,
    val_ratio: float = 0.15,
) -> tuple["AUTSLKeypointDataset", "AUTSLKeypointDataset"]:
    """Train CSV'sini rastgele bölerek train/val dataset oluşturur (fallback)."""
    csv_path = Path(csv_path)
    keypoints_dir = Path(keypoints_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Etiket CSV bulunamadı: {csv_path}")
    if not keypoints_dir.exists():
        raise FileNotFoundError(f"Keypoint dizini bulunamadı: {keypoints_dir}")

    samples = _load_csv_samples(csv_path, keypoints_dir, "Train")
    labels = [s[1] for s in samples]

    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_ratio,
        random_state=SEED,
        stratify=labels,
    )

    train_ds = AUTSLKeypointDataset(train_samples, keypoints_dir, augment=True)
    val_ds = AUTSLKeypointDataset(val_samples, keypoints_dir, augment=False)
    return train_ds, val_ds
