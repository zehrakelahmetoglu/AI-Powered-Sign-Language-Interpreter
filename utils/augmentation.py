import numpy as np

# Keypoint layout (per frame, 258 values):
#   Pose:       indices 0:132   — 33 landmarks × [x, y, z, visibility]
#   Left hand:  indices 132:195 — 21 landmarks × [x, y, z]
#   Right hand: indices 195:258 — 21 landmarks × [x, y, z]

_POSE_X_IDX = list(range(0, 132, 4))
_LHAND_X_IDX = list(range(132, 195, 3))
_RHAND_X_IDX = list(range(195, 258, 3))


def add_gaussian_noise(x: np.ndarray, std: float = 0.01) -> np.ndarray:
    return x + (np.random.randn(*x.shape).astype(np.float32) * std)


def temporal_mask(x: np.ndarray, max_frames: int = 3, prob: float = 0.3) -> np.ndarray:
    if np.random.random() >= prob:
        return x
    x = x.copy()
    T = x.shape[0]
    n = np.random.randint(1, min(max_frames, T) + 1)
    for idx in np.random.choice(T, size=n, replace=False):
        x[idx] = 0.0
    return x


def horizontal_flip(x: np.ndarray, prob: float = 0.5) -> np.ndarray:
    if np.random.random() >= prob:
        return x
    x = x.copy()
    x[:, _POSE_X_IDX] = 1.0 - x[:, _POSE_X_IDX]
    x[:, _LHAND_X_IDX] = 1.0 - x[:, _LHAND_X_IDX]
    x[:, _RHAND_X_IDX] = 1.0 - x[:, _RHAND_X_IDX]
    # swap left and right hand blocks
    tmp = x[:, 132:195].copy()
    x[:, 132:195] = x[:, 195:258]
    x[:, 195:258] = tmp
    return x


def apply_augmentation(x: np.ndarray) -> np.ndarray:
    x = add_gaussian_noise(x)
    x = temporal_mask(x)
    x = horizontal_flip(x)
    return x
