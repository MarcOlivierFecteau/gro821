import numpy as np


def translation(tx: float, ty: float, tz: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = (tx, ty, tz)
    return transform


def rotation_y(angle: float) -> np.ndarray:
    """`angle` in radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def rotation_z(angle: float) -> np.ndarray:
    """`angle` in radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def flip_yz() -> np.ndarray:
    """Return a reflection matrix across the YZ plane."""
    transform = np.eye(4, dtype=np.float32)
    transform[0, 0] = -1
    return transform
