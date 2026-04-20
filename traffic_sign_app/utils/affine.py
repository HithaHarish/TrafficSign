import cv2
import numpy as np


def affine_transform(image, strength=0.03):
    """Apply a mild affine correction (near-identity) for geometric cleanup."""
    rows, cols = image.shape[:2]
    strength = max(0.0, min(0.08, float(strength)))

    dx = cols * strength
    dy = rows * strength

    src = np.float32(
        [
            [0, 0],
            [cols - 1, 0],
            [0, rows - 1],
        ]
    )
    dst = np.float32(
        [
            [dx, dy],
            [cols - 1 - dx, 0],
            [0, rows - 1 - dy],
        ]
    )
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT)

