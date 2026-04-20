import cv2
import numpy as np


def translate_image(image, tx=25, ty=15):
    """Shift image by tx, ty pixels."""
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )

