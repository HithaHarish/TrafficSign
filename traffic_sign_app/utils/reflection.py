import cv2


def reflect_image(image, mode=1):
    """Reflect image. mode: 1 horizontal, 0 vertical, -1 both."""
    return cv2.flip(image, mode)

