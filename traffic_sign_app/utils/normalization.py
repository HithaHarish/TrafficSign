import math

import cv2
import numpy as np


def _largest_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 140)
    kernel = np.ones((5, 5), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def estimate_upright_angle(image):
    """
    Estimate correction angle (degrees) to make dominant object upright.
    Positive value means rotate counter-clockwise.
    """
    contour = _largest_contour(image)
    if contour is None or len(contour) < 5:
        return 0.0

    vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    line_angle = math.degrees(math.atan2(float(vy), float(vx)))
    if line_angle < 0:
        line_angle += 180.0

    # Make the principal axis align with vertical (90 degrees).
    delta = ((line_angle - 90.0 + 90.0) % 180.0) - 90.0
    correction = -delta
    return float(np.clip(correction, -45.0, 45.0))


def detect_bbox(image):
    """Return dominant contour bounding box as (x, y, w, h)."""
    contour = _largest_contour(image)
    if contour is None:
        h, w = image.shape[:2]
        return 0, 0, w, h
    return cv2.boundingRect(contour)


def estimate_center_shift(image):
    """Estimate translation values required to center dominant object."""
    h, w = image.shape[:2]
    x, y, bw, bh = detect_bbox(image)
    cx_obj = x + (bw / 2.0)
    cy_obj = y + (bh / 2.0)
    cx_img = w / 2.0
    cy_img = h / 2.0
    tx = int(round(cx_img - cx_obj))
    ty = int(round(cy_img - cy_obj))
    return tx, ty


def estimate_scale(image, target_fill=0.62):
    """
    Estimate scaling factor so dominant object fills a target fraction of frame.
    """
    h, w = image.shape[:2]
    _, _, bw, bh = detect_bbox(image)
    current = max(1.0, float(max(bw, bh)))
    target = target_fill * float(min(w, h))
    scale = target / current
    return float(np.clip(scale, 0.8, 2.2))

