import cv2


def scale_image(image, fx=0.75, fy=0.75):
    """
    Scale image around center while keeping original canvas size.
    fx, fy > 1.0 zoom in, fx, fy < 1.0 zoom out.
    """
    h, w = image.shape[:2]
    new_w = max(1, int(w * fx))
    new_h = max(1, int(h * fy))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if new_w >= w and new_h >= h:
        # Center-crop when zoomed in.
        x0 = (new_w - w) // 2
        y0 = (new_h - h) // 2
        return resized[y0 : y0 + h, x0 : x0 + w]

    # Center-pad when zoomed out.
    left = (w - new_w) // 2
    right = w - new_w - left
    top = (h - new_h) // 2
    bottom = h - new_h - top
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REPLICATE,
    )

