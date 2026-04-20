import streamlit as st
import cv2
import os
import numpy as np

from utils.rotation import rotate_image
from utils.translation import translate_image
from utils.scaling import scale_image
from utils.affine import affine_transform
from utils.normalization import (
    estimate_upright_angle,
    estimate_center_shift,
    estimate_scale,
)

# -------------------------
# CONFIG
# -------------------------
IMAGE_FOLDER = "images"

st.set_page_config(layout="wide")
st.title("Traffic Sign Recognition - Preprocessing (Clean Output)")

# -------------------------
# LOAD IMAGES
# -------------------------
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpg", ".png", ".jpeg"))]

selected = st.selectbox("Select Traffic Sign Image", image_files)

# -------------------------
# READ IMAGE
# -------------------------
image_path = os.path.join(IMAGE_FOLDER, selected)
image = cv2.imread(image_path)

# Convert to RGB for display
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------
# PROCESSING PIPELINE (FIXED FOR YOUR DATA)
# -------------------------
def process_image(img):
    base = img.copy()

    # 1. Detect angle → straighten
    angle = estimate_upright_angle(base)
    rotated = rotate_image(base, angle)

    # 2. Center object
    tx, ty = estimate_center_shift(rotated)
    shifted = translate_image(rotated, tx, ty)

    # 3. Scale (zoom to focus sign)
    scale = estimate_scale(shifted)
    scaled = scale_image(shifted, scale, scale)

    # 4. VERY LIGHT affine (avoid distortion)
    final = affine_transform(scaled, strength=0.01)

    return final


# -------------------------
# RUN PROCESS
# -------------------------
if st.button("Process Image"):
    final_img = process_image(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(to_rgb(image), use_container_width=True)

    with col2:
        st.subheader("Final Processed Image")
        st.image(to_rgb(final_img), use_container_width=True)