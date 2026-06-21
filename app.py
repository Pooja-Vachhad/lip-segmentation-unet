"""
app.py
Streamlit lipstick virtual try-on app.
"""

import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Lip Studio", page_icon="💄", layout="wide")

st.markdown("""
    <style>
        .title { font-size: 2rem; font-weight: 700; color: #f2a7c3; text-align: center; }
        .subtitle { font-size: 0.95rem; color: #aaa; text-align: center; margin-bottom: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💄 Lip Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload · Pick a shade · See the look</div>', unsafe_allow_html=True)

# ── shade palette ──
SHADES = {
    "Pink": {
        "Baby Pink":  (255, 182, 193),
        "Rose Pink":  (255, 105, 148),
        "Hot Pink":   (255,  20, 147),
        "Blush Pink": (255, 160, 180),
        "Deep Pink":  (220,  50, 120),
    },
    "Nude": {
        "Warm Nude":  (210, 160, 130),
        "Beige Nude": (220, 185, 160),
        "Peach Nude": (230, 170, 140),
        "Mauve Nude": (190, 140, 140),
        "Dark Nude":  (170, 110,  90),
    },
    "Red": {
        "Classic Red": (220,  20,  60),
        "Cherry Red":  (180,   0,  30),
        "Brick Red":   (160,  50,  40),
        "Coral Red":   (255,  80,  60),
        "Deep Red":    (130,   0,  20),
    },
    "Blue": {
        "Sky Blue":    ( 80, 180, 255),
        "Royal Blue":  ( 30,  80, 220),
        "Midnight":    ( 20,  30, 120),
        "Teal Blue":   ( 20, 160, 180),
        "Periwinkle":  (130, 140, 240),
    },
    "Green": {
        "Mint":    (100, 220, 180),
        "Olive":   (100, 140,  50),
        "Forest":  ( 30, 100,  50),
        "Lime":    (160, 220,  50),
        "Emerald": ( 20, 160,  80),
    },
}

@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")

session = load_model()

def predict_mask(image_rgb):
    img = cv2.resize(image_rgb, (256, 256))
    img_norm = img.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    outputs = session.run(["output"], {"input": img_input})
    logits = outputs[0]
    pred = 1 / (1 + np.exp(-logits))
    pred = (pred > 0.5).astype(np.uint8)
    return pred[0, 0]

def apply_lip_color(image_rgb, mask, color, opacity):
    img = cv2.resize(image_rgb, (256, 256))
    color_layer = np.zeros_like(img)
    color_layer[:] = color
    blended = cv2.addWeighted(img, 1 - opacity, color_layer, opacity, 0)
    # soften edges with blur so color blends more naturally
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)
    mask_blur = np.stack([mask_blur, mask_blur, mask_blur], axis=-1)
    result = (mask_blur * blended + (1 - mask_blur) * img).astype(np.uint8)
    return result

# ── layout: left controls | right image ──
left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader("Upload a front-facing photo", type=["jpg", "jpeg", "png"])

    if uploaded:
        st.markdown("**Shade Family**")
        family = st.radio("", list(SHADES.keys()), horizontal=True, label_visibility="collapsed")

        # FIX: reset selected_shade when family changes so no KeyError
        if st.session_state.get("last_family") != family:
            st.session_state["selected_shade"] = list(SHADES[family].keys())[0]
            st.session_state["last_family"] = family

        st.markdown(f"**{family} Shades**")
        shade_names = list(SHADES[family].keys())
        cols = st.columns(len(shade_names))

        for col, name in zip(cols, shade_names):
            r, g, b = SHADES[family][name]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            with col:
                st.markdown(
                    f'<div style="width:32px;height:32px;border-radius:50%;background:{hex_color};'
                    f'margin:auto;border:2px solid #555;"></div>'
                    f'<div style="text-align:center;font-size:0.65rem;color:#ccc;margin-top:3px;">{name}</div>',
                    unsafe_allow_html=True
                )
                if st.button("✓", key=f"btn_{family}_{name}"):
                    st.session_state["selected_shade"] = name

        selected_shade = st.session_state.get("selected_shade", shade_names[0])
        selected_color = SHADES[family][selected_shade]

        r, g, b = selected_color
        hex_sel = f"#{r:02x}{g:02x}{b:02x}"
        st.markdown(
            f'<div style="margin-top:8px;">Selected: <span style="background:{hex_sel};'
            f'padding:2px 10px;border-radius:10px;color:white;font-size:0.85rem;">{selected_shade}</span></div>',
            unsafe_allow_html=True
        )

        opacity = st.slider("Opacity", 0.1, 0.9, 0.4, 0.05)

with right:
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        image_np = np.array(pil_image)

        with st.spinner("Applying..."):
            mask = predict_mask(image_np)
            result = apply_lip_color(image_np, mask, selected_color, opacity)

        # small side by side inside right column
        r1, r2 = st.columns(2)
        with r1:
            st.image(cv2.resize(image_np, (256, 256)), caption="Original", use_column_width=True)
        with r2:
            st.image(result, caption=selected_shade, use_column_width=True)
    else:
        st.info("Upload a photo on the left to get started.")
