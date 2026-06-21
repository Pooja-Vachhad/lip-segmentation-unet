"""
app.py

Streamlit lipstick virtual try-on app.
User uploads a face image, picks a lipstick shade, and sees the result live.
"""

import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

# ── page config ──
st.set_page_config(page_title="Lip Studio", page_icon="💄", layout="centered")

st.markdown("""
    <style>
        body { background-color: #1a1a1a; }
        .title { font-size: 2.2rem; font-weight: 700; color: #f2a7c3; text-align: center; margin-bottom: 0; }
        .subtitle { font-size: 1rem; color: #aaa; text-align: center; margin-bottom: 2rem; }
        .shade-label { font-size: 0.85rem; color: #ccc; margin-bottom: 0.3rem; }
        .stButton > button {
            border-radius: 20px;
            border: 2px solid #f2a7c3;
            background: transparent;
            color: #f2a7c3;
            padding: 0.3rem 1rem;
            font-size: 0.85rem;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            background: #f2a7c3;
            color: #1a1a1a;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💄 Lip Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your photo · Pick a shade · See the look</div>', unsafe_allow_html=True)

# ── shade palette ──
SHADES = {
    "Pink": {
        "Baby Pink":    (255, 182, 193),
        "Rose Pink":    (255, 105, 148),
        "Hot Pink":     (255,  20, 147),
        "Blush Pink":   (255, 160, 180),
        "Deep Pink":    (220,  50, 120),
    },
    "Nude": {
        "Warm Nude":    (210, 160, 130),
        "Beige Nude":   (220, 185, 160),
        "Peach Nude":   (230, 170, 140),
        "Mauve Nude":   (190, 140, 140),
        "Dark Nude":    (170, 110,  90),
    },
    "Red": {
        "Classic Red":  (220,  20,  60),
        "Cherry Red":   (180,   0,  30),
        "Brick Red":    (160,  50,  40),
        "Coral Red":    (255,  80,  60),
        "Deep Red":     (130,   0,  20),
    },
    "Blue": {
        "Sky Blue":     ( 80, 180, 255),
        "Royal Blue":   ( 30,  80, 220),
        "Midnight":     ( 20,  30, 120),
        "Teal Blue":    ( 20, 160, 180),
        "Periwinkle":   (130, 140, 240),
    },
    "Green": {
        "Mint":         (100, 220, 180),
        "Olive":        (100, 140,  50),
        "Forest":       ( 30, 100,  50),
        "Lime":         (160, 220,  50),
        "Emerald":      ( 20, 160,  80),
    },
}

# ── load onnx model (cached) ──
@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")

session = load_model()

# ── inference function ──
def predict_mask(image_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(image_rgb, (256, 256))
    img_norm = img.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))   # [H,W,C] → [C,H,W]
    img_input = np.expand_dims(img_input, axis=0)    # → [1,C,H,W]

    outputs = session.run(["output"], {"input": img_input})
    logits = outputs[0]
    pred = 1 / (1 + np.exp(-logits))                # sigmoid
    pred = (pred > 0.5).astype(np.uint8)
    return pred[0, 0]                                # [256, 256]

# ── overlay function ──
def apply_lip_color(image_rgb: np.ndarray, mask: np.ndarray, color: tuple, opacity: float) -> np.ndarray:
    img = cv2.resize(image_rgb, (256, 256))

    color_layer = np.zeros_like(img)
    color_layer[:] = color                           # fill with chosen color

    # blend image + color layer
    blended = cv2.addWeighted(img, 1 - opacity, color_layer, opacity, 0)

    # apply only on lip pixels
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    result = np.where(mask_3ch == 1, blended, img)
    return result.astype(np.uint8)

# ── ui layout ──
uploaded = st.file_uploader("Upload a front-facing photo", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_image = Image.open(uploaded).convert("RGB")
    image_np = np.array(pil_image)

    st.markdown("### Choose a shade family")
    family = st.radio("", list(SHADES.keys()), horizontal=True, label_visibility="collapsed")

    st.markdown(f"### {family} shades")
    shade_names = list(SHADES[family].keys())

    # show color swatches as columns
    cols = st.columns(len(shade_names))
    selected_shade = st.session_state.get("selected_shade", shade_names[0])

    for i, (col, name) in enumerate(zip(cols, shade_names)):
        r, g, b = SHADES[family][name]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        with col:
            st.markdown(
                f'<div style="width:40px;height:40px;border-radius:50%;background:{hex_color};'
                f'margin:auto;border:2px solid #444;"></div>'
                f'<div style="text-align:center;font-size:0.7rem;color:#ccc;margin-top:4px;">{name}</div>',
                unsafe_allow_html=True
            )
            if st.button("Pick", key=f"btn_{family}_{name}"):
                st.session_state["selected_shade"] = name
                selected_shade = name

    selected_color = SHADES[family][selected_shade]
    st.markdown(f"**Selected:** {selected_shade}")

    opacity = st.slider("Opacity", min_value=0.1, max_value=0.9, value=0.4, step=0.05)

    # run model + apply color
    with st.spinner("Applying lip color..."):
        mask = predict_mask(image_np)
        result = apply_lip_color(image_np, mask, selected_color, opacity)

    # show side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.resize(image_np, (256, 256)), caption="Original", use_column_width=True)
    with col2:
        st.image(result, caption=f"{family} · {selected_shade}", use_column_width=True)

else:
    st.info("Upload a photo above to get started.")