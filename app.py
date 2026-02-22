# AI Portrait Cleaner — ID-Style
# Fresh build: clean white/near-white/blue background, natural bright tone, centered crop, crisp details.
# No Gemini required. Uses rembg for precise cut-out + edge feather + tuned relight pipeline.

import streamlit as st
import io
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2

# Optional background removal
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    HAS_REMBG = False

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ID-Style Portrait Cleaner", page_icon="🪪", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# EXPORT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def save_jpeg_bytes(pil_img, quality=95, subsampling="4:4:4"):
    """
    High-quality JPEG export with 4:4:4 subsampling to keep skin gradients smooth.
    """
    buf = io.BytesIO()
    pil_img = pil_img.convert("RGB")
    subsampling_map = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
    ss = subsampling_map.get(subsampling, 0)
    pil_img.save(buf, format="JPEG", quality=quality, subsampling=ss, optimize=True)
    return buf.getvalue()

def mm_to_pixels(mm, dpi=300):
    inches = float(mm) / 25.4
    return int(round(inches * dpi))

def place_on_canvas(pil_img, target_mm=(35, 45), dpi=300, bg=(255, 255, 255)):
    """
    Center the image on an exact-size canvas (e.g., 35x45 mm) at chosen DPI.
    Keeps small margin around subject for clean print.
    """
    tw = mm_to_pixels(target_mm[0], dpi)
    th = mm_to_pixels(target_mm[1], dpi)
    canvas = Image.new("RGB", (tw, th), bg)
    iw, ih = pil_img.size
    # Fit inside with a small border
    scale = min((tw - 12) / iw, (th - 12) / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = pil_img.resize((nw, nh), Image.LANCZOS)
    x = (tw - nw) // 2
    y = (th - nh) // 2
    canvas.paste(resized, (x, y))
    return canvas

# ──────────────────────────────────────────────────────────────────────────────
# FACE & FRAMING
# ──────────────────────────────────────────────────────────────────────────────
def detect_faces_bboxes(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    return faces

def face_focus_crop(pil_img, target_aspect=3/4, pad=0.22):
    """
    Crop to a portrait-friendly 3:4 aspect around the largest face,
    with margins that keep hairline and shoulders for an ID look.
    """
    w, h = pil_img.size
    faces = detect_faces_bboxes(pil_img)
    if len(faces) == 0:
        # No face detected → return original (user can crop manually if needed)
        return pil_img

    # largest face
    (x, y, fw, fh) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    cx, cy = x + fw / 2, y + fh / 2

    # Box height with padding; width from aspect; ensure shoulders margin
    box_h = fh * (1 + pad * 3.2)
    box_w = max(fw * (1 + pad * 2.4), box_h * target_aspect)

    left = int(max(0, cx - box_w / 2))
    top = int(max(0, cy - fh * 0.70))       # slightly higher to include more shoulder
    right = int(min(w, left + box_w))
    bottom = int(min(h, top + box_h))

    # Enforce 3:4 exact
    cw, ch = right - left, bottom - top
    cur_aspect = cw / max(1, ch)
    if cur_aspect > target_aspect:
        new_w = int(ch * target_aspect)
        dx = (cw - new_w) // 2
        left += dx
        right = left + new_w
    else:
        new_h = int(cw / target_aspect)
        dy = (ch - new_h) // 2
        top += dy
        bottom = top + new_h

    left, top, right, bottom = map(int, [max(0, left), max(0, top), min(w, right), min(h, bottom)])
    return pil_img.crop((left, top, right, bottom))

# ──────────────────────────────────────────────────────────────────────────────
# TONE / RELIGHT PIPELINE (natural brightness, clean color, crisp details)
# ──────────────────────────────────────────────────────────────────────────────
def auto_white_balance_bgr(img_bgr):
    """
    Gray-world white balance for natural color; robust and fast.
    """
    result = img_bgr.astype(np.float32)
    b, g, r = result[:,:,0], result[:,:,1], result[:,:,2]
    avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    eps = 1e-6
    result[:,:,0] *= avg_gray / (avg_b + eps)
    result[:,:,1] *= avg_gray / (avg_g + eps)
    result[:,:,2] *= avg_gray / (avg_r + eps)
    return np.clip(result, 0, 255).astype(np.uint8)

def lift_shadows_preserve_highlights_bgr(img_bgr, strength=0.34):
    """
    Lift shadowed regions more than highlights via an S-curve in Lab L channel.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32) / 255.0
    lifted = Lf + strength * (1 - np.exp(-3.0 * Lf))
    L_out = (np.clip(lifted, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L_out, A, B]), cv2.COLOR_LAB2BGR)

def local_contrast_bgr(img_bgr, clip_limit=1.9, tiles=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tiles)
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

def gentle_skin_balance_bgr(img_bgr, reduce_red=4, calm_yellow=2):
    """
    Ease redness/yellowness gently in LAB using subtraction to avoid uint8 underflow.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    A = cv2.subtract(A, np.full_like(A, reduce_red, dtype=np.uint8))
    B = cv2.subtract(B, np.full_like(B, calm_yellow, dtype=np.uint8))
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

def mild_sharpen_bgr(img_bgr, amount=0.55, radius=1):
    blur = cv2.GaussianBlur(img_bgr, (radius*2+1, radius*2+1), 0)
    return cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)

def suppress_color_noise_bgr(img_bgr, strength=6):
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, strength, strength, 7, 21)

def tone_pipeline(pil_img, bright_strength=0.34, contrast_clip=1.9, sharpen_amt=0.55):
    """
    End-to-end tonal cleanup tuned for ID-style result:
    WB → shadow lift → gentle skin balance → local contrast → mild denoise → sharpen → subtle color
    """
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bgr = auto_white_balance_bgr(bgr)
    bgr = lift_shadows_preserve_highlights_bgr(bgr, strength=bright_strength)
    bgr = gentle_skin_balance_bgr(bgr, reduce_red=4, calm_yellow=2)
    bgr = local_contrast_bgr(bgr, clip_limit=contrast_clip, tiles=(8,8))
    bgr = suppress_color_noise_bgr(bgr, strength=6)
    bgr = mild_sharpen_bgr(bgr, amount=sharpen_amt, radius=1)
    out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    # Subtle finishing
    out = ImageEnhance.Color(out).enhance(1.04)
    out = out.filter(ImageFilter.UnsharpMask(radius=1, percent=45, threshold=3))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND CLEANUP (cut-out + soft edge)
# ──────────────────────────────────────────────────────────────────────────────
def soft_paste_on_bg(fg_rgba, bg_rgb=(255,255,255), feather_px=2, pad_px=16):
    """
    Place the cut-out subject on solid background with soft edges and a bit of padding.
    """
    fg_w, fg_h = fg_rgba.size
    canvas = Image.new("RGB", (fg_w + pad_px*2, fg_h + pad_px*2), bg_rgb)

    if fg_rgba.mode != "RGBA":
        fg_rgba = fg_rgba.convert("RGBA")
    r, g, b, a = fg_rgba.split()

    if feather_px > 0:
        a = a.filter(ImageFilter.GaussianBlur(radius=feather_px))

    fg_rgba = Image.merge("RGBA", (r, g, b, a))
    canvas_rgba = Image.new("RGBA", canvas.size, bg_rgb + (255,))
    canvas_rgba.paste(fg_rgba, (pad_px, pad_px), mask=fg_rgba)
    return canvas_rgba.convert("RGB")

def cutout_with_rembg(pil_img, bg_choice="white", feather_px=2):
    """
    Remove background and place on selected clean background.
    """
    if bg_choice == "white":
        bg = (255, 255, 255)
    elif bg_choice == "blue":
        bg = (200, 224, 255)  # light ID blue
    else:
        bg = (244, 246, 249)  # near-white default

    cut = rembg_remove(pil_img.convert("RGBA"))
    out = soft_paste_on_bg(cut, bg_rgb=bg, feather_px=feather_px, pad_px=18)
    # Micro smoothing of edge halo
    out = out.filter(ImageFilter.GaussianBlur(radius=0.3))
    return out

def grabcut_fallback(pil_img, bg_choice="white", feather_px=2):
    """
    Fallback background cleanup using GrabCut when rembg is not available.
    Uses center-rect initialization; not as precise, but acceptable.
    """
    if bg_choice == "white":
        bg = (255, 255, 255)
    elif bg_choice == "blue":
        bg = (200, 224, 255)
    else:
        bg = (244, 246, 249)

    img = pil_img.convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    mask = np.zeros(bgr.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    h, w = mask.shape
    rect = (int(w*0.12), int(h*0.12), int(w*0.76), int(h*0.76))
    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = mask2
    cut = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

    out = soft_paste_on_bg(cut, bg_rgb=bg, feather_px=feather_px, pad_px=18)
    out = out.filter(ImageFilter.GaussianBlur(radius=0.3))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("🪪 ID‑Style Portrait Cleaner")
st.caption("Clean background • Natural bright tone • Centered head‑and‑shoulders • Print‑ready export")

with st.sidebar:
    st.header("⚙️ Settings")

    uploaded = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])

    st.markdown("### 🎨 Background")
    bg_choice = st.selectbox("Choose background", ["near-white", "white", "blue"], index=0)
    feather_px = st.slider("Edge feather (px)", 0, 6, 2, 1)

    st.markdown("### 🧭 Framing")
    do_crop = st.checkbox("Face‑aware portrait crop (3:4)", value=True)

    st.markdown("### 🪄 Tone")
    shadow_lift = st.slider("Shadow lift", 0.0, 0.7, 0.34, 0.01)
    clahe_clip  = st.slider("Local contrast (CLAHE)", 1.2, 3.0, 1.9, 0.1)
    sharpen_amt = st.slider("Sharpen", 0.0, 1.2, 0.55, 0.05)

    st.markdown("### 📐 Export")
    size_opt = st.selectbox("Final size", ["As‑is", "35×45 mm", "2×2 in"], index=0)
    dpi = st.number_input("DPI", value=300, min_value=150, max_value=600, step=50)

    run = st.button("✨ Enhance", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN ACTION
# ──────────────────────────────────────────────────────────────────────────────
if uploaded and run:
    try:
        original = Image.open(uploaded)
        original = ImageOps.exif_transpose(original).convert("RGB")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            st.image(original, use_container_width=True)
            st.caption(f"Size: {original.size[0]}×{original.size[1]}")

        with c2:
            st.subheader("Result")

            # 1) Framing (3:4 head-and-shoulders)
            work = original.copy()
            if do_crop:
                work = face_focus_crop(work, target_aspect=3/4, pad=0.22)

            # 2) Tonal cleanup (natural bright)
            work = tone_pipeline(work,
                                 bright_strength=shadow_lift,
                                 contrast_clip=clahe_clip,
                                 sharpen_amt=sharpen_amt)

            # 3) Background cleanup (preferred: rembg)
            if HAS_REMBG:
                work = cutout_with_rembg(work, bg_choice=bg_choice, feather_px=feather_px)
            else:
                work = grabcut_fallback(work, bg_choice=bg_choice, feather_px=feather_px)
                st.info("Note: Using GrabCut fallback. Install 'rembg' for cleaner edges.")

            # 4) Export size
            if size_opt != "As‑is":
                if size_opt == "35×45 mm":
                    work = place_on_canvas(work, target_mm=(35, 45), dpi=dpi, bg=(255,255,255))
                elif size_opt == "2×2 in":
                    work = place_on_canvas(work, target_mm=(50.8, 50.8), dpi=dpi, bg=(255,255,255))

            st.image(work, use_container_width=True)
            st.success("✅ Done")

            # Download (high-quality JPEG)
            jpeg_bytes = save_jpeg_bytes(work, quality=95, subsampling="4:4:4")
            st.download_button("📥 Download", data=jpeg_bytes, file_name="portrait_clean.jpg",
                                mime="image/jpeg", use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload a photo and click **Enhance** to generate the clean ID‑style portrait.")
    st.markdown("---")
    st.markdown(
        "Tips: Use a well‑lit photo, face towards the camera, avoid strong color tints in the room. "
        "Keep **Face‑aware crop** ON and **Background = near‑white/white** for the cleanest look."
    )
