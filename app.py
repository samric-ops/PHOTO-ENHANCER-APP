# Natural Classroom ID Portrait — tuned for green cast (classroom walls) + soft shadow removal
# 100% local processing. Clean background (white / near-white / blue). 3:4 head-shoulders crop.

import streamlit as st
import io
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2

# Optional background removal
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    HAS_REMBG = False

# ──────────────────────────────────────────────────────────────────────────────
# APP CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Natural Classroom ID Portrait", page_icon="🪪", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE / EXPORT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def save_jpeg_bytes(pil_img, quality=95, subsampling="4:4:4"):
    buf = io.BytesIO()
    pil_img = pil_img.convert("RGB")
    ss_map = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
    pil_img.save(buf, format="JPEG", quality=quality, subsampling=ss_map.get(subsampling, 0), optimize=True)
    return buf.getvalue()

def mm_to_pixels(mm, dpi=300):
    return int(round((mm/25.4)*dpi))

def place_on_canvas(img: Image.Image, target_mm=(35,45), dpi=300, bg=(255,255,255), margin_px=12):
    tw, th = mm_to_pixels(target_mm[0], dpi), mm_to_pixels(target_mm[1], dpi)
    canvas = Image.new("RGB", (tw, th), bg)
    iw, ih = img.size
    scale = min((tw - margin_px*2)/iw, (th - margin_px*2)/ih)
    nw, nh = max(1, int(iw*scale)), max(1, int(ih*scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas.paste(resized, ((tw-nw)//2, (th-nh)//2))
    return canvas

# ──────────────────────────────────────────────────────────────────────────────
# FACE & FRAMING (3:4 portrait crop)
# ──────────────────────────────────────────────────────────────────────────────
def detect_faces_bboxes(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    return faces

def face_focus_crop(pil_img, target_aspect=3/4, pad=0.22):
    """Crop to 3:4 around the largest face with shoulder/hairline margin."""
    w, h = pil_img.size
    faces = detect_faces_bboxes(pil_img)
    if len(faces) == 0:
        return pil_img
    (x, y, fw, fh) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    cx, cy = x + fw/2, y + fh/2

    box_h = fh * (1 + pad*3.0)
    box_w = max(fw * (1 + pad*2.2), box_h * target_aspect)

    left = int(max(0, cx - box_w/2))
    top  = int(max(0, cy - fh*0.70))
    right = int(min(w, left + box_w))
    bottom = int(min(h, top + box_h))

    cw, ch = right - left, bottom - top
    cur_aspect = cw / max(1, ch)
    if cur_aspect > target_aspect:
        new_w = int(ch * target_aspect)
        dx = (cw - new_w)//2
        left += dx
        right = left + new_w
    else:
        new_h = int(cw / target_aspect)
        dy = (ch - new_h)//2
        top += dy
        bottom = top + new_h

    left, top, right, bottom = map(int, [max(0,left), max(0,top), min(w,right), min(h,bottom)])
    return pil_img.crop((left, top, right, bottom))

# ──────────────────────────────────────────────────────────────────────────────
# COLOR / LIGHTING — natural look, anti‑green, soft shadows
# ──────────────────────────────────────────────────────────────────────────────
def shades_of_gray_wb_bgr(img_bgr, p=6):
    """Shades-of-Gray WB (p-norm): reliable sa indoor casts."""
    img = img_bgr.astype(np.float32); eps = 1e-6
    r = np.power(img[:,:,2], p).mean() ** (1.0/p)
    g = np.power(img[:,:,1], p).mean() ** (1.0/p)
    b = np.power(img[:,:,0], p).mean() ** (1.0/p)
    avg = (r + g + b) / 3.0
    img[:,:,2] *= (avg/(r+eps)); img[:,:,1] *= (avg/(g+eps)); img[:,:,0] *= (avg/(b+eps))
    return np.clip(img, 0, 255).astype(np.uint8)

def neutralize_tint_lab(img_bgr, limit=6):
    """Center LAB A/B medians gently toward 128 (clamped to keep it natural)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mA, mB = float(np.median(A)), float(np.median(B))
    dA = int(np.clip(round(mA - 128), -limit, limit))
    dB = int(np.clip(round(mB - 128), -limit, limit))
    if dA != 0: A = cv2.subtract(A, np.full_like(A, dA, dtype=np.uint8))
    if dB != 0: B = cv2.subtract(B, np.full_like(B, dB, dtype=np.uint8))
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

def retinex_msr_bgr(img_bgr, sigma_list=(15, 80, 250), gain=0.28):
    """Multi‑Scale Retinex: flatten uneven illumination (softer shadows)."""
    img = img_bgr.astype(np.float32) + 1.0
    log_img = np.log(img)
    msr = np.zeros_like(img, dtype=np.float32)
    w = 1.0/len(sigma_list)
    for s in sigma_list:
        blur = cv2.GaussianBlur(img, (0,0), sigmaX=s, sigmaY=s)
        msr += w * (log_img - np.log(blur + 1.0))
    msr = img * (1.0 + gain * msr)
    return np.clip(msr, 0, 255).astype(np.uint8)

def local_contrast_bgr(img_bgr, clip=1.4):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

def skin_mask_ycrcb(img_bgr):
    """Loose skin mask (YCrCb) para localized tweaks."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    _, Cr, Cb = cv2.split(ycrcb)
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    return mask

def soften_shadows_on_skin(img_bgr, strength=0.12):
    """Lift shadows slightly on skin only (para hindi ma-flatten ang damit/background)."""
    mask = skin_mask_ycrcb(img_bgr) / 255.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32) / 255.0
    lift = Lf + strength * (1 - np.exp(-3.0 * Lf))
    L_out = (np.clip(Lf*(1-mask) + lift*mask, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([L_out, A, B]), cv2.COLOR_LAB2BGR)

def green_spill_suppression(rgb_img, alpha=None, strength=0.6):
    """
    Targeted 'green spill' reduction (common sa classroom walls).
    • Works globally on green hues; if alpha (cutout) is provided, extra suppression near edges.
    """
    img = rgb_img.astype(np.float32)
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # Green range in OpenCV hue scale [0..179]: around 40..90
    green_mask = cv2.inRange(H, 40, 90).astype(np.float32) / 255.0
    # Edge emphasis if alpha available
    if alpha is not None:
        a = alpha.astype(np.float32) / 255.0
        edge = cv2.Canny((a*255).astype(np.uint8), 30, 80)
        edge = cv2.dilate(edge, np.ones((3,3), np.uint8), iterations=1).astype(np.float32) / 255.0
        green_mask = np.clip(green_mask + edge*0.7, 0, 1)

    # Reduce saturation on green regions
    S = np.clip(S * (1 - 0.35*strength), 0, 255)
    hsv_suppressed = cv2.merge([H, S, V])
    out = cv2.cvtColor(hsv_suppressed, cv2.COLOR_HSV2RGB).astype(np.float32)

    # Gentle channel mix to neutralize leftover G dominance
    g = out[:,:,1]
    r = out[:,:,0]
    b = out[:,:,2]
    out[:,:,1] = np.clip(g*(1 - 0.18*strength) + 0.09*strength*(r+b), 0, 255)

    return np.clip(out, 0, 255).astype(np.uint8)

def natural_classroom_pipeline(pil_img,
                               retinex_gain=0.28,
                               wb_p=6,
                               tint_limit=6,
                               clahe_clip=1.4,
                               skin_shadow=0.12,
                               sharpen=0.30,
                               green_strength=0.6):
    """
    Tuned sequence for classroom indoor photos (green walls, mixed lighting):
    Retinex → WB → Tint neutralization → Local contrast → Skin shadow soften → Green spill suppression → light sharpen + tiny color pop
    """
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 1) Even illumination (soften shadows)
    bgr = retinex_msr_bgr(bgr, gain=retinex_gain)

    # 2) Natural white balance
    bgr = shades_of_gray_wb_bgr(bgr, p=wb_p)

    # 3) Gentle tint cleanup
    bgr = neutralize_tint_lab(bgr, limit=tint_limit)

    # 4) Local contrast (gentle)
    bgr = local_contrast_bgr(bgr, clip=clahe_clip)

    # 5) Skin‑only shadow softening
    bgr = soften_shadows_on_skin(bgr, strength=skin_shadow)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 6) Green spill suppression (global; extra near edges later after cutout)
    rgb = green_spill_suppression(rgb, alpha=None, strength=green_strength)

    # 7) Mild denoise + mild sharpen (keep texture natural)
    bgr2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr2 = cv2.fastNlMeansDenoisingColored(bgr2, None, 2, 2, 7, 21)
    blur = cv2.GaussianBlur(bgr2, (3,3), 0)
    bgr2 = cv2.addWeighted(bgr2, 1.0 + sharpen, blur, -sharpen, 0)

    out = Image.fromarray(cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB))
    out = ImageEnhance.Color(out).enhance(1.02)  # very subtle
    return out

# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND CLEANUP (cut-out + soft edges + edge de-spill)
# ──────────────────────────────────────────────────────────────────────────────
def soft_paste_on_bg(fg_rgba: Image.Image, bg_rgb=(244,246,249), feather_px=2, pad_px=18, green_strength=0.6):
    W, H = fg_rgba.size
    canvas = Image.new("RGB", (W + pad_px*2, H + pad_px*2), bg_rgb)

    if fg_rgba.mode != "RGBA":
        fg_rgba = fg_rgba.convert("RGBA")
    r,g,b,a = fg_rgba.split()

    # Edge feather
    if feather_px > 0:
        a = a.filter(ImageFilter.GaussianBlur(radius=feather_px))

    # Extra green spill suppression near edges using alpha
    fg_rgb = Image.merge("RGB", (r,g,b))
    arr = np.array(fg_rgb)
    alpha = np.array(a)
    arr = green_spill_suppression(arr, alpha=alpha, strength=green_strength)

    fg_rgba2 = Image.merge("RGBA", (Image.fromarray(arr[:,:,0]),
                                    Image.fromarray(arr[:,:,1]),
                                    Image.fromarray(arr[:,:,2]),
                                    a))

    comp_rgba = Image.new("RGBA", (canvas.size[0], canvas.size[1]), bg_rgb + (255,))
    comp_rgba.paste(fg_rgba2, (pad_px, pad_px), mask=fg_rgba2)
    out = comp_rgba.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.25))  # micro-smooth
    return out

def clean_background(pil_img, bg_choice="near-white", feather_px=2, green_strength=0.6):
    if bg_choice == "white":
        bg = (255,255,255)
    elif bg_choice == "blue":
        bg = (200,224,255)  # light ID blue
    else:
        bg = (244,246,249)  # near-white

    if HAS_REMBG:
        cut = rembg_remove(pil_img.convert("RGBA"))
        return soft_paste_on_bg(cut, bg_rgb=bg, feather_px=feather_px, pad_px=18, green_strength=green_strength)
    else:
        # Simple GrabCut fallback
        bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
        h, w = mask.shape
        rect = (int(w*0.12), int(h*0.12), int(w*0.76), int(h*0.76))
        try:
            cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA); rgba[:,:,3] = mask2
            cut = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
            return soft_paste_on_bg(cut, bg_rgb=bg, feather_px=feather_px, pad_px=18, green_strength=green_strength)
        except Exception:
            back = Image.new("RGB", pil_img.size, bg); back.paste(pil_img, (0,0)); return back

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("🪪 Natural Classroom ID Portrait")
st.caption("Natural light • Anti‑green • Minimal shadows • Clean background (white / near‑white / blue)")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload Photo", type=["jpg","jpeg","png"])

    st.markdown("### 🧭 Framing")
    do_crop = st.checkbox("Face‑aware 3:4 crop (head‑and‑shoulders)", value=True)

    st.markdown("### 🌈 Background")
    bg_choice = st.selectbox("Background", ["near-white", "white", "blue"], index=0)
    feather_px = st.slider("Edge feather", 0, 6, 2, 1)

    st.markdown("### 🎛 Natural Classroom Preset Controls")
    green_strength = st.slider("Green spill suppression", 0.0, 1.0, 0.60, 0.05)
    retinex_gain   = st.slider("Shadow removal (Retinex)", 0.0, 0.6, 0.28, 0.01)
    wb_p           = st.slider("White balance p‑norm", 2, 12, 6, 1)
    tint_limit     = st.slider("Tint neutralization limit (LAB units)", 0, 12, 6, 1)
    clahe_clip     = st.slider("Local contrast (CLAHE)", 1.2, 3.0, 1.4, 0.1)
    skin_shadow    = st.slider("Skin shadow softening", 0.0, 0.4, 0.12, 0.01)
    sharpen        = st.slider("Sharpen", 0.0, 1.0, 0.30, 0.05)

    st.markdown("### 📐 Export")
    size_opt = st.selectbox("Final size", ["As‑is", "35×45 mm", "2×2 in"], index=0)
    dpi = st.number_input("DPI", value=300, min_value=150, max_value=600, step=50)

    run = st.button("✨ Enhance", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
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

            # 1) Framing
            work = original.copy()
            if do_crop:
                work = face_focus_crop(work, target_aspect=3/4, pad=0.22)

            # 2) Natural classroom tone pipeline
            work = natural_classroom_pipeline(
                work,
                retinex_gain=retinex_gain,
                wb_p=wb_p,
                tint_limit=tint_limit,
                clahe_clip=clahe_clip,
                skin_shadow=skin_shadow,
                sharpen=sharpen,
                green_strength=green_strength
            )

            # 3) Clean background with edge de-spill
            work = clean_background(work, bg_choice=bg_choice, feather_px=feather_px, green_strength=green_strength)

            # 4) Export size
            if size_opt != "As‑is":
                if size_opt == "35×45 mm":
                    work = place_on_canvas(work, target_mm=(35,45), dpi=dpi, bg=(255,255,255))
                elif size_opt == "2×2 in":
                    work = place_on_canvas(work, target_mm=(50.8,50.8), dpi=dpi, bg=(255,255,255))

            st.image(work, use_container_width=True)
            st.success("✅ Enhanced successfully")

            st.download_button("📥 Download",
                               data=save_jpeg_bytes(work, quality=95, subsampling="4:4:4"),
                               file_name="classroom_id_portrait.jpg",
                               mime="image/jpeg",
                               use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload a photo then click **Enhance**.")
    st.markdown("---")
    st.markdown(
        "Tip: Kung may greenish pa rin, **taasan ang Green spill suppression** (0.7–0.9). "
        "Kung may shadow pa, itaas ang **Retinex** (0.35–0.45) at **Skin shadow softening** (0.18–0.25). "
        "Kung masyadong contrasty, ibaba ang **CLAHE** sa 1.2–1.3."
    )
