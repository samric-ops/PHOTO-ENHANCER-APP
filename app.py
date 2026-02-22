# Auto Portrait Cleaner (Cloud-First, No Controls)
# - Auto crop (3:4 head-and-shoulders)
# - Natural light correction (no green tint), soft shadow reduction
# - Clean background (near-white by default)
# - Uses cloud AI if keys exist (Clipdrop Relight + Remove BG / Clipdrop Remove BG)
# - Falls back to local pipeline (OpenCV + rembg) if no keys

import streamlit as st
import io, os, time
import requests
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# Optional background removal (local)
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    HAS_REMBG = False

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Auto Portrait Cleaner (Cloud-First)", page_icon="🪪", layout="centered")
st.title("🪪 Auto Portrait Cleaner")
st.caption("Automatic lighting + clean background • Uses cloud AI if keys are set; otherwise local pipeline")

# Load keys from Streamlit Secrets (recommended)
CLIPDROP_API_KEY = st.secrets.get("CLIPDROP_API_KEY", os.getenv("CLIPDROP_API_KEY", ""))
REMOVEBG_API_KEY = st.secrets.get("REMOVEBG_API_KEY", os.getenv("REMOVEBG_API_KEY", ""))

# Background color (near-white)
BG_RGB = (244, 246, 249)

# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
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

def to_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)

def merge_u8(chs):
    chs = [to_u8(c) for c in chs]
    h, w = chs[0].shape[:2]
    fixed = []
    for c in chs:
        if c.shape[:2] != (h, w):
            c = cv2.resize(c, (w, h), interpolation=cv2.INTER_NEAREST)
        fixed.append(c)
    return cv2.merge(fixed)

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
        left += dx; right = left + new_w
    else:
        new_h = int(cw / target_aspect)
        dy = (ch - new_h)//2
        top += dy; bottom = top + new_h
    left, top, right, bottom = map(int, [max(0,left), max(0,top), min(w,right), min(h,bottom)])
    return pil_img.crop((left, top, right, bottom))

# ──────────────────────────────────────────────────────────────────────────────
# AUTO COLOR / LIGHTING — natural, anti-green, soft shadows
# ──────────────────────────────────────────────────────────────────────────────
def shades_of_gray_wb_bgr(img_bgr, p=6):
    img = img_bgr.astype(np.float32); eps = 1e-6
    r = np.power(img[:,:,2], p).mean() ** (1.0/p)
    g = np.power(img[:,:,1], p).mean() ** (1.0/p)
    b = np.power(img[:,:,0], p).mean() ** (1.0/p)
    avg = (r + g + b) / 3.0
    img[:,:,2] *= (avg/(r+eps)); img[:,:,1] *= (avg/(g+eps)); img[:,:,0] *= (avg/(b+eps))
    return to_u8(img)

def neutralize_tint_lab(img_bgr, limit=6):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    mA, mB = float(np.median(A)), float(np.median(B))
    dA = int(np.clip(round(mA - 128), -limit, limit))
    dB = int(np.clip(round(mB - 128), -limit, limit))
    if dA != 0: A = cv2.subtract(A, np.full_like(A, dA, dtype=np.uint8))
    if dB != 0: B = cv2.subtract(B, np.full_like(B, dB, dtype=np.uint8))
    return cv2.cvtColor(merge_u8([L, A, B]), cv2.COLOR_LAB2BGR)

def retinex_msr_bgr(img_bgr, sigma_list=(15, 80, 250), gain=0.28):
    img = img_bgr.astype(np.float32) + 1.0
    log_img = np.log(img)
    msr = np.zeros_like(img, dtype=np.float32)
    w = 1.0/len(sigma_list)
    for s in sigma_list:
        blur = cv2.GaussianBlur(img, (0,0), sigmaX=s, sigmaY=s)
        msr += w * (log_img - np.log(blur + 1.0))
    msr = img * (1.0 + gain * msr)
    return to_u8(msr)

def local_contrast_bgr(img_bgr, clip=1.5):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    L = clahe.apply(L)
    return cv2.cvtColor(merge_u8([L, A, B]), cv2.COLOR_LAB2BGR)

def skin_mask_ycrcb(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    return cv2.GaussianBlur(mask, (7,7), 0)

def soften_shadows_on_skin(img_bgr, strength=0.12):
    mask = (skin_mask_ycrcb(img_bgr) / 255.0).astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32) / 255.0
    lift = Lf + strength * (1 - np.exp(-3.0 * Lf))
    L_out = to_u8(np.clip(Lf*(1-mask) + lift*mask, 0, 1) * 255.0)
    return cv2.cvtColor(merge_u8([L_out, A, B]), cv2.COLOR_LAB2BGR)

def green_spill_suppression(rgb_img, alpha=None, strength=0.6):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)  # uint8
    green_mask = cv2.inRange(H, 40, 90).astype(np.float32) / 255.0
    if alpha is not None:
        a = alpha.astype(np.float32) / 255.0
        edge = cv2.Canny((a*255).astype(np.uint8), 30, 80)
        edge = cv2.dilate(edge, np.ones((3,3), np.uint8), iterations=1).astype(np.float32) / 255.0
        green_mask = np.clip(green_mask + edge*0.7, 0, 1)
    S_adj = np.clip(S.astype(np.float32) * (1 - 0.35*strength*green_mask), 0, 255).astype(np.uint8)
    hsv_suppressed = merge_u8([H, S_adj, V])
    out = cv2.cvtColor(hsv_suppressed, cv2.COLOR_HSV2RGB).astype(np.float32)
    g = out[:,:,1]; r = out[:,:,0]; b = out[:,:,2]
    out[:,:,1] = np.clip(g*(1 - 0.18*strength) + 0.09*strength*(r+b), 0, 255)
    return to_u8(out)

# ──────────────────────────────────────────────────────────────────────────────
# CLOUD ADAPTERS
# ──────────────────────────────────────────────────────────────────────────────
def clipdrop_remove_bg(pil_img):
    # Endpoint per Clipdrop Remove Background API
    # Docs/overview pages: https://clipdrop.co/apis ; pricing tiers show API availability.
    # Relighting endpoint reference on /relight docs.
    url = "https://clipdrop-api.co/remove-background/v1"
    headers = {"x-api-key": CLIPDROP_API_KEY}
    buf = io.BytesIO(); pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
    files = {"image_file": ("img.jpg", buf.getvalue(), "image/jpeg")}
    r = requests.post(url, headers=headers, files=files, timeout=60)
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGBA")
    raise RuntimeError(f"Clipdrop remove-bg error {r.status_code}: {r.text[:200]}")

def clipdrop_relight(pil_img):
    # Portrait relight API (surface normals endpoint)
    url = "https://clipdrop-api.co/portrait-surface-normals/v1"
    headers = {"x-api-key": CLIPDROP_API_KEY}
    buf = io.BytesIO(); pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
    files = {"image_file": ("img.jpg", buf.getvalue(), "image/jpeg")}
    r = requests.post(url, headers=headers, files=files, timeout=60)
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    raise RuntimeError(f"Clipdrop relight error {r.status_code}: {r.text[:200]}")

def removebg_cutout(pil_img):
    # remove.bg API (50 free low-res calls/month; scale up for HD as needed)
    url = "https://api.remove.bg/v1.0/removebg"
    headers = {"X-Api-Key": REMOVEBG_API_KEY}
    buf = io.BytesIO(); pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
    files = {"image_file": ("img.jpg", buf.getvalue(), "image/jpeg")}
    data = {"size": "auto"}  # let API choose best
    r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGBA")
    raise RuntimeError(f"remove.bg error {r.status_code}: {r.text[:200]}")

# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND COMPOSITOR
# ──────────────────────────────────────────────────────────────────────────────
def soft_paste_on_bg(fg_rgba: Image.Image, bg_rgb=(244,246,249), feather_px=2, pad_px=18):
    W, H = fg_rgba.size
    canvas = Image.new("RGB", (W + pad_px*2, H + pad_px*2), bg_rgb)
    if fg_rgba.mode != "RGBA":
        fg_rgba = fg_rgba.convert("RGBA")
    r,g,b,a = fg_rgba.split()
    if feather_px > 0:
        a = a.filter(ImageFilter.GaussianBlur(radius=feather_px))
    # Edge-aware green de-spill near alpha edges
    arr = np.stack([np.array(r), np.array(g), np.array(b)], axis=-1)
    alpha = np.array(a)
    arr = green_spill_suppression(arr, alpha=alpha, strength=0.60)
    fg_rgba2 = Image.merge("RGBA", (
        Image.fromarray(arr[:,:,0]), Image.fromarray(arr[:,:,1]), Image.fromarray(arr[:,:,2]), a
    ))
    comp_rgba = Image.new("RGBA", (canvas.size[0], canvas.size[1]), bg_rgb + (255,))
    comp_rgba.paste(fg_rgba2, (pad_px, pad_px), mask=fg_rgba2)
    return comp_rgba.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.25))

# ──────────────────────────────────────────────────────────────────────────────
# AUTO PIPELINES
# ──────────────────────────────────────────────────────────────────────────────
def local_auto_pipeline(pil_img):
    """All-local automatic clean-up (no API keys)."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    img = face_focus_crop(img, target_aspect=3/4, pad=0.22)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bgr = retinex_msr_bgr(bgr, gain=0.28)
    bgr = shades_of_gray_wb_bgr(bgr, p=6)
    bgr = neutralize_tint_lab(bgr, limit=6)
    bgr = local_contrast_bgr(bgr, clip=1.5)
    bgr = soften_shadows_on_skin(bgr, strength=0.12)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = green_spill_suppression(rgb, alpha=None, strength=0.60)
    bgr2 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr2 = cv2.fastNlMeansDenoisingColored(bgr2, None, 2, 2, 7, 21)
    blur = cv2.GaussianBlur(bgr2, (3,3), 0)
    bgr2 = cv2.addWeighted(bgr2, 1.30, blur, -0.30, 0)
    img2 = Image.fromarray(cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB))
    img2 = ImageEnhance.Color(img2).enhance(1.02)
    # Local background removal
    if HAS_REMBG:
        cut = rembg_remove(img2.convert("RGBA"))
        out = soft_paste_on_bg(cut, bg_rgb=BG_RGB, feather_px=2, pad_px=18)
    else:
        out = Image.new("RGB", img2.size, BG_RGB); out.paste(img2, (0,0))
    # Lock to 35x45 mm @ 300 DPI
    out = place_on_canvas(out, target_mm=(35,45), dpi=300, bg=(255,255,255), margin_px=12)
    return out

def cloud_auto_pipeline(pil_img):
    """
    Cloud-first path:
      1) 3:4 crop + gentle natural correction locally
      2) Clipdrop Relight (if key)
      3) Background removal (remove.bg preferred else Clipdrop)
      4) Compose on near-white + finalize canvas
    """
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    img = face_focus_crop(img, target_aspect=3/4, pad=0.22)

    # light local normalization first (keeps output natural)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    bgr = shades_of_gray_wb_bgr(bgr, p=6)
    bgr = neutralize_tint_lab(bgr, limit=5)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_norm = Image.fromarray(rgb)

    # 2) Clipdrop Relight if available
    img_relit = img_norm
    if CLIPDROP_API_KEY:
        try:
            img_relit = clipdrop_relight(img_norm)
        except Exception as e:
            # fallback to normalized image
            img_relit = img_norm

    # 3) Background removal (prefer remove.bg if key)
    cut = None
    if REMOVEBG_API_KEY:
        try:
            cut = removebg_cutout(img_relit)
        except Exception as e:
            cut = None
    if cut is None and CLIPDROP_API_KEY:
        try:
            cut = clipdrop_remove_bg(img_relit)
        except Exception as e:
            cut = None

    if cut is None:
        # fall back to local cutout if cloud fails
        if HAS_REMBG:
            cut = rembg_remove(img_relit.convert("RGBA"))
        else:
            # last resort: paste without cutout
            back = Image.new("RGB", img_relit.size, BG_RGB)
            back.paste(img_relit, (0,0))
            out = back
            return place_on_canvas(out, target_mm=(35,45), dpi=300, bg=(255,255,255), margin_px=12)

    # 4) Composite on near-white with edge de-spill
    out = soft_paste_on_bg(cut, bg_rgb=BG_RGB, feather_px=2, pad_px=18)
    out = ImageEnhance.Color(out).enhance(1.02)
    # Lock to 35x45 mm @ 300 DPI
    out = place_on_canvas(out, target_mm=(35,45), dpi=300, bg=(255,255,255), margin_px=12)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# UI (no sliders)
# ──────────────────────────────────────────────────────────────────────────────
file = st.file_uploader("Upload Photo", type=["jpg","jpeg","png"])
go = st.button("✨ Auto-Enhance", type="primary", use_container_width=True)

if file and go:
    try:
        original = Image.open(file)
        with st.spinner("Optimizing lighting and background..."):
            use_cloud = bool(CLIPDROP_API_KEY or REMOVEBG_API_KEY)
            result = cloud_auto_pipeline(original) if use_cloud else local_auto_pipeline(original)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            st.image(ImageOps.exif_transpose(original), use_container_width=True)
        with c2:
            st.subheader("Result")
            st.image(result, use_container_width=True)
            st.download_button(
                "📥 Download",
                data=save_jpeg_bytes(result, quality=95, subsampling="4:4:4"),
                file_name="portrait_clean_auto.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
        st.success("✅ Done")
        if not (CLIPDROP_API_KEY or REMOVEBG_API_KEY):
            st.info("Running in local mode. Add API keys in Streamlit Secrets to enable cloud AI.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
elif not file:
    st.info("Upload a photo then click **Auto‑Enhance**.")
