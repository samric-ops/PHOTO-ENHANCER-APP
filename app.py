import streamlit as st
import requests
import io
import base64
import re
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2

# Optional background removal
try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except Exception:
    HAS_REMBG = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Photo Enhancer", page_icon="📸", layout="wide")

# --- GEMINI API CONFIG ---
GEMINI_MODELS = {
    "Gemini 2.5 Flash (Latest)": "gemini-2.5-flash",
    "Gemini 1.5 Pro (Best Quality)": "gemini-1.5-pro",
    "Gemini 1.5 Flash (Balanced)": "gemini-1.5-flash"
}

# ------------------------------
# Utility: Safe JPEG export (studio)
# ------------------------------
def save_jpeg_bytes(pil_img, quality=95, subsampling="4:4:4"):
    """
    Exports JPEG with high quality and 4:4:4 subsampling to reduce color banding.
    Returns bytes.
    """
    buf = io.BytesIO()
    pil_img = pil_img.convert("RGB")
    subsampling_val = 0 if subsampling == "4:4:4" else "keep"
    pil_img.save(buf, format="JPEG", quality=quality, subsampling=subsampling_val, optimize=True)
    return buf.getvalue()

# ------------------------------
# Face detection and framing
# ------------------------------
def detect_faces_bboxes(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    return faces

def face_focus_crop(pil_img, target_aspect=3/4, pad=0.18):
    """
    Crop around the largest face to portrait 3:4 aspect with padding.
    """
    img_w, img_h = pil_img.size
    faces = detect_faces_bboxes(pil_img)
    if len(faces) == 0:
        return pil_img

    faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    (x, y, w, h) = faces_sorted[0]

    cx = x + w/2
    cy = y + h/2

    box_h = h * (1 + pad*3.0)
    box_w = box_h * target_aspect
    box_w = max(box_w, w * (1 + pad*2.0))

    left = int(max(0, cx - box_w/2))
    top = int(max(0, cy - h*0.6))
    right = int(min(img_w, left + box_w))
    bottom = int(min(img_h, top + box_h))

    crop_w = right - left
    crop_h = bottom - top
    current_aspect = crop_w / max(1, crop_h)
    if current_aspect > target_aspect:
        new_w = int(crop_h * target_aspect)
        dx = (crop_w - new_w)//2
        left += dx
        right = left + new_w
    else:
        new_h = int(crop_w / target_aspect)
        dy = (crop_h - new_h)//2
        top += dy
        bottom = top + new_h

    left, top, right, bottom = map(lambda v: int(np.clip(v, 0, None)), (left, top, right, bottom))
    return pil_img.crop((left, top, right, bottom))

# ------------------------------
# Gemini direct enhancement (text-friendly; parses base64 if present)
# ------------------------------
def enhance_with_gemini_direct(image: Image.Image, api_key: str, model_name: str, instructions: str):
    """
    Tries to get an enhanced image from Gemini.
    - Removes response_mime_type to avoid 400 errors.
    - Parses inline_data if present.
    - If only text is returned, looks for a data URL or raw base64.
    Returns PIL.Image or None.
    """
    try:
        buffered = io.BytesIO()
        image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=100)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

        prompt = f"""
You are an expert studio photo retoucher. Apply ONLY the requested edits, preserving identity and natural texture.

Instructions:
{instructions}

Hard requirements:
- If possible, return the ENHANCED IMAGE as base64 (prefer a data URL format like: data:image/jpeg;base64,....).
- If you cannot return image bytes, respond with a single line JSON object: {{"note":"no_image_capability"}}.
"""

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 2048
            }
        }

        response = requests.post(
            f"{api_url}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text[:200]}")
            return None

        result = response.json()

        # 1) Try standard inline_data
        if 'candidates' in result and len(result['candidates']) > 0:
            parts = result['candidates'][0].get('content', {}).get('parts', [])

            for part in parts:
                if 'inline_data' in part and 'data' in part['inline_data']:
                    try:
                        img_data = base64.b64decode(part['inline_data']['data'])
                        return Image.open(io.BytesIO(img_data)).convert("RGB")
                    except Exception:
                        pass

            # 2) Parse base64 from text parts
            text_blobs = []
            for part in parts:
                if 'text' in part and isinstance(part['text'], str):
                    text_blobs.append(part['text'])
            combined = "\n".join(text_blobs).strip()

            if combined:
                # data URL
                m = re.search(r'data:image/(?:jpeg|jpg|png);base64,([A-Za-z0-9+/=]+)', combined, re.IGNORECASE)
                if m:
                    try:
                        img_data = base64.b64decode(m.group(1))
                        return Image.open(io.BytesIO(img_data)).convert("RGB")
                    except Exception:
                        pass

                # longest base64-ish chunk
                candidates = re.findall(r'([A-Za-z0-9+/=\s]{200,})', combined)
                candidates = [c.replace("\n", "").replace(" ", "") for c in candidates]
                candidates = sorted(candidates, key=len, reverse=True)
                for c in candidates[:3]:
                    try:
                        img_data = base64.b64decode(c)
                        return Image.open(io.BytesIO(img_data)).convert("RGB")
                    except Exception:
                        continue

        return None

    except Exception as e:
        st.error(f"Gemini error: {str(e)}")
        return None

# ------------------------------
# Local PRO-grade fallback pipeline
# ------------------------------
def auto_white_balance_bgr(img_bgr):
    result = img_bgr.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(result[:,:,0]), np.mean(result[:,:,1]), np.mean(result[:,:,2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    scale_b = avg_gray / (avg_b + 1e-6)
    scale_g = avg_gray / (avg_g + 1e-6)
    scale_r = avg_gray / (avg_r + 1e-6)
    result[:,:,0] *= scale_b
    result[:,:,1] *= scale_g
    result[:,:,2] *= scale_r
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def lift_shadows_preserve_highlights_bgr(img_bgr, strength=0.35):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32) / 255.0
    lifted = Lf + strength * (1 - np.exp(-3.0 * Lf))
    lifted = np.clip(lifted, 0, 1)
    L_out = (lifted * 255).astype(np.uint8)
    lab = cv2.merge([L_out, A, B])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def local_contrast_bgr(img_bgr, clip_limit=2.0, tile_grid=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    L = clahe.apply(L)
    merged = cv2.merge([L, A, B])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def gentle_skin_tone_balance_bgr(img_bgr, reduce_red=5, calm_yellow=3):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    A = cv2.add(A, np.full_like(A, -reduce_red))
    B = cv2.add(B, np.full_like(B, -calm_yellow))
    merged = cv2.merge([L, A, B])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def mild_sharpen_bgr(img_bgr, amount=0.7, radius=1):
    blur = cv2.GaussianBlur(img_bgr, (radius*2+1, radius*2+1), 0)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
    return sharp

def eye_teeth_pop(pil_img, face_boxes, lift=0.08):
    """
    Subtle brightness lift around eyes/teeth area.
    """
    if len(face_boxes) == 0:
        return pil_img
    img = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.float32)

    for (x, y, fw, fh) in face_boxes:
        eye_y1 = int(y + fh*0.25)
        eye_y2 = int(y + fh*0.55)
        x1 = int(x + fw*0.15)
        x2 = int(x + fw*0.85)
        region = np.zeros_like(mask)
        region[eye_y1:eye_y2, x1:x2] = 1.0
        region = cv2.GaussianBlur(region, (0,0), max(1, int(fh*0.08)))
        mask = np.maximum(mask, region)

    mask = np.clip(mask, 0, 1) * lift
    img[:,:,0] = np.clip(img[:,:,0] + mask, 0, 1)
    img[:,:,1] = np.clip(img[:,:,1] + mask, 0, 1)
    img[:,:,2] = np.clip(img[:,:,2] + mask, 0, 1)

    img = (img*255).astype(np.uint8)
    return Image.fromarray(img)

def background_cleanup(pil_img, mode="keep", bg_color=(245,245,245)):
    """
    mode: "keep" (no change), "clean" (denoise background), "remove" (cut subject, replace with solid bg)
    """
    if mode == "keep":
        return pil_img

    if mode == "clean":
        img = np.array(pil_img.convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        clean = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=40, sigmaSpace=40)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        return Image.fromarray(clean)

    if mode == "remove":
        if not HAS_REMBG:
            st.warning("Background removal not available (rembg not installed).")
            return pil_img
        cut = rembg_remove(pil_img.convert("RGBA"))
        bg = Image.new("RGBA", cut.size, bg_color + (255,))
        out = Image.alpha_composite(bg, cut)
        return out.convert("RGB")

    return pil_img

def enhance_with_gemini_fallback(image: Image.Image, instructions: str,
                                 strength_bright=0.35, contrast_clip=2.0,
                                 sharpen_amt=0.6, bg_mode="keep", face_crop=False):
    """
    PRO-grade local pipeline for reliable studio-like portrait enhancement.
    """
    img = ImageOps.exif_transpose(image).convert("RGB")
    if face_crop:
        img = face_focus_crop(img, target_aspect=3/4, pad=0.20)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    bgr = auto_white_balance_bgr(bgr)
    bgr = lift_shadows_preserve_highlights_bgr(bgr, strength=strength_bright)
    bgr = gentle_skin_tone_balance_bgr(bgr, reduce_red=5, calm_yellow=3)
    bgr = local_contrast_bgr(bgr, clip_limit=contrast_clip, tile_grid=(8,8))
    bgr = mild_sharpen_bgr(bgr, amount=sharpen_amt, radius=1)

    out = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    faces = detect_faces_bboxes(out)
    out = eye_teeth_pop(out, faces, lift=0.07)

    out = background_cleanup(out, mode=bg_mode, bg_color=(245,245,245))

    out = ImageEnhance.Color(out).enhance(1.06)
    out = out.filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=3))

    return out

# ------------------------------
# UI
# ------------------------------
st.title("📸 AI Photo Enhancer")
st.markdown("### ✨ Gemini AI + Pro Local Pipeline for Studio-Grade Results")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password",
                            help="Get from https://makersuite.google.com/app/apikey")
    selected_model = st.selectbox("Gemini Model", list(GEMINI_MODELS.keys()), index=0)
    model_name = GEMINI_MODELS[selected_model]

    uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "png", "jpeg"])

    st.markdown("---")
    st.subheader("🎨 Enhancement Style")
    style = st.selectbox(
        "Select Style",
        [
            "Ultra Bright ID Photo",
            "Professional Studio Portrait",
            "Natural Bright",
            "Custom"
        ]
    )
    if style == "Custom":
        custom_instructions = st.text_area(
            "Custom Instructions",
            value="Bright, even studio lighting on face, natural skin tones, remove facial shadows, preserve texture, crisp details."
        )
    else:
        if style == "Ultra Bright ID Photo":
            instructions = ("Make this an ultra bright ID portrait. Face should be very bright and evenly lit, "
                            "no harsh shadows, natural skin tone, clean neutral background feel, sharp details. "
                            "Preserve identity and natural texture.")
        elif style == "Professional Studio Portrait":
            instructions = ("Enhance into a professional studio portrait: balanced key+fill lighting, bright and clear face, "
                            "natural skin tone, remove minor blemishes, preserve pores, gentle contrast and clarity.")
        elif style == "Natural Bright":
            instructions = ("Brighten the face while keeping it natural. Lift shadows softly, maintain skin texture, "
                            "avoid over-smoothing, gentle contrast and color balance.")

    st.markdown("---")
    st.subheader("🪄 Pro Controls")
    face_crop = st.checkbox("Face-aware portrait crop (3:4)", value=(style in ["Ultra Bright ID Photo", "Professional Studio Portrait"]))
    bg_mode = st.selectbox("Background", ["keep", "clean", "remove"], index=0)
    bright_strength = st.slider("Shadow lift / Brightness", 0.0, 0.7, 0.35, 0.01)
    contrast_clip = st.slider("Local contrast (CLAHE clip)", 1.0, 3.0, 2.0, 0.1)
    sharpen_amt = st.slider("Sharpen amount", 0.0, 1.2, 0.6, 0.05)

    enhance_btn = st.button("✨ Enhance with AI", type="primary", use_container_width=True)

# ------------------------------
# Main
# ------------------------------
if uploaded_file and enhance_btn:
    try:
        original = Image.open(uploaded_file)
        original = ImageOps.exif_transpose(original)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original")
            st.image(original, use_container_width=True)
            st.caption(f"Size: {original.size[0]}×{original.size[1]}")

        with col2:
            st.subheader("✨ AI Enhanced")

            if style == "Custom":
                inst = custom_instructions
            else:
                inst = instructions

            enhanced = None
            used_gemini = False

            if api_key:
                with st.spinner(f"🤖 {selected_model} (Gemini) is enhancing your photo..."):
                    enhanced = enhance_with_gemini_direct(original, api_key, model_name, inst)
                    if enhanced is not None:
                        used_gemini = True

            if enhanced is None:
                with st.spinner("Applying local pro-grade enhancement..."):
                    enhanced = enhance_with_gemini_fallback(
                        original, inst,
                        strength_bright=bright_strength,
                        contrast_clip=contrast_clip,
                        sharpen_amt=sharpen_amt,
                        bg_mode=bg_mode,
                        face_crop=face_crop
                    )
                st.info("ℹ️ Gemini model/endpoint returned text only or no image. Using built-in pro-grade enhancement.")

            st.image(enhanced, use_container_width=True)
            if used_gemini:
                st.success("✅ Enhanced by Gemini AI!")
            else:
                st.success("✅ Enhanced by Pro Local Pipeline")

            jpeg_bytes = save_jpeg_bytes(enhanced, quality=95, subsampling="4:4:4")
            st.download_button(
                label="📥 Download Enhanced Photo",
                data=jpeg_bytes,
                file_name="ai_enhanced_photo.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("👆 Upload a photo and click 'Enhance with AI'")

    st.markdown("---")
    st.markdown("### How to get the BEST results:")
    st.markdown("""
1. **Get a free API key** from https://makersuite.google.com/app/apikey  
2. **Upload a clear, well-exposed photo** (avoid heavy motion blur)  
3. **Select a style**:
   - **Ultra Bright ID Photo** – sobrang linis at maliwanag, pang-ID
   - **Professional Studio Portrait** – parang studio shot
   - **Natural Bright** – maliwanag pero natural ang dating  
4. **Optional**: Turn on **Face-aware crop** and **Background clean/remove**  
5. **Download** the result as high-quality JPEG (4:4:4, Q95)
""")
