import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from rembg import remove

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro AI ID Studio", page_icon="👤")

def pro_enhance(image):
    # Convert PIL to OpenCV
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. DENOISING (Ito ang secret ng Remini para kuminis ang balat)
    # Tinatanggal ang "grain" o "noise" lalo na sa madidilim na kuha
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # 2. GAMMA CORRECTION (Para sa "Natural" na liwanag, hindi washed out)
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 3. SMART CONTRAST (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # Convert back to PIL
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 4. FINAL SHARPENING & COLOR BOOST
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    color_enhancer = ImageEnhance.Color(pil_img)
    pil_img = color_enhancer.enhance(1.1) # Konting dagdag sa sigla ng kulay
    
    return pil_img

def process_id_photo(image, size_type, remove_bg):
    # AI Background Removal
    if remove_bg:
        image = remove(image)
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, (0, 0), image)
        image = new_image.convert('RGB')

    # Standard Sizes
    sizes = {
        "2x2 (600x600 px)": (600, 600),
        "1x1 (300x300 px)": (300, 300),
        "Passport (413x531 px)": (413, 531)
    }
    
    target_size = sizes.get(size_type, (600, 600))
    # Resizing with high-quality filter
    return image.resize(target_size, Image.Resampling.LANCZOS)

# --- UI ---
st.title("👤 Pro AI ID Photo Studio")
st.write("Inaayos ang lighting at sharpness gamit ang AI (Remini-style filters).")

# FIX: Ginamit na natin ang 'file_uploader' sa halip na 'file_upload'
uploaded_file = st.file_uploader("I-upload ang iyong larawan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    st.sidebar.header("Controls")
    size_opt = st.sidebar.selectbox("Size:", ["2x2 (600x600 px)", "1x1 (300x300 px)", "Passport (413x531 px)"])
    use_bg_rem = st.sidebar.checkbox("White Background", value=True)

    if st.button("✨ Enhance and Generate"):
        with st.spinner("Kinikinis at nillilinaw ang picture..."):
            # Step 1: Pro Enhancement
            enhanced = pro_enhance(img)
            # Step 2: Size & BG
            final = process_id_photo(enhanced, size_opt, use_bg_rem)
            
            c1, c2 = st.columns(2)
            c1.image(img, caption="Original", use_container_width=True)
            c2.image(final, caption="Enhanced Result", use_container_width=True)
            
            # Download
            buf = io.BytesIO()
            final.save(buf, format="JPEG", quality=100)
            st.download_button("📥 Download Photo", buf.getvalue(), "id_photo.jpg", "image/jpeg")
