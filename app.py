import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from rembg import remove

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro AI ID Studio", page_icon="👤", layout="wide")

# --- CUSTOM CSS PARA SA MAAYOS NA LAYOUT ---
st.markdown("""
<style>
    .stImage {
        margin-bottom: 10px;
    }
    .element-container {
        margin-bottom: 10px !important;
    }
    .stButton button {
        width: 100%;
        margin-top: 10px;
    }
    .css-1r6slb0 {
        gap: 20px;
    }
    .css-183lzff {
        margin-bottom: 15px;
        text-align: center;
    }
    hr {
        margin: 20px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- PROFESSIONAL ENHANCEMENT FUNCTIONS ---
def professional_face_enhance(image):
    """Professional enhancement na hindi namumula ang mukha"""
    # Convert PIL to OpenCV
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Light Denoising (mas light para hindi magmukhang plastic)
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 2. Skin Smoothing (very subtle)
    img = cv2.bilateralFilter(img, 5, 50, 50)
    
    # 3. Color Correction - FIX PARA HINDI MAPULA
    # Convert to LAB for better color handling
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Light CLAHE lang (para hindi mag-over contrast)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Balance ang a at b channels (para hindi mamula)
    a = cv2.addWeighted(a, 1.0, np.zeros_like(a), 0, 0)  # No change sa a channel (red-green)
    b = cv2.addWeighted(b, 1.0, np.zeros_like(b), 0, 0)  # No change sa b channel (blue-yellow)
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 4. White Balance Correction
    # Para hindi masyadong warm (yellow/red) ang dating
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    
    # 5. Gentle Adjustments (minimal lang)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.02)  # Very slight brightness
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.05)   # Slight contrast
    
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(1.0)    # NO color enhancement (para hindi mamula)
    
    # 6. Very subtle sharpening
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=0.3, percent=30, threshold=1))
    
    return pil_img

def resize_with_padding(image, target_size, fill_color=(255, 255, 255)):
    """Resize proportionally with white padding"""
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    new_image = Image.new("RGB", target_size, fill_color)
    left = (target_size[0] - image.size[0]) // 2
    top = (target_size[1] - image.size[1]) // 2
    
    new_image.paste(image, (left, top))
    return new_image

def process_id_photo(image, size_type, remove_bg):
    """Process ID photo with proper sizing and background"""
    
    if remove_bg:
        with st.spinner("Removing background..."):
            image_no_bg = remove(image)
            
            white_bg = Image.new("RGB", image_no_bg.size, (255, 255, 255))
            
            if image_no_bg.mode == 'RGBA':
                white_bg.paste(image_no_bg, mask=image_no_bg.split()[3])
            else:
                white_bg.paste(image_no_bg)
            
            image = white_bg
    
    # Standard Sizes
    sizes = {
        "2x2 (600x600 px)": (600, 600),
        "1x1 (300x300 px)": (300, 300),
        "Passport (413x531 px)": (413, 531),
        "PRC Size (531x531 px)": (531, 531)
    }
    
    target_size = sizes.get(size_type, (600, 600))
    
    if image.size[0] < target_size[0]:
        scale_factor = target_size[0] / image.size[0]
        new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return resize_with_padding(image, target_size)

# --- UI ---
st.title("👤 Pro AI ID Photo Studio")
st.markdown("### Professional ID Photo Generator - Natural Skin Tones")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("📸 Upload Photo", type=["jpg", "png", "jpeg", "jfif"])
    
    size_opt = st.selectbox("📏 Select Output Size:", 
                           ["2x2 (600x600 px)", "1x1 (300x300 px)", 
                            "Passport (413x531 px)", "PRC Size (531x531 px)"])
    
    use_bg_rem = st.checkbox("🎯 Remove Background (White)", value=True)
    
    st.markdown("---")
    st.markdown("### 🎨 Color Settings")
    
    # Add color temperature control
    color_temp = st.slider("🌡️ Color Temperature", -10, 10, 0, 
                          help="Adjust kung gusto mo ng mas cool (-) o warm (+) na skin tone")
    
    generate_btn = st.button("🎨 Generate ID Photo", type="primary", use_container_width=True)

# Main Content
if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img)
        
        if generate_btn:
            # Create three columns
            col1, col2, col3 = st.columns(3, gap="large")
            
            with col1:
                st.markdown("### 📷 Original")
                st.image(img, use_container_width=True)
                st.caption(f"Size: {img.size[0]}x{img.size[1]} px")
                st.markdown("<br>", unsafe_allow_html=True)

            with col2:
                st.markdown("### ✨ Enhanced")
                with st.spinner("Enhancing..."):
                    enhanced_img = professional_face_enhance(img)
                    
                    # Apply color temperature adjustment kung may slider
                    if color_temp != 0:
                        # Convert to numpy for color adjustment
                        enhanced_np = np.array(enhanced_img)
                        
                        if color_temp > 0:  # Warmer (less red)
                            enhanced_np[:,:,0] = np.clip(enhanced_np[:,:,0] * (1 - color_temp/100), 0, 255).astype(np.uint8)
                            enhanced_np[:,:,2] = np.clip(enhanced_np[:,:,2] * (1 + color_temp/100), 0, 255).astype(np.uint8)
                        else:  # Cooler
                            enhanced_np[:,:,0] = np.clip(enhanced_np[:,:,0] * (1 + abs(color_temp)/100), 0, 255).astype(np.uint8)
                            enhanced_np[:,:,2] = np.clip(enhanced_np[:,:,2] * (1 - abs(color_temp)/100), 0, 255).astype(np.uint8)
                        
                        enhanced_img = Image.fromarray(enhanced_np)
                    
                    st.image(enhanced_img, use_container_width=True)
                    st.caption("Natural Skin Tones")
                    st.markdown("<br>", unsafe_allow_html=True)

            with col3:
                st.markdown("### 🎯 Final ID")
                with st.spinner("Finalizing..."):
                    final_result = process_id_photo(enhanced_img, size_opt, use_bg_rem)
                    st.image(final_result, use_container_width=True)
                    st.caption(f"{size_opt}")
                    
                    # Download button
                    buf = io.BytesIO()
                    final_result.save(buf, format="JPEG", quality=100, dpi=(300,300))
                    
                    st.download_button(
                        label="📥 Download Photo",
                        data=buf.getvalue(),
                        file_name=f"id_photo_{size_opt.split()[0]}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
            st.markdown("---")
            st.success("✅ ID Photo generated with natural skin tones!")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Upload photo to begin")
