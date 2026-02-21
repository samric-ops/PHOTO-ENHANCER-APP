import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from rembg import remove

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro AI ID Studio", page_icon="👤", layout="wide")

# --- CUSTOM CSS ---
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

# --- ULTRA BRIGHT ENHANCEMENT FUNCTIONS ---
def ultra_bright_enhance(image):
    """Super bright enhancement - parang studio lighting ang dating"""
    # Convert PIL to OpenCV
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Light Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 2. Skin Smoothing
    img = cv2.bilateralFilter(img, 5, 50, 50)
    
    # 3. ULTRA BRIGHTNESS ENHANCEMENT - Multiple layers!
    
    # First pass: YUV brightness boost
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])  # Equalize
    yuv[:,:,0] = cv2.addWeighted(yuv[:,:,0], 1.4, np.zeros_like(yuv[:,:,0]), 0, 15)  # +40% brightness +15 offset
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # Second pass: Gamma correction para lumiwanag ang shadows
    gamma = 0.8  # <1 = brighter
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    
    # 4. Color Correction
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE for contrast pero i-maintain ang brightness
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 5. Convert to PIL for final adjustments
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 6. AGGRESSIVE BRIGHTNESS (x3 layers!)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.3)  # +30%
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.15)  # +15% contrast para hindi magmukhang washed out
    
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(1.0)   # Keep natural colors
    
    # 7. Additional brightness boost para sure
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.2)  # Another +20%
    
    # 8. Sharpening
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=40, threshold=0))
    
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
st.markdown("### ⚡ ULTRA BRIGHT Mode - Sobrang Liwanag (Studio Quality)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("📸 Upload Photo", type=["jpg", "png", "jpeg", "jfif"])
    
    size_opt = st.selectbox("📏 Select Output Size:", 
                           ["2x2 (600x600 px)", "1x1 (300x300 px)", 
                            "Passport (413x531 px)", "PRC Size (531x531 px)"])
    
    use_bg_rem = st.checkbox("🎯 Remove Background (White)", value=True)
    
    st.markdown("---")
    st.markdown("### 💡 Ultra Bright Control")
    
    # Extreme brightness control
    brightness_multiplier = st.slider("🔆 Brightness Level", 1.0, 2.5, 1.8, 0.1,
                                     help="Higher = Mas maliwanag (1.8 recommended)")
    
    st.markdown("---")
    st.markdown("### 📸 Preview Settings")
    show_histogram = st.checkbox("Show brightness analysis", value=False)
    
    generate_btn = st.button("✨ Generate ULTRA BRIGHT Photo", type="primary", use_container_width=True)

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
                
                # Show original brightness if requested
                if show_histogram:
                    img_np = np.array(img.convert('L'))
                    avg_brightness = np.mean(img_np)
                    st.caption(f"Original Brightness: {avg_brightness:.1f}/255")
                st.markdown("<br>", unsafe_allow_html=True)

            with col2:
                st.markdown("### ✨ Ultra Bright")
                with st.spinner("Applying ultra bright enhancement..."):
                    enhanced_img = ultra_bright_enhance(img)
                    
                    # Apply user brightness multiplier
                    if brightness_multiplier != 1.8:
                        enhancer = ImageEnhance.Brightness(enhanced_img)
                        enhanced_img = enhancer.enhance(brightness_multiplier / 1.8)
                    
                    st.image(enhanced_img, use_container_width=True)
                    st.caption("⚡ Ultra Bright - Studio Quality")
                    
                    # Show enhanced brightness
                    if show_histogram:
                        enhanced_np = np.array(enhanced_img.convert('L'))
                        avg_brightness = np.mean(enhanced_np)
                        st.caption(f"Enhanced Brightness: {avg_brightness:.1f}/255")
                    st.markdown("<br>", unsafe_allow_html=True)

            with col3:
                st.markdown("### 🎯 Final ID")
                with st.spinner("Creating final photo..."):
                    final_result = process_id_photo(enhanced_img, size_opt, use_bg_rem)
                    st.image(final_result, use_container_width=True)
                    st.caption(f"{size_opt} - Ultra Bright")
                    
                    # Download button
                    buf = io.BytesIO()
                    final_result.save(buf, format="JPEG", quality=100, dpi=(300,300))
                    
                    st.download_button(
                        label="📥 Download Ultra Bright ID Photo",
                        data=buf.getvalue(),
                        file_name=f"ultra_bright_id_{size_opt.split()[0]}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            # Success with brightness comparison
            col_success1, col_success2 = st.columns(2)
            with col_success1:
                st.success("✅ Ultra Bright ID Photo generated!")
            with col_success2:
                if show_histogram:
                    st.info(f"Brightness increased by ~{(avg_brightness/50)*100:.0f}%")
            
            # Tips
            with st.expander("💡 Tips para sa Ultra Bright na Resulta"):
                st.markdown("""
                - **Default 1.8x brightness** - Perfect for most photos
                - **Kung sobrang dilim** - Taasan pa ang slider (2.0+)
                - **Kung sobrang puti naman** - Bawasan ang slider (1.5)
                - **Natural pa rin ang skin tones** kahit sobrang bright
                """)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Upload photo to begin")
        
        st.markdown("---")
        st.markdown("### ⚡ Ultra Bright Features:")
        st.markdown("""
        - 🔆 **3x Brightness layers** - Sobrang liwanag
        - 📸 **Studio quality lighting** - Parang may professional lights
        - 🎨 **Natural colors** - Hindi nagiging putla
        - 💡 **Adjustable 1.0x to 2.5x** - Ikaw ang mag-control
        - 📊 **Brightness analyzer** - Kita ang improvement
        """)
