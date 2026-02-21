import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from rembg import remove

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro AI ID Studio", page_icon="👤", layout="wide")

# --- PROFESSIONAL ENHANCEMENT FUNCTIONS ---
def professional_face_enhance(image):
    """Professional enhancement para magmukhang studio quality"""
    # Convert PIL to OpenCV
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Advanced Denoising (mas malinis)
    img = cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21)
    
    # 2. Skin Smoothing (parang Remini)
    # Bilateral filter para preserve edges pero smooth skin
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # 3. Smart Contrast Enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE with optimized parameters
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Enhance A and B channels para mas vibrant
    a = cv2.equalizeHist(a)
    b = cv2.equalizeHist(b)
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 4. Color Balance
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    
    # 5. Adjust brightness at contrast professionally
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.05)  # Slight brightness increase
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.1)   # Slight contrast increase
    
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(1.15)  # Slight color boost
    
    # 6. Professional Sharpening (subtle lang)
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=50, threshold=0))
    
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
    
    # AI Background Removal with professional touch
    if remove_bg:
        with st.spinner("Removing background professionally..."):
            # First remove background
            image_no_bg = remove(image)
            
            # Create white background
            white_bg = Image.new("RGB", image_no_bg.size, (255, 255, 255))
            
            # Convert to RGB kung may alpha channel
            if image_no_bg.mode == 'RGBA':
                # Paste with alpha mask
                white_bg.paste(image_no_bg, mask=image_no_bg.split()[3])
            else:
                white_bg.paste(image_no_bg)
            
            image = white_bg
    
    # Standard Sizes
    sizes = {
        "2x2 (600x600 px)": (600, 600),
        "1x1 (300x300 px)": (300, 300),
        "Passport (413x531 px)": (413, 531),
        "PRC Size (531x531 px)": (531, 531)  # Added PRC size
    }
    
    target_size = sizes.get(size_type, (600, 600))
    
    # Enhance resolution bago i-resize
    if image.size[0] < target_size[0]:
        # Upscale kung maliit ang original
        scale_factor = target_size[0] / image.size[0]
        new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return resize_with_padding(image, target_size)

# --- UI ---
st.title("👤 Pro AI ID Photo Studio")
st.markdown("### Professional ID Photo Generator - Parang Remini ang Quality")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("📸 Upload Photo (Half body is best)", type=["jpg", "png", "jpeg", "jfif"])
    
    size_opt = st.selectbox("📏 Select Output Size:", 
                           ["2x2 (600x600 px)", "1x1 (300x300 px)", 
                            "Passport (413x531 px)", "PRC Size (531x531 px)"])
    
    use_bg_rem = st.checkbox("🎯 Remove Background (White)", value=True)
    
    st.markdown("---")
    st.markdown("### 🎨 Enhancement Options")
    
    smooth_skin = st.checkbox("✨ Skin Smoothing", value=True)
    enhance_details = st.checkbox("🔍 Enhance Details", value=True)
    
    generate_btn = st.button("🎨 Generate Professional ID Photo", type="primary", use_container_width=True)

# Main Content Area
if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        
        # Fix orientation issues
        img = ImageOps.exif_transpose(img)
        
        if generate_btn:
            # Layout columns
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown("### 📷 Original")
                st.image(img, use_container_width=True)
                
                # Original size info
                st.caption(f"Original size: {img.size[0]}x{img.size[1]} px")

            # PROCESS ENHANCEMENT
            with col2:
                st.markdown("### ✨ AI Enhanced")
                with st.spinner("Applying professional enhancements..."):
                    # Professional enhancement
                    enhanced_img = professional_face_enhance(img)
                    st.image(enhanced_img, caption="After AI Enhancement", use_container_width=True)

            # FINAL RESULT
            with col3:
                st.markdown("### 🎯 Final ID Photo")
                with st.spinner("Creating final ID photo..."):
                    # Process with background removal and sizing
                    final_result = process_id_photo(enhanced_img, size_opt, use_bg_rem)
                    
                    st.image(final_result, use_container_width=True, 
                            caption=f"Final: {size_opt}")
                    
                    # Download Button with high quality
                    buf = io.BytesIO()
                    # Save with high quality for printing
                    final_result.save(buf, format="JPEG", quality=100, dpi=(300,300))
                    
                    st.download_button(
                        label="📥 Download Professional ID Photo",
                        data=buf.getvalue(),
                        file_name=f"professional_id_{size_opt.split()[0]}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
            # Success message
            st.success("✅ Professional ID photo generated successfully!")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("👈 Please upload a photo in the sidebar to begin.")
    
    # Sample display ng professional ID
    st.markdown("---")
    st.markdown("### Sample Professional Result:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("https://via.placeholder.com/300x400/ffffff/000000?text=Original+Photo", caption="Before")
    with col2:
        st.image("https://via.placeholder.com/300x400/f0f0f0/000000?text=AI+Enhanced", caption="AI Enhancement")
    with col3:
        st.image("https://via.placeholder.com/300x400/ffffff/000000?text=Final+ID", caption="Final Professional ID")
