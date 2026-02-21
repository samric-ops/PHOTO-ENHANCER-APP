import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from rembg import remove

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Photo Enhancer", page_icon="📸", layout="wide")

# --- ENHANCEMENT FUNCTIONS ---
def enhance_with_gemini(image):
    """Enhance photo to look like professional studio quality"""
    
    # Convert to numpy array
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Remove noise
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 2. Skin smoothing
    img = cv2.bilateralFilter(img, 5, 50, 50)
    
    # 3. Brightness enhancement
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    yuv[:,:,0] = cv2.addWeighted(yuv[:,:,0], 1.3, np.zeros_like(yuv[:,:,0]), 0, 10)
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 4. Color correction
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # Convert back to PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Final touches
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    return pil_img

def remove_background(image):
    """Remove background and replace with white"""
    image_no_bg = remove(image)
    white_bg = Image.new("RGB", image_no_bg.size, (255, 255, 255))
    if image_no_bg.mode == 'RGBA':
        white_bg.paste(image_no_bg, mask=image_no_bg.split()[3])
    else:
        white_bg.paste(image_no_bg)
    return white_bg

def resize_to_id(image, size_type):
    """Resize to standard ID sizes"""
    sizes = {
        "2x2 (600x600)": (600, 600),
        "1x1 (300x300)": (300, 300),
        "Passport (413x531)": (413, 531),
        "PRC Size (531x531)": (531, 531)
    }
    target = sizes.get(size_type, (600, 600))
    
    # Resize proportionally
    image.thumbnail(target, Image.Resampling.LANCZOS)
    
    # Add white padding
    new_img = Image.new("RGB", target, (255, 255, 255))
    left = (target[0] - image.size[0]) // 2
    top = (target[1] - image.size[1]) // 2
    new_img.paste(image, (left, top))
    
    return new_img

# --- UI ---
st.title("📸 AI Photo Enhancer")
st.markdown("### Parang Remini - Professional Quality")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key (optional)
    if GEMINI_AVAILABLE:
        api_key = st.text_input("Gemini API Key (optional)", type="password")
        if api_key:
            genai.configure(api_key=api_key)
    
    # Upload
    uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "png", "jpeg", "jfif"])
    
    # Options
    st.markdown("---")
    st.subheader("Enhancement Options")
    
    bg_remove = st.checkbox("Remove Background (White)", value=True)
    
    size_option = st.selectbox(
        "Output Size",
        ["2x2 (600x600)", "1x1 (300x300)", "Passport (413x531)", "PRC Size (531x531)"]
    )
    
    enhance_btn = st.button("✨ Enhance Photo", type="primary", use_container_width=True)

# Main content
if uploaded_file and enhance_btn:
    try:
        # Load image
        original = Image.open(uploaded_file)
        original = ImageOps.exif_transpose(original)
        
        # Process
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(original, use_container_width=True)
            
            # Show info
            st.caption(f"Size: {original.size[0]}x{original.size[1]}")
            
            # Calculate brightness
            img_gray = np.array(original.convert('L'))
            brightness = np.mean(img_gray)
            st.caption(f"Brightness: {brightness:.1f}/255")
        
        with col2:
            st.subheader("Enhanced")
            
            with st.spinner("Enhancing... This may take a moment"):
                # Step 1: Enhance
                enhanced = enhance_with_gemini(original)
                
                # Step 2: Remove background if requested
                if bg_remove:
                    enhanced = remove_background(enhanced)
                
                # Step 3: Resize
                final = resize_to_id(enhanced, size_option)
                
                st.image(final, use_container_width=True)
                
                # Show enhanced brightness
                final_gray = np.array(final.convert('L'))
                final_brightness = np.mean(final_gray)
                st.caption(f"Enhanced Brightness: {final_brightness:.1f}/255")
                
                # Download button
                buf = io.BytesIO()
                final.save(buf, format="JPEG", quality=95)
                
                st.download_button(
                    label="📥 Download Enhanced Photo",
                    data=buf.getvalue(),
                    file_name="enhanced_photo.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
        
        # Show improvement
        improvement = ((final_brightness / brightness) - 1) * 100
        st.success(f"✅ Enhancement complete! Brightness improved by {improvement:.0f}%")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    # Welcome screen
    st.info("👆 Upload a photo and click 'Enhance Photo' to start")
    
    st.markdown("---")
    st.markdown("### Features:")
    st.markdown("""
    - ✨ **AI Enhancement** - Gaya ng Remini
    - 🎨 **Color Correction** - Natural na kulay
    - 🔆 **Brightness Boost** - Maliwanag ang mukha
    - 🖼️ **Background Removal** - White background
    - 📏 **ID Sizes** - 2x2, 1x1, Passport, PRC
    """)
