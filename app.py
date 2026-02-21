import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
from rembg import remove

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI ID Photo Studio", page_icon="📸")

def enhance_lighting(image):
    # Convert PIL to OpenCV format
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 1. CLAHE for Intelligent Brightness (hindi nasisira ang mukha)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Convert back to PIL for further sharpening
    enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))
    
    # 2. Add a bit of Sharpness para malinaw ang details
    enhancer = ImageEnhance.Sharpness(enhanced_pil)
    return enhancer.enhance(1.5)

def process_id_photo(image, size_type, remove_bg):
    # Remove Background if selected
    if remove_bg:
        image = remove(image)
        # Create white background
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, (0, 0), image)
        image = new_image.convert('RGB')

    # Standard sizes (300 DPI)
    sizes = {
        "2x2 (600x600 px)": (600, 600),
        "1x1 (300x300 px)": (300, 300),
        "Passport Size (413x531 px)": (413, 531)
    }
    
    target_size = sizes.get(size_type, (600, 600))
    return image.resize(target_size, Image.Resampling.LANCZOS)

# --- UI INTERFACE ---
st.title("📸 AI ID Photo Studio")
st.markdown("I-transform ang iyong selfie sa isang professional ID photo.")

uploaded_file = st.file_uploader("Mag-upload ng iyong picture (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    original = Image.open(uploaded_file)
    
    # Sidebar options
    st.sidebar.header("Settings")
    size_option = st.sidebar.selectbox("Piliin ang Size:", [
        "2x2 (600x600 px)", 
        "1x1 (300x300 px)", 
        "Passport Size (413x531 px)"
    ])
    
    bg_option = st.sidebar.checkbox("Gawing White Background (AI)", value=True)
    
    if st.button("Generate ID Photo"):
        with st.spinner("Processing... Hintayin lang sandali."):
            # Step 1: Enhance Light & Sharpness
            brightened = enhance_lighting(original)
            
            # Step 2: Background Removal & Resizing
            final_result = process_id_photo(brightened, size_option, bg_option)
            
            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.info("Original Preview")
                st.image(original, use_container_width=True)
            
            with col2:
                st.success(f"Enhanced {size_option}")
                st.image(final_result, use_container_width=True)
                
                # Download Button
                buf = io.BytesIO()
                final_result.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="Download Ready-to-Print Photo",
                    data=buf.getvalue(),
                    file_name=f"ID_Photo_{size_option.split()[0]}.jpg",
                    mime="image/jpeg"
                )
