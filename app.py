import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="AI ID Photo Generator", layout="centered")

def enhance_image(image):
    # Convert PIL to OpenCV format
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to LAB color space para ma-adjust ang brightness (L channel)
    # Ito ang technique para lumiwanag pero hindi masira ang mukha
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge back
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

def resize_image(image, size_type):
    # Standard sizes at 300 DPI
    sizes = {
        "2x2 (600x600px)": (600, 600),
        "1x1 (300x300px)": (300, 300),
        "Passport Size (413x531px)": (413, 531)
    }
    
    target_size = sizes[size_type]
    
    # Resize keeping quality (LANCZOS)
    # Note: Mas maganda kung i-crop muna ito manually ng user, 
    # pero ito ay automatic scaling para sa prototype.
    return image.resize(target_size, Image.Resampling.LANCZOS)

# --- UI DESIGN ---
st.title("📸 AI ID Photo Generator")
st.write("I-enhance ang madidilim na picture at i-convert sa standard ID sizes.")

uploaded_file = st.file_upload("Mag-upload ng iyong picture", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(original_image, use_container_width=True)

    # Options
    size_option = st.selectbox("Piliin ang Size:", ["2x2 (600x600px)", "1x1 (300x300px)", "Passport Size (413x531px)"])
    
    if st.button("Process & Enhance Picture"):
        with st.spinner("Processing... Ginagawa nating maliwanag ang picture..."):
            # 1. Enhance Lighting
            enhanced = enhance_image(original_image)
            
            # 2. Resize
            final_photo = resize_image(enhanced, size_option)
            
            with col2:
                st.subheader("Enhanced & Resized")
                st.image(final_photo, use_container_width=True)
                
                # Download Button
                buf = io.BytesIO()
                final_photo.save(buf, format="JPEG", quality=100)
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download ID Photo",
                    data=byte_im,
                    file_name=f"id_photo_{size_option}.jpg",
                    mime="image/jpeg"
                )
