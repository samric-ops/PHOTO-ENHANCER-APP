import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import io
from rembg import remove
import os
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="PRC ID Generator", page_icon="🪪", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stImage {
        margin-bottom: 10px;
    }
    .prc-preview {
        border: 2px solid #ccc;
        padding: 20px;
        border-radius: 10px;
        background: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# --- PRC ID GENERATION FUNCTIONS ---
def create_prc_id_template(photo_img, name, license_no, expiry, birthday, sex, status, height, weight):
    """Create PRC ID with proper layout"""
    
    # Create base canvas (PRC ID size: 2x2 inches at 300 DPI = 600x600)
    canvas_size = (600, 600)
    canvas = Image.new('RGB', canvas_size, color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load fonts (with fallback)
    try:
        # For deployment, use default font
        font_small = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_large = ImageFont.load_default()
    except:
        font_small = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    # Process photo (remove background, make it square)
    if photo_img:
        # Remove background
        photo_no_bg = remove(photo_img)
        
        # Create white background
        white_bg = Image.new("RGB", photo_no_bg.size, (255, 255, 255))
        if photo_no_bg.mode == 'RGBA':
            white_bg.paste(photo_no_bg, mask=photo_no_bg.split()[3])
        else:
            white_bg.paste(photo_no_bg)
        photo = white_bg
        
        # Resize photo to fit in PRC template (200x250 approx)
        photo = photo.resize((180, 220), Image.Resampling.LANCZOS)
    else:
        # Placeholder if no photo
        photo = Image.new('RGB', (180, 220), color='#cccccc')
    
    # Position photo on the right side
    canvas.paste(photo, (380, 80))
    
    # Draw PRC Header
    header_text = "REPUBLIC OF THE PHILIPPINES\nPROFESSIONAL REGULATION COMMISSION"
    draw.text((50, 30), header_text, fill='black', font=font_small)
    
    # Draw "PRC ID" prominently
    draw.text((50, 70), "PRC ID", fill='navy', font=font_large)
    
    # Draw registration number
    draw.text((50, 100), f"Registration No.: {license_no}", fill='black', font=font_small)
    
    # Draw name
    draw.text((50, 130), f"NAME: {name}", fill='black', font=font_medium)
    
    # Draw profession (default to Registered Professional)
    draw.text((50, 160), "PROFESSION: Registered Professional", fill='black', font=font_small)
    
    # Draw expiry date
    draw.text((50, 190), f"VALID UNTIL: {expiry}", fill='red', font=font_small)
    
    # Draw personal details on left side
    details_y = 250
    draw.text((50, details_y), "PERSONAL DETAILS:", fill='navy', font=font_small)
    draw.text((50, details_y + 25), f"Date of Birth: {birthday}", fill='black', font=font_small)
    draw.text((50, details_y + 50), f"Sex: {sex}", fill='black', font=font_small)
    draw.text((50, details_y + 75), f"Civil Status: {status}", fill='black', font=font_small)
    draw.text((50, details_y + 100), f"Height: {height}", fill='black', font=font_small)
    draw.text((50, details_y + 125), f"Weight: {weight}", fill='black', font=font_small)
    
    # Draw signature line
    draw.text((50, 450), "_________________________", fill='black', font=font_small)
    draw.text((70, 470), "Signature of Bearer", fill='black', font=font_small)
    
    # Draw QR code placeholder
    draw.rectangle([450, 450, 550, 550], outline='black', width=2)
    draw.text((470, 490), "QR CODE", fill='black', font=font_small)
    
    # Draw borders and lines
    draw.rectangle([0, 0, 599, 599], outline='navy', width=3)
    
    return canvas

def professional_photo_enhance(image):
    """Enhance photo for PRC ID (professional look)"""
    
    # Convert to numpy array
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Professional enhancement
    # 1. Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 2. Skin smoothing (subtle)
    img = cv2.bilateralFilter(img, 5, 50, 50)
    
    # 3. Brightness adjustment (professional level)
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
    
    # Final touch
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    return pil_img

# --- UI ---
st.title("🪪 PRC ID Professional Generator")
st.markdown("### Gaya ng PRC RICHARD.jpg - Professional ID Format")

# Sidebar for inputs
with st.sidebar:
    st.header("📝 PRC ID Information")
    
    # Photo upload
    uploaded_file = st.file_uploader("Upload Photo (1x1 or passport size)", type=["jpg", "png", "jpeg", "jfif"])
    
    st.markdown("---")
    st.subheader("Personal Information")
    
    # Name fields
    last_name = st.text_input("Last Name", "SAMORANOS")
    first_name = st.text_input("First Name", "RICHARD")
    middle_name = st.text_input("Middle Name", "PELONITA")
    full_name = f"{last_name}, {first_name} {middle_name}"
    
    # License details
    license_no = st.text_input("License No.", "01234567")
    expiry = st.date_input("Expiry Date", datetime.now())
    
    # Personal details
    col1, col2 = st.columns(2)
    with col1:
        birthday = st.date_input("Birthday", datetime(1990, 1, 1))
        sex = st.selectbox("Sex", ["Male", "Female"])
    with col2:
        status = st.selectbox("Civil Status", ["Single", "Married", "Widowed", "Separated"])
        height = st.text_input("Height", "5'6\"")
        weight = st.text_input("Weight", "65 kg")
    
    # Enhancement options
    st.markdown("---")
    st.subheader("🎨 Photo Enhancement")
    enhance_photo = st.checkbox("Enhance Photo (Professional)", value=True)
    remove_bg = st.checkbox("Remove Background", value=True)
    
    generate_btn = st.button("🪪 Generate PRC ID", type="primary", use_container_width=True)

# Main content
if uploaded_file and generate_btn:
    try:
        # Load and process photo
        photo = Image.open(uploaded_file)
        photo = ImageOps.exif_transpose(photo)
        
        # Enhance if requested
        if enhance_photo:
            with st.spinner("Enhancing photo..."):
                photo = professional_photo_enhance(photo)
        
        # Generate PRC ID
        with st.spinner("Generating PRC ID..."):
            prc_id = create_prc_id_template(
                photo_img=photo,
                name=full_name,
                license_no=license_no,
                expiry=expiry.strftime("%B %d, %Y"),
                birthday=birthday.strftime("%B %d, %Y"),
                sex=sex,
                status=status,
                height=height,
                weight=weight
            )
        
        # Display result
        st.markdown("### ✅ PRC ID Generated Successfully!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Original Photo:**")
            st.image(photo, use_container_width=True)
        
        with col2:
            st.markdown("**PRC ID Result:**")
            st.image(prc_id, use_container_width=True)
            
            # Download button
            buf = io.BytesIO()
            prc_id.save(buf, format="JPEG", quality=100, dpi=(300,300))
            
            st.download_button(
                label="📥 Download PRC ID",
                data=buf.getvalue(),
                file_name=f"PRC_ID_{last_name}_{first_name}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
        
        # Show information
        with st.expander("📋 PRC ID Details"):
            st.json({
                "name": full_name,
                "license_no": license_no,
                "expiry": expiry.strftime("%B %d, %Y"),
                "birthday": birthday.strftime("%B %d, %Y"),
                "sex": sex,
                "status": status,
                "height": height,
                "weight": weight
            })
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    # Show sample/preview
    st.markdown("### 👆 Fill up the form and upload photo to generate PRC ID")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sample PRC ID Format:**")
        st.markdown("""
        ```
        REPUBLIC OF THE PHILIPPINES
        PROFESSIONAL REGULATION COMMISSION
        PRC ID
        Registration No.: 01234567
        NAME: SAMORANOS, RICHARD PELONITA
        PROFESSION: Registered Professional
        VALID UNTIL: December 31, 2026
        
        PERSONAL DETAILS:
        Date of Birth: January 1, 1990
        Sex: Male
        Civil Status: Single
        Height: 5'6"
        Weight: 65 kg
        ```
        """)
    
    with col2:
        st.markdown("**Photo Requirements:**")
        st.markdown("""
        - 📸 White background
        - 👔 Formal attire (collar)
        - 😊 Neutral expression
        - 🎯 Front facing
        - ✨ With collar visible
        """)
