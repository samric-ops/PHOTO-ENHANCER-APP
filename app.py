import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from rembg import remove

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro AI ID Studio", page_icon="👤", layout="wide")

# --- HELPER FUNCTION: Proportional Resizing with Padding ---
def resize_with_padding(image, target_size, fill_color=(255, 255, 255)):
    """
    Resizes an image proportionally to fit within target_size,
    padding excess areas with fill_color (white).
    """
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create a new blank canvas with the target size and fill color
    new_image = Image.new("RGB", target_size, fill_color)
    
    # Calculate positions to center the resized image
    left = (target_size[0] - image.size[0]) // 2
    top = (target_size[1] - image.size[1]) // 2
    
    # Paste the proportional image onto the canvas
    new_image.paste(image, (left, top))
    return new_image

# --- MAIN PROCESSING FUNCTIONS ---
def pro_enhance(image):
    # Convert PIL to OpenCV
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Light Denoising (Cleaner look)
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    # 2. Gamma Correction (Better midtones/natural brightness)
    gamma = 1.1
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 3. CLAHE (Smart Contrast)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # Convert back to PIL
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 4. Subtle Sharpening
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    return pil_img

def process_id_photo(image, size_type, remove_bg):
    # AI Background Removal
    if remove_bg:
        with st.spinner("Removing background..."):
            image = remove(image)
            # Ensure it's RGB for the next steps by adding white bg immediately
            new_bg = Image.new("RGBA", image.size, "WHITE")
            new_bg.paste(image, (0, 0), image)
            image = new_bg.convert('RGB')

    # Standard Sizes definitions
    sizes = {
        "2x2 (600x600 px)": (600, 600),
        "1x1 (300x300 px)": (300, 300),
        "Passport (413x531 px)": (413, 531)
    }
    
    target_size = sizes.get(size_type, (600, 600))
    
    # --- THE FIX IS HERE ---
    # Instead of direct resize, we use proportional resizing with padding
    return resize_with_padding(image, target_size)

# --- UI ---
st.title("👤 Pro AI ID Photo Studio (Proportional Fix)")
st.write("Generates proportional ID photos with AI enhancement and white background.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Photo (Half body is best)", type=["jpg", "png", "jpeg"])
    size_opt = st.selectbox("Select Output Size:", ["2x2 (600x600 px)", "1x1 (300x300 px)", "Passport (413x531 px)"])
    use_bg_rem = st.checkbox("Force White Background (AI)", value=True)
    generate_btn = st.button("✨ Generate ID Photo", type="primary")

# Main Content Area
if uploaded_file:
    img = Image.open(uploaded_file)
    
    # fix orientation issues from some phone cameras
    img = ImageOps.exif_transpose(img) 

    if generate_btn:
        # Layout columns
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Original")
            st.image(img, use_container_width=True)

        # PROCESS
        with col2:
            st.subheader("Processing...")
            with st.spinner("Enhancing lighting & details..."):
                # 1. Enhance first
                enhanced_img = pro_enhance(img)
                st.image(enhanced_img, caption="AI Enhanced (Before Resize)", use_container_width=True)

        with col3:
            st.subheader("Final Proportional Result")
            with st.spinner("Finalizing size and background..."):
                # 2. Remove BG and Resize Proportionally
                final_result = process_id_photo(enhanced_img, size_opt, use_bg_rem)
                
                st.image(final_result, use_container_width=True, caption=f"Final {size_opt}")
                
                # Download Button
                buf = io.BytesIO()
                final_result.save(buf, format="JPEG", quality=95, dpi=(300,300))
                st.download_button(
                    label="📥 Download Ready-to-Print Photo",
                    data=buf.getvalue(),
                    file_name=f"proportional_id_{size_opt.split()[0]}.jpg",
                    mime="image/jpeg"
                )
else:
    st.info("👈 Please upload a photo in the sidebar to begin.")
