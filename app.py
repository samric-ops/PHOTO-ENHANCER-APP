import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from rembg import remove
import google.generativeai as genai
import os

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
    .gemini-badge {
        background: linear-gradient(135deg, #4285f4, #9b72cb);
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- GEMINI AI SETUP ---
@st.cache_resource
def setup_gemini():
    """Setup Gemini AI with API key"""
    # Get API key from secrets or environment
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    return None

def analyze_image_with_gemini(image, model):
    """Use Gemini to analyze image and suggest improvements"""
    if not model:
        return None
    
    try:
        # Convert PIL to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create prompt for Gemini
        prompt = """
        Analyze this ID photo and provide specific adjustments needed:
        1. Brightness level (0-100) - current and recommended
        2. Contrast level (0-100)
        3. Skin tone issues (redness, paleness, etc.)
        4. Lighting problems (shadows, glare, etc.)
        5. Specific adjustments needed (in percentages)
        
        Format the response as JSON only:
        {
            "brightness": {"current": 0, "recommended": 0, "adjustment": 0},
            "contrast": {"current": 0, "recommended": 0, "adjustment": 0},
            "skin_tone_issues": ["issue1", "issue2"],
            "lighting_issues": ["issue1", "issue2"],
            "adjustments": {
                "brightness_boost": 0,
                "contrast_boost": 0,
                "color_correction": 0,
                "shadow_boost": 0
            }
        }
        """
        
        # Send to Gemini
        response = model.generate_content([prompt, img_byte_arr])
        
        # Parse JSON response
        import json
        try:
            # Extract JSON from response
            response_text = response.text
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            return None
            
    except Exception as e:
        st.error(f"Gemini analysis error: {str(e)}")
        return None

# --- AI ENHANCEMENT FUNCTIONS ---
def gemini_smart_enhance(image, adjustments):
    """Apply smart adjustments based on Gemini analysis"""
    
    # Convert to numpy array
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Get adjustments from Gemini
    brightness_boost = adjustments.get('adjustments', {}).get('brightness_boost', 20) / 100 + 1
    contrast_boost = adjustments.get('adjustments', {}).get('contrast_boost', 10) / 100 + 1
    color_correction = adjustments.get('adjustments', {}).get('color_correction', 0)
    shadow_boost = adjustments.get('adjustments', {}).get('shadow_boost', 15) / 100 + 1
    
    # 1. Smart Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 2. Skin Smoothing
    img = cv2.bilateralFilter(img, 5, 50, 50)
    
    # 3. Smart Brightness based on Gemini
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Boost shadows lalo na
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    yuv[:,:,0] = cv2.addWeighted(yuv[:,:,0], brightness_boost, np.zeros_like(yuv[:,:,0]), 0, 10 * shadow_boost)
    
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 4. Gamma correction (if needed)
    if shadow_boost > 1.1:
        gamma = 0.85  # Brighter
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
    
    # 5. Color correction para hindi mamula
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Adjust color channels based on Gemini
    if color_correction > 0:
        a = cv2.addWeighted(a, 1.0, np.zeros_like(a), 0, -color_correction)  # Reduce redness
        b = cv2.addWeighted(b, 1.0, np.zeros_like(b), 0, color_correction)   # Add blue
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 6. Convert to PIL for final adjustments
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 7. Apply contrast boost
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast_boost)
    
    # 8. Final brightness check
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness_boost)
    
    # 9. Sharpening
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
st.markdown("### 🤖 GEMINI AI POWERED - Smart Photo Enhancement")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key input
    with st.expander("🔑 Gemini API Settings", expanded=False):
        api_key = st.text_input("Enter Gemini API Key", type="password", 
                               help="Get API key from https://makersuite.google.com/app/apikey")
        if api_key:
            genai.configure(api_key=api_key)
            st.success("✅ API Key set!")
    
    uploaded_file = st.file_uploader("📸 Upload Photo", type=["jpg", "png", "jpeg", "jfif"])
    
    size_opt = st.selectbox("📏 Select Output Size:", 
                           ["2x2 (600x600 px)", "1x1 (300x300 px)", 
                            "Passport (413x531 px)", "PRC Size (531x531 px)"])
    
    use_bg_rem = st.checkbox("🎯 Remove Background (White)", value=True)
    
    st.markdown("---")
    st.markdown("### 🤖 AI Mode")
    
    ai_mode = st.radio("Enhancement Mode:", 
                      ["🌐 Gemini AI Smart Adjust", "⚙️ Manual Adjust"],
                      help="Gemini AI will automatically analyze and perfect your photo")
    
    if ai_mode == "⚙️ Manual Adjust":
        brightness = st.slider("Brightness", 0.8, 2.0, 1.2)
        contrast = st.slider("Contrast", 0.8, 1.5, 1.1)
    
    generate_btn = st.button("✨ Generate AI Perfect Photo", type="primary", use_container_width=True)

# Main Content
if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img)
        
        if generate_btn:
            # Setup Gemini
            model = None
            if api_key:
                model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Analyze with Gemini if in AI mode
            adjustments = None
            if ai_mode == "🌐 Gemini AI Smart Adjust" and model:
                with st.spinner("🤖 Gemini AI is analyzing your photo..."):
                    adjustments = analyze_image_with_gemini(img, model)
                    if adjustments:
                        st.success("✅ AI Analysis complete!")
                        
                        # Show analysis results
                        with st.expander("📊 AI Analysis Results"):
                            col_a1, col_a2 = st.columns(2)
                            with col_a1:
                                st.metric("Brightness", f"{adjustments.get('brightness', {}).get('current', 0)}%", 
                                        f"{adjustments.get('brightness', {}).get('adjustment', 0)}%")
                                st.metric("Contrast", f"{adjustments.get('contrast', {}).get('current', 0)}%",
                                        f"{adjustments.get('contrast', {}).get('adjustment', 0)}%")
                            with col_a2:
                                if adjustments.get('skin_tone_issues'):
                                    st.write("Skin issues:", ", ".join(adjustments['skin_tone_issues']))
                                if adjustments.get('lighting_issues'):
                                    st.write("Lighting issues:", ", ".join(adjustments['lighting_issues']))
            
            # Create three columns
            col1, col2, col3 = st.columns(3, gap="large")
            
            with col1:
                st.markdown("### 📷 Original")
                st.image(img, use_container_width=True)
                st.caption(f"Size: {img.size[0]}x{img.size[1]} px")
                
                # Calculate original brightness
                img_np = np.array(img.convert('L'))
                avg_brightness = np.mean(img_np)
                st.caption(f"Original Brightness: {avg_brightness:.1f}/255")
                
                st.markdown("<br>", unsafe_allow_html=True)

            with col2:
                st.markdown("### ✨ AI Enhanced")
                with st.spinner("Applying AI enhancements..."):
                    
                    if ai_mode == "🌐 Gemini AI Smart Adjust" and adjustments:
                        # Use Gemini adjustments
                        enhanced_img = gemini_smart_enhance(img, adjustments)
                        caption = "🤖 Gemini AI Perfect Adjust"
                    else:
                        # Manual adjustments
                        enhanced_img = gemini_smart_enhance(img, {
                            'adjustments': {
                                'brightness_boost': (brightness - 1) * 100,
                                'contrast_boost': (contrast - 1) * 100,
                                'color_correction': 0,
                                'shadow_boost': 15
                            }
                        })
                        caption = "⚙️ Manual Adjust"
                    
                    st.image(enhanced_img, use_container_width=True)
                    st.caption(caption)
                    
                    # Show enhanced brightness
                    enhanced_np = np.array(enhanced_img.convert('L'))
                    enhanced_brightness = np.mean(enhanced_np)
                    st.caption(f"Enhanced Brightness: {enhanced_brightness:.1f}/255")
                    
                    st.markdown("<br>", unsafe_allow_html=True)

            with col3:
                st.markdown("### 🎯 Final ID")
                with st.spinner("Creating final photo..."):
                    final_result = process_id_photo(enhanced_img, size_opt, use_bg_rem)
                    st.image(final_result, use_container_width=True)
                    st.caption(f"{size_opt} - AI Perfect")
                    
                    # Download button
                    buf = io.BytesIO()
                    final_result.save(buf, format="JPEG", quality=100, dpi=(300,300))
                    
                    st.download_button(
                        label="📥 Download AI Perfect ID Photo",
                        data=buf.getvalue(),
                        file_name=f"ai_perfect_id_{size_opt.split()[0]}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            # Success message
            brightness_increase = ((enhanced_brightness / avg_brightness) - 1) * 100
            st.success(f"✅ AI Perfect ID Photo! Brightness increased by {brightness_increase:.0f}%")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Upload photo to begin")
        
        st.markdown("---")
        st.markdown("### 🤖 Gemini AI Features:")
        st.markdown("""
        - 🧠 **Smart Analysis** - AI detects lighting issues
        - ✨ **Perfect Brightness** - Automatic adjustment
        - 🎨 **Natural Skin Tones** - Hindi namumula
        - 💡 **Shadow Enhancement** - Lumiwanag ang dark areas
        - 📊 **Detailed Analysis** - Kita ang improvements
        
        **Paano gamitin:**
        1. Get free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. I-paste sa sidebar
        3. Upload photo and let AI do its magic!
        """)
